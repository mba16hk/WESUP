import os.path as osp
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_lr_finder import LRFinder
from torchvision import models
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.segmentation import slic, felzenszwalb, quickshift, watershed

from utils import empty_tensor
from utils import is_empty_tensor
from utils.data import SegmentationDataset
from utils.data import Digest2019PointDataset
from .base import BaseConfig, BaseTrainer

def _preprocess_superpixels(segments, mask=None, epsilon=1e-7):
    """Segment superpixels of a given image and return segment maps and their labels.

    Args:
        segments: slic segments tensor with shape (H, W)
        mask (optional): annotation mask tensor with shape (C, H, W). Each pixel is a one-hot
            encoded label vector. If this vector is all zeros, then its class is unknown.
    Returns:
        sp_maps: superpixel maps with shape (N, H, W)
        sp_labels: superpixel labels with shape (N_l, C), where N_l is the number of labeled samples.
    """
    # ordering of superpixels
    sp_idx_list = segments.unique()
    #print('mask unique', mask.shape, segments.shape)
    if mask is not None and not is_empty_tensor(mask) :#and (torch.max(mask)>0):
        def compute_superpixel_label(sp_idx):
            sp_mask = (mask * (segments == sp_idx).long()).float()
            return sp_mask.sum(dim=(1, 2)) / (sp_mask.sum() + epsilon)

        # compute labels for each superpixel
        sp_labels = torch.cat([
            compute_superpixel_label(sp_idx).unsqueeze(0)
            for sp_idx in range(segments.max() + 1)
        ])
        #print('sp_labels unique0', sp_labels.shape, sp_labels.sum(0), sp_labels)

        # move labeled superpixels to the front of `sp_idx_list`
        labeled_sps = (sp_labels.sum(dim=-1) > 0).nonzero().flatten()
        unlabeled_sps = (sp_labels.sum(dim=-1) == 0).nonzero().flatten()
        sp_idx_list = torch.cat([labeled_sps, unlabeled_sps])
       
        # quantize superpixel labels (e.g., from (0.7, 0.3) to (1.0, 0.0))
        sp_labels = sp_labels[labeled_sps]
        sp_labels = (sp_labels == sp_labels.max(
            dim=-1, keepdim=True)[0]).float() #This is a 1 hot encoded vector of sp (finding the maximum)
        #print('sp_labels unique2', sp_labels.sum(0))
    else:  # no supervision provided
        sp_labels = empty_tensor().to(segments.device)

    # stacking normalized superpixel segment maps
    sp_maps = segments == sp_idx_list[:, None, None]
    sp_maps = sp_maps.squeeze().float()

    # make sure each superpixel map sums to one
    sp_maps = sp_maps / sp_maps.sum(dim=(1, 2), keepdim=True)
    return sp_maps, sp_labels


def _cross_entropy(y_hat, y_true, class_weights=None, epsilon=1e-7):
    """Semi-supervised cross entropy loss function.

    Args:
        y_hat: prediction tensor with size (N, C), where C is the number of classes
        y_true: label tensor with size (N, C). A sample won't be counted into loss
            if its label is all zeros.
        class_weights: class weights tensor with size (C,)
        epsilon: numerical stability term

    Returns:
        cross_entropy: cross entropy loss computed only on samples with labels
    """
    device = y_hat.device

    # clamp all elements to prevent numerical overflow/underflow
    y_hat = torch.clamp(y_hat, min=epsilon, max=(1 - epsilon))

    # number of samples with labels
    labeled_samples = torch.sum(y_true.sum(dim=1) > 0).float()

    if labeled_samples.item() == 0:
        return torch.tensor(0.).to(device)

    ce = -y_true * torch.log(y_hat)
    if class_weights is not None:
        ce = ce * class_weights
        #ce = ce * class_weights.unsqueeze(0).float()
        
    return torch.sum(ce) / labeled_samples


def _label_propagate(features, y_l, threshold=0.95):
    """Perform random walk based label propagation with similarity graph.

    Arguments:
        features: features of size (N, D), where N is the number of superpixels
            and D is the dimension of input features
        y_l: label tensor of size (N, C), where C is the number of classes
        threshold: similarity threshold for label propagation

    Returns:
        pseudo_labels: propagated label tensor of size (N, C)
    """

    # disable gradient computation
    features = features.detach()
    y_l = y_l.detach()

    # number of labeled and unlabeled samples
    n_l = y_l.size(0)
    n_u = features.size(0) - n_l

    # feature affinity matrix
    W = torch.exp(-torch.einsum('ijk,ijk->ij',
                                features - features.unsqueeze(1),
                                features - features.unsqueeze(1)))

    # sub-matrix of W containing similarities between labeled and unlabeled samples
    W_ul = W[n_l:, :n_l]

    # max_similarities is the maximum similarity for each unlabeled sample
    # src_indexes is the respective labeled sample index
    max_similarities, src_indexes = W_ul.max(dim=1)

    # initialize y_u with zeros
    y_u = torch.zeros(n_u, y_l.size(1)).to(y_l.device)

    # only propagate labels if maximum similarity is above the threshold
    propagated_samples = max_similarities > threshold
    y_u[propagated_samples] = y_l[src_indexes[propagated_samples]]

    return y_u


class WESUPConfig(BaseConfig):
    """Configuration for WESUP model."""

    # Rescale factor to subsample input images.
    rescale_factor = 0.5

    # multi-scale range for training
    multiscale_range = (0.3, 0.4)

    # Number of target classes.
    n_classes = 2

    # Class weights for cross-entropy loss function.
    class_weights= None

    # Superpixel parameters.
    sp_area = 150 #200
    sp_compactness = 60 #40

    # whether to enable label propagation
    enable_propagation = True

    # Weight for label-propagated samples when computing loss function
    propagate_threshold = 0.8

    # Weight for label-propagated samples when computing loss function
    propagate_weight = 0.5

    # Optimization parameters.
    momentum = 0.9
    weight_decay = 0.001
    lr = 5e-5

    # Whether to freeze backbone.
    freeze_backbone = False

    # Training configurations.
    batch_size = 1
    epochs = 300


class WESUP(nn.Module):
    """Weakly supervised histopathology image segmentation with sparse point annotations."""

    def __init__(self, n_classes=2, D=64, **kwargs):
        """Initialize a WESUP model.

        Kwargs:
            n_classes: number of target classes (default to 2)
            D: output dimension of superpixel features

        Returns:
            model: a new WESUP model
        """

        super().__init__()

        self.kwargs = kwargs
        self.backbone = models.vgg16(pretrained=True).features

        # sum of channels of all feature maps
        self.fm_channels_sum = 0

        # side convolution layers after each conv feature map
        for layer in self.backbone:
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._hook_fn)
                setattr(self, f'side_conv{self.fm_channels_sum}',
                        nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1))
                self.fm_channels_sum += layer.out_channels // 2
        
        # fully-connected layers for dimensionality reduction
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fm_channels_sum, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, D),
            nn.ReLU()
        )

        # final softmax classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(D, D // 2),
        #     nn.ReLU(),
        #     nn.Linear(D // 2, n_classes)),
        #     nn.Softmax(dim=1)
        # )
        self.n_classes=n_classes
        self.classifier = nn.Sequential(
            nn.Linear(D, n_classes), #applies a linear transformation on incoming dataset
            nn.Softmax(dim=1) #normalise across the first dimension (each channel has a prob dist for belonging to each class)
        )
        #print('No of classes in wesup.py 2',n_classes)
        # store conv feature maps
        self.feature_maps = None

        # spatial size of first feature map
        self.fm_size = None

        # label propagation input features
        self.sp_features = None

        # superpixel predictions (tracked to compute loss)
        self.sp_pred = None

    def _hook_fn(self, _, input_, output):
        if self.feature_maps is None:
            self.fm_size = (input_[0].size(2), input_[0].size(3))
            side_conv_name = 'side_conv0'
        else:
            side_conv_name = f'side_conv{self.feature_maps.size(0)}'

        output = getattr(self, side_conv_name)(output.clone())
        output = F.interpolate(output, self.fm_size,
                               mode='bilinear', align_corners=True)

        if self.feature_maps is None:
            self.feature_maps = output.squeeze()
        else:
            self.feature_maps = torch.cat(
                (self.feature_maps, output.squeeze()))

    def forward(self, x):
        """Running a forward pass.

        Args:
            x: a tuple containing input tensor of size (1, C, H, W) and
                stacked superpixel maps with size (N, H, W)

        Returns:
            pred: prediction with size (1, H, W)
        """
        #x would be the imported RGB image of size (1,N,H,W). Values range between 0-1
        x, sp_maps = x
        n_superpixels, height, width = sp_maps.size()

        # extract conv feature maps and flatten
        self.feature_maps = None
        _ = self.backbone(x) #uses the vgg16 backbone to creature feature_maps

        x = self.feature_maps #x is a 3D tensor
        x = x.view(x.size(0), -1) #x is collapsed to a 2D tensor

        # calculate features for each superpixel
        sp_maps = sp_maps.view(sp_maps.size(0), -1) #sp_maps collapsed to a 2D tensor

        #torch.mm is the matrix multiplication between the 2 matrices between the brackets
        x = torch.mm(sp_maps, x.t()) #x.t() is the transpose of the 2D x tensor

        # reduce superpixel feature dimensions with fully connected layers
        x = self.fc_layers(x)
        self.sp_features = x

        # classify each superpixel, self.pred would be of size [n_superpixels, N]
        self.sp_pred = self.classifier(x)

        # flatten sp_maps to one channel
        sp_maps = sp_maps.view(n_superpixels, height, width).argmax(dim=0)

        # initialize prediction mask
        pred = torch.zeros(height, width, self.sp_pred.size(1))
        pred = pred.to(sp_maps.device)
        for sp_idx in range(sp_maps.max().item() + 1):
            pred[sp_maps == sp_idx] = self.sp_pred[sp_idx]

        # Code for multi-class classification
        if self.n_classes==2:
            # for binary masks
            pred=pred.unsqueeze(0)[..., 1]
        else:
            # for multi-class masks
            vals, indicies=pred.max(2)
            pred= indicies.unsqueeze(0)

        return pred


class WESUPPixelInference(WESUP):
    """Weakly supervised histopathology image segmentation with sparse point annotations."""

    def __init__(self, n_classes=2, D=64, **kwargs):
        """Initialize a WESUP model.

        Kwargs:
            n_classes: number of target classes (default to 2)
            D: output dimension of superpixel features

        Returns:
            model: a new WESUP model
        """

        super().__init__()

        self.kwargs = kwargs
        self.backbone = models.vgg16(pretrained=True).features
        
        # sum of channels of all feature maps
        self.fm_channels_sum = 0

        # side convolution layers after each conv feature map
        for layer in self.backbone:
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self._hook_fn)
                setattr(self, f'side_conv{self.fm_channels_sum}',
                        nn.Conv2d(layer.out_channels, layer.out_channels // 2, 1))
                self.fm_channels_sum += layer.out_channels // 2

        # fully-connected layers for dimensionality reduction
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fm_channels_sum, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, D),
            nn.ReLU()
        )

        # final softmax classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(D, D // 2),
        #     nn.ReLU(),
        #     nn.Linear(D // 2, self.kwargs.get('n_classes', 2)),
        #     nn.Softmax(dim=1)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(D, n_classes),
            nn.Softmax(dim=1)
        )

        # store conv feature maps
        self.feature_maps = None

        # spatial size of first feature map
        self.fm_size = None

    def _hook_fn(self, _, input_, output):
        if self.feature_maps is None:
            self.fm_size = (input_[0].size(2), input_[0].size(3))
            side_conv_name = 'side_conv0'
        else:
            side_conv_name = f'side_conv{self.feature_maps.size(0)}'

        output = getattr(self, side_conv_name)(output.clone())
        output = F.interpolate(output, self.fm_size,
                               mode='bilinear', align_corners=True)

        if self.feature_maps is None:
            self.feature_maps = output.squeeze()
        else:
            self.feature_maps = torch.cat(
                (self.feature_maps, output.squeeze()))

    def forward(self, x):
        """Running a forward pass.

        Args:
            x: input image tensor of size (1, 3, H, W)

        Returns:
            pred: prediction with size (H, W, C)
        """

        height, width = x.size()[-2:]

        self.feature_maps = None
        _ = self.backbone(x)
        x = self.feature_maps
        x = x.view(x.size(0), -1)
        x = self.classifier(self.fc_layers(x.t()))

        return x.view(height, width, -1)


class WESUPTrainer(BaseTrainer):
    """Trainer for WESUP."""

    def __init__(self, model, **kwargs):
        """Initialize a WESUPTrainer instance.

        Kwargs:
            rescale_factor: rescale factor to subsample input images
            multiscale_range: multi-scale range for training
            class_weights: class weights for cross-entropy loss function
            sp_area: area of each superpixel
            sp_compactness: compactness parameter of SLIC
            enable_propagation: whether to enable label propagation
            propagate_threshold: threshold for label propagation
            propagate_weight: weight for label-propagated samples in loss function
            momentum: SGD momentum
            weight_decay: weight decay for optimizer
            lr: learning rate for optimizer
            freeze_backbone: whether to freeze backbone

        Returns:
            trainer: a new WESUPTrainer instance
        """
        
        config = WESUPConfig()
        if config.freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
        kwargs = {**config.to_dict(), **kwargs}
        super().__init__(model, **kwargs)
        # cross-entropy loss function
        self.xentropy = partial(_cross_entropy, class_weights=kwargs.get('class_weights'))

    def get_default_dataset(self, root_dir, train=True, proportion=1.0):
        if train:
            if osp.exists(osp.join(root_dir, 'points')):
                #Enters this only if a points directory exists
                return Digest2019PointDataset(root_dir, proportion=proportion, 
                                              multiscale_range=self.kwargs.get('multiscale_range'),
                                              rescale_factor=self.kwargs.get('rescale_factor'))
            return SegmentationDataset(root_dir, proportion=proportion,
                                       multiscale_range=self.kwargs.get('multiscale_range'),
                                       rescale_factor=self.kwargs.get('rescale_factor'),
                                       n_classes=self.kwargs.get('n_classes'),
                                       swap0=self.kwargs.get('swap0'))
        return SegmentationDataset(root_dir, rescale_factor=self.kwargs.get('rescale_factor'), train=False,
                                   n_classes=self.kwargs.get('n_classes'),
                                   swap0=self.kwargs.get('swap0'))

    def get_default_optimizer(self):
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.kwargs.get('lr'),
            momentum=self.kwargs.get('momentum'),
            weight_decay=self.kwargs.get('weight_decay'),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)

        return optimizer, None
    
    def LR_finder(self, optimizer):
        optimizer, _ = get_default_optimizer(self)
        criterion = nn.CrossEntropyLoss()
        lr_finder = LRFinder(self.model, optimizer, criterion, device=self.device)

    def preprocess(self, *data):
        data = [datum.to(self.device) for datum in data]
        if len(data) == 3:
            img, pixel_mask, point_mask = data
        elif len(data) == 2:
            img, pixel_mask = data
            point_mask = empty_tensor()
        elif len(data) == 1:
            img, = data
            point_mask = empty_tensor()
            pixel_mask = empty_tensor()
        else:
            raise ValueError('Invalid input data for WESUP')

        # if img.shape[0]>1:
        #     for i in range(0,img.shape[0]):

        if self.kwargs.get('sp_seg') =='slic':
            segments = slic(
            img.squeeze().cpu().numpy().transpose(1, 2, 0),
            n_segments=int(img.size(-2) * img.size(-1) /
                           self.kwargs.get('sp_area')),
            compactness=self.kwargs.get('sp_compactness')#,sigma = 0.3
            )

            #print('slic segments', np.unique(segments))

        elif self.kwargs.get('sp_seg') =='fz':
            
            segments = felzenszwalb(img.squeeze().cpu().numpy().transpose(1, 2, 0), 
            scale=20, sigma=0.8, min_size=30)
            
            #print('fz segments', segments.shape, len(np.unique(segments)))

        elif self.kwargs.get('sp_seg') =='q':

            quickshift_image = img
            quickshift_image = quickshift_image.type(torch.DoubleTensor)
            segments = quickshift(quickshift_image.squeeze().cpu().numpy().transpose(1, 2, 0),
             kernel_size=3, max_dist=6, ratio=0.5)

            #print('quickshift segments size and shape', segments.shape, len(np.unique(segments)))

        elif self.kwargs.get('sp_seg') =='w':
            # Not working, throws ValueError: loss is nan
            gradient = sobel(rgb2gray(img.squeeze().cpu().numpy().transpose(1, 2, 0)))
            segments = watershed(gradient, markers=500, compactness=0.001)

            #print('watershed segments size and shape', segments.shape, len(np.unique(segments)))


        segments = torch.as_tensor(
            segments, dtype=torch.long, device=self.device)

        if point_mask is not None and not is_empty_tensor(point_mask):
            mask = point_mask.squeeze()
        elif pixel_mask is not None and not is_empty_tensor(pixel_mask):
            mask = pixel_mask.squeeze()
            #print('pixel mask', mask.sum(1).sum(1))
        else:
            mask = None

        sp_maps, sp_labels = _preprocess_superpixels(
            segments, mask, epsilon=self.kwargs.get('epsilon'))

        return (img, sp_maps), (pixel_mask, sp_labels)

    def compute_loss(self, pred, target, metrics=None):
        _, sp_labels = target
        
        

        sp_features = self.model.sp_features
        sp_pred = self.model.sp_pred
        print(sp_pred.shape)

        if sp_pred is None:
            raise RuntimeError(
                'You must run a forward pass before computing loss.')

        # total number of superpixels
        total_num = sp_pred.size(0)
        print('tot superpixels',total_num)

        # number of labeled superpixels
        labeled_num = sp_labels.size(0) #number of rows
        print('labeled_superpixels',labeled_num)

        if labeled_num < total_num:
            # weakly-supervised mode
            loss = self.xentropy(sp_pred[:labeled_num], sp_labels)

            if self.kwargs.get('enable_propagation'):
                propagated_labels = _label_propagate(sp_features, sp_labels,
                                                     threshold=self.kwargs.get('propagate_threshold'))

                propagate_loss = self.xentropy(
                    sp_pred[labeled_num:], propagated_labels)
                loss += self.kwargs.get('propagate_weight') * propagate_loss

            if metrics is not None and isinstance(metrics, dict):
                metrics['labeled_sp_ratio'] = labeled_num / total_num
                if self.kwargs.get('enable_propagation'):
                    metrics['propagated_labels'] = propagated_labels.sum().item()
                    metrics['propagate_loss'] = propagate_loss.item()
            print("weakly supervised")
        else:  # fully-supervised mode
            print("fully supervised")
            loss = self.xentropy(sp_pred, sp_labels)


        # clear outdated superpixel prediction
        self.model.sp_pred = None

        return loss

    def postprocess(self, pred, target=None):
        if self.kwargs.get('n_classes')==2:
            pred = pred.round().long()

        if target is not None:
            return pred, target[0].argmax(dim=1)
        return pred

    def post_epoch_hook(self, epoch):
        if self.scheduler is not None:
            labeled_loss = np.mean(self.tracker.history['loss'])

            # only adjust learning rate according to loss of labeled examples
            if 'propagate_loss' in self.tracker.history:
                labeled_loss -= np.mean(self.tracker.history['propagate_loss'])

            self.scheduler.step(labeled_loss)
