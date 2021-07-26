#!/usr/bin/env python3
"""
Inference module.
"""
import warnings
from math import ceil
from pathlib import Path

import numpy as np
import torch
import os
import torch.nn.functional as F

import argparse
from tqdm import tqdm
from PIL import Image
from skimage.morphology import opening
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

from models import initialize_trainer
from utils.data import SegmentationDataset

""" 
For this module pass the following arguments:

-k or --checkpoint, for the path to the checkpoint you want to test at
-o or --output, for the the path to save the predictions in
-N or --n_classes if the number of classes is > 2
-w or --num_workers optionally
-M or --conf_matrix as a flag to speicify if a confusion matrix should be made or not
"""

j = os.path.join

def build_cli_parser():
    parser = argparse.ArgumentParser('Inference Function.')
    parser.add_argument('dataset_path',
     help='Path to folder of dataset.')
    parser.add_argument('-N', '--n_classes', default=2, type=int,
     help='Number of object classes, a non-zero integer')
    parser.add_argument('-k', '--checkpoint', default=None,
     help='Path to checkpoint, found in RECORDS directory.')
    parser.add_argument('-o', '--output',
     help='Path to output directory')
    parser.add_argument('-w', '--num_workers', default=4, type=int,
     help='Number of workers in the dataloader, a non-zero integer')
    parser.add_argument('-M', '--conf_matrix', default="F", choices=['T', 'F'],
     help='A flag to specify if the user wants a confusion matrix.')
    parser.add_argument('-S', '--sp_segmentation', default="slic", choices=['slic', 'fz', 'q', 'w'],
     help='The type of superpixel segmentation algorithm to be used. This can be slic, fz for felzenszwalb, q for quickshift, or w for watershed.')
    parser.add_argument('-D', default=32, type=int,
     help='The dim(0) output of the classifier. Values should be integers that are powers of 2.')
    parser.add_argument('--swap0', action='store_true',
     help='Swap labels 0 and 1 (amgad dataset)')
    return parser

def predict_single_image(trainer, img, mask, output_size):
    input_, target = trainer.preprocess(img, mask.long())

    with torch.no_grad():
        pred = trainer.model(input_) #calls on WESUP(nn.Module in wesup.py, passes input_ as x)
    pred, _ = trainer.postprocess(pred, target)
    pred_desired = pred
    tensor1, tensor2 = target
    tensor1 = tensor1.squeeze(0)
    pred = pred.float().unsqueeze(0)
    pred = F.interpolate(pred, size=output_size, mode='nearest')

    return pred, tensor1, pred_desired


def predict(trainer, dataset, input_size=None, scales=(0.5,),
            num_workers=4, device='cpu', n_classes = 2, ConfMat = "T"):
    """Predict on a directory of images.

    Arguments:
        trainer: trainer instance (subclass of `models.base.BaseTrainer`)
        dataset: instance of `torch.utils.data.Dataset`
        input_size: spatial size of input image
        scales: rescale factors for multi-scale inference
        num_workers: number of workers to load data
        device: target device

    Returns:
        predictions: list of model predictions of size (H, W)
    """

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers)

    size_info = f'input size {input_size}' if input_size else f'scales {scales}'
    print(f'\nPredicting {len(dataset)} images with {size_info} ...')
    CM_tot = np.zeros((n_classes,n_classes), dtype=int)
    predictions = []
    for data in tqdm(dataloader, total=len(dataset)):
        img = data[0].to(device)
        mask = data[1].to(device).float()

        # original spatial size of input image (height, width)
        orig_size = (img.size(2), img.size(3))

        if input_size is not None:
            img = F.interpolate(img, size=input_size, mode='bilinear')
            mask = F.interpolate(mask, size=input_size, mode='nearest')
            
            prediction, target, pred_orig = predict_single_image(trainer, img, mask, orig_size)
        
        else:
            multiscale_preds = []
            for scale in scales:
                target_size = [ceil(size * scale) for size in orig_size]
                img = F.interpolate(img, size=target_size, mode='bilinear')
                mask = F.interpolate(mask, size=target_size, mode='nearest')
                pred, target, pred_orig = predict_single_image(trainer, img, mask, orig_size)
                multiscale_preds.append(pred)

            prediction = torch.cat(multiscale_preds).mean(dim=0).round()

        if ConfMat == "T":
            G = torch.flatten(target[1].cpu())
            S = torch.flatten(pred_orig.squeeze(0).cpu())
            cm = confusion_matrix(G, S)
            if cm.shape == (n_classes, n_classes):
                CM_tot += cm
        

        prediction = prediction.squeeze().cpu().numpy()

        # apply morphology postprocessing (i.e. opening) for multi-scale inference
        if input_size is None and len(scales) > 1:
            def get_selem(size):
                assert size % 2 == 1
                selem = np.zeros((size, size))
                center = int((size + 1) / 2)
                selem[center, :] = 1
                selem[:, center] = 1
                return selem
            prediction = opening(prediction, selem=get_selem(9))

        predictions.append(prediction)

    return predictions, CM_tot


def save_predictions(predictions, dataset, output_dir='predictions', n_classes = 2):
    """Save predictions to disk.

    Args:
        predictions: model predictions of size (N, H, W)
        dataset: dataset for prediction, used for naming the prediction output
        output_dir: path to output directory
    """

    print(f'\nSaving prediction to {output_dir} ...')

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    for pred, img_path in tqdm(zip(predictions, dataset.img_paths), total=len(predictions)):
        pred = pred.astype('uint8')
        if n_classes ==2 :
            Image.fromarray(pred * 255).save(output_dir / f'{img_path.stem}.png')
        else:
            #outputs multiclass predictions as low contrast masks with original classes, post-processing required
            Image.fromarray(pred).save(output_dir / f'{img_path.stem}.png')


def infer(trainer, data_dir, output_dir=None, input_size=None, 
          scales=(0.5,), num_workers=4, device='cpu', n_classes=2, ConfMat = "T"):
    """Making inference on a directory of images with given model checkpoint."""
    trainer.model.eval()
    #check what segmentation dataset does
    dataset = SegmentationDataset(data_dir, train=False, n_classes=n_classes)
    predictions, CM = predict(trainer, dataset, input_size=input_size, scales=scales,
                          num_workers=num_workers, device=device, n_classes = n_classes,
                          ConfMat = ConfMat)

    if output_dir is not None:
        save_predictions(predictions, dataset, output_dir, n_classes=n_classes)
        if ConfMat == "T":
            plot_confusion_matrix(output_dir, CM, n_classes)

    return predictions

def plot_confusion_matrix(output_dir, cm, n_classes, normalize=False,  cmap=None):

    """Confusion matrix plotting.

    Arguments:
        output_dir: Directory where the confusion matrix will be saved
        cm : The confusion matrix
        n_classes : The number of classes in the masks

    Citation: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    target_names = list(range(n_classes))
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    #plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # cbar = plt.colorbar(boundaries=np.linspace(0,1,5))
    # cbar.ax.set_title("")
    plt.ylabel('True label')
    plt.title('Confusion Matrix\nAccuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_Matrix.pdf')


def main(data_dir, model_type='wesup', checkpoint=None, output_dir=None, ConfMat = "T",
         input_size=None, scales=(0.5,), num_workers=4, device=None, n_classes=2, D=32, **kwargs):
    if output_dir is None and checkpoint is not None:
        checkpoint = Path(checkpoint)
        output_dir = checkpoint.parent.parent / 'results'
        if not output_dir.exists():
            output_dir.mkdir()
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = initialize_trainer(model_type, device=device, n_classes=n_classes, D=D, **kwargs)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)

    infer(trainer, data_dir, output_dir, input_size=input_size, ConfMat = ConfMat,
          scales=scales, num_workers=num_workers, device=device, n_classes=n_classes)

#copy the cli parser and keep only the things that we need 

if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()
    main(data_dir=args.dataset_path, model_type='wesup', checkpoint=args.checkpoint, output_dir=args.output, sp_seg=args.sp_segmentation,
         input_size=None, scales=(0.5,), num_workers=args.num_workers, device=None, n_classes=args.n_classes, D=args.D, ConfMat = args.conf_matrix)
