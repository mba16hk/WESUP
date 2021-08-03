#!/usr/bin/env python3
"""
Training module.
"""
import logging
from shutil import rmtree

from utils.metrics import accuracy, iou
from utils.metrics import dice, hausdorff
import argparse
import csv
import torch
import models

def build_cli_parser():
    parser = argparse.ArgumentParser('Training Function.')
    parser.add_argument('dataset_path',
     help='Path to folder of dataset.')
    parser.add_argument('-c', '--class_weights', default=None,
     help='Path to calculated weights')
    parser.add_argument('-e', '--epochs', default=10, type=int,
     help='Number of training epochs, a non-zero integer')
    parser.add_argument('-N', '--n_classes', default=2, type=int,
     help='Number of object classes, a non-zero integer')
    parser.add_argument('-p', '--proportion', default=1, type=float,
     help='Proportion of images to be trained on, a number between 0 and 1.')
    parser.add_argument('-b', '--batch', default=1, type=int,
     help='Batch size for training at each epoch, a non-zero integer.')
    parser.add_argument('-k', '--checkpoint', default=None,
     help='Path to checkpoint, found in RECORDS directory.')
    parser.add_argument('-r', '--rescale_factor', default=0.4, type=float,
     help='Rescaling Factor, a number between 0 and 1')
    parser.add_argument('--momentum', default=0.9, type=float,
     help='Momentum term, a number between 0 and 1')
    parser.add_argument('--lr', default=5e-5, type=float,
     help='Learning rate. Any value between 0 and 1.')
    parser.add_argument('--wd', default=0.001, type=float,
     help='Weight Decay. Any value between 0 and 1.')
    parser.add_argument('-D', default=32, type=int,
     help='The dim(0) output of the classifier. Values should be integers that are powers of 2.')
    parser.add_argument('--shift_classes', default="F", choices=["T", "F"],
     help='A flag that shifts the class label down by 1 for each class. used for the Amgad dataset with no 0 class.')
    parser.add_argument('-S', '--sp_segmentation', default="slic", choices=['slic', 'fz', 'q', 'w'],
     help='The type of superpixel segmentation algorithm to be used. This can be slic, fz for felzenszwalb, q for quickshift, or w for watershed.')
    parser.add_argument('-m', '--multiscale_range', default=None, type=float, nargs='+',
     help='multiscale_range, takes 2 numbers, where the first number passed is less than the second number. Both numbers can be any values between 0 and 1.')
    return parser

def read_class_weights(weights_file):
    results = []
    with open(weights_file) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            results.append(row)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    list_to_tensor = torch.tensor(results, dtype=torch.float, device=device).squeeze(1)
    return list_to_tensor

def fit(dataset_path, model='wesup', **kwargs):
    # Initialize logger.
    logger = logging.getLogger('Train')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    trainer = models.initialize_trainer(model, logger=logger, **kwargs)

    try:
        if kwargs.get('n_classes')>2:
            #iou calculated for multiclass data
            trainer.train(dataset_path, metrics=[accuracy, dice, iou], **kwargs)
        else:
            #iou not calculated for binary data
            trainer.train(dataset_path, metrics=[accuracy, dice], **kwargs)
    finally:
        if kwargs.get('smoke'):
            rmtree(trainer.record_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()
    
    if args.class_weights is not None:
        weights=read_class_weights(args.class_weights)
       
    else:
        weights=None

    if args.sp_segmentation=="fz":
        seg_method = "Felzenszwalb"
    elif args.sp_segmentation=="q":
        seg_method = "Quickshift"
    elif args.sp_segmentation=="w":
        seg_method = "Watershed"
    else:
        seg_method= "SLIC"

    print("Using", seg_method,"super-pixel segmentation.")
    fit(args.dataset_path, model= "wesup", class_weights=weights, n_classes=args.n_classes,
     D=args.D, sp_seg=args.sp_segmentation, weight_decay = args.wd, epochs=args.epochs, shift_classes = args.shift_classes,
     batch_size=args.batch, proportion=args.proportion, checkpoint=args.checkpoint, lr = args.lr,
     rescale_factor=args.rescale_factor, multiscale_range=args.multiscale_range, momentum=args.momentum)
