"""
Training module.
"""

import logging
from shutil import rmtree

from utils.metrics import accuracy
from utils.metrics import dice
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
    parser.add_argument('-e', '--epochs', default=2, type=int,
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
    parser.add_argument('--swap0', action="store_true",
     help='Swap labels  0 and 1 (amgad dataset)')
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
        trainer.train(dataset_path,
                      metrics=[accuracy, dice], **kwargs)
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

    fit(args.dataset_path, model= "wesup", class_weights=weights, n_classes=args.n_classes,
     epochs=args.epochs, batch_size=args.batch, proportion=args.proportion, checkpoint=args.checkpoint,
     rescale_factor=args.rescale_factor, multiscale_range=args.multiscale_range, momentum=args.momentum,swap0=args.swap0)
