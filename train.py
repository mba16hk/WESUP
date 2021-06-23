"""
Training module.
"""

import logging
from shutil import rmtree

import fire

#from models import initialize_trainer
import models
from utils.metrics import accuracy
from utils.metrics import dice
import argparse
import csv
import torch

def build_cli_parser():
    parser = argparse.ArgumentParser('Training Function.')
    parser.add_argument('dataset_path', help='Path to folder of dataset.')
    parser.add_argument('-c', '--class_weights', default=None, help='Path to calculated weights')
    parser.add_argument('-e', '--epochs', default=2, type=int, help='Number of training epochs')
    parser.add_argument('-N', '--n_classes', default=2, type=int, help='Number of object classes')
    parser.add_argument('-p', '--proportion', default=1, type=float, help='Proportion of images to be trained on.')
    parser.add_argument('-b', '--batch', default=1, type=int, help='Batch size for training at each epoch.')
    return parser

def read_class_weights(weights_file):
    results = []
    with open(weights_file) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            results.append(row)
    list_to_tensor = torch.FloatTensor(results)
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
        weights=3,1
    
    fit(args.dataset_path, model= "wesup", class_weights=weights, n_classes=args.n_classes,
     epochs=args.epochs, batch_size=args.batch, proportion=args.proportion)
    #fire.Fire(fit)
