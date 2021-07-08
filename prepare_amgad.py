#!/usr/bin/env python3
"""
Shuffle GlaS dataset to train/val/test sets.
"""

import argparse
import os
import warnings
import glob
from shutil import copyfile

import pandas as pd
from skimage.io import imread, imsave
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

j = os.path.join #joins the directory paths given to it


def build_cli_parser():
    parser = argparse.ArgumentParser('Dataset generator for Amgad Dataset.')
    
    #accepts directory for dataset of interest
    parser.add_argument(
        'dataset_path', help='Path to original unzipped Amgad dataset.')
    #accepts argument for validation set size, otherwise use default
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation size (between 0 and 1)')
    #accepts to output params into designated directory
    parser.add_argument('-o', '--output', default='data',
                        help='Path to output dataset')

    return parser


#orig path is the path to he directory of interest
def split_train_val_test(orig_path, val_size=0.1):
    """Split image names into training set and validation set.
    """
    #Tells the program where to look for the image set
    trains = list(glob.glob(j(orig_path, 'images/*.png')))
    y = [0] * len(trains)

    #splits the trains and val sets such that they dont overlap
    train_set, val_set, _, _ = train_test_split(
        trains, y, test_size=val_size)
    
    tests = []
    tests.extend(train_set[0:round(len(train_set)*0.25)])
    del train_set[0:round(len(train_set)*0.25)]
    print("tests:", len(tests), "trains:", len(train_set), "vals:", len(val_set))
    return train_set, val_set, tests #returns these datasets

#dst_path is the path to the destination, names are the names of the files
def prepare_images(orig_path, dst_path, paths):
    if not os.path.exists(dst_path): #if the destination path doesnt exist
        os.mkdir(dst_path)#make a directory with the destination path

    dst_img_dir = j(dst_path, 'images') #destination of the images
    dst_mask_dir = j(dst_path, 'masks') #destination of the masks
    #make directories for the images and masks
    os.mkdir(dst_img_dir) 
    os.mkdir(dst_mask_dir)

    for path in paths: #names can be training of validation sets
        
        #basename finds the last element of the path
        img_name = os.path.basename(path)
        #dirname returns the directory without the last element
        img_dir = os.path.dirname(path)
        mask_path = j(os.path.dirname(img_dir), "masks")
        
        orig_img_path = path
        dst_img_path = j(dst_img_dir, img_name)
        orig_mask_path = j(mask_path, img_name)
        dst_mask_path = j(dst_mask_dir, img_name)

        # copy original image to destination
        copyfile(orig_img_path, dst_img_path)

        # save binarized mask to destination
        copyfile(orig_mask_path, dst_mask_path)

if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()
    
    #split the training and testing data
    train_set, val_set, test_set = split_train_val_test(
        args.dataset_path, args.val_size)
    
    #if a directory for the output doesnt exist, make one instead
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    
    #The training directory will be located in that joint path
    train_dir = j(args.output, 'train')
    #The validation directory will be located in that joint path
    val_dir = j(args.output, 'val')
    #The test directory will be located in that joint path
    test_dir = j(args.output, 'test')

    prepare_images(args.dataset_path, train_dir, train_set)
    print('Training data is done.')

    prepare_images(args.dataset_path, val_dir, val_set)
    print('Validation data is done.')

    prepare_images(args.dataset_path, test_dir, test_set)
    print('Test data is done.')
