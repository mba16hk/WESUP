"""
Shuffle CRAG dataset to train/val/test sets.
"""

import argparse
import os
import glob
import warnings
from shutil import copyfile

import pandas as pd
from skimage.io import imread, imsave
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

j = os.path.join


def build_cli_parser():
    parser = argparse.ArgumentParser('Dataset generator for CRAG dataset.')
    parser.add_argument(
        'dataset_path', help='Path to original unzipped CRAG dataset.')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Validation size (between 0 and 1)')
    parser.add_argument('-o', '--output', default='data',
                        help='Path to output dataset')

    return parser


def split_train_val_test(orig_path, val_size=0.1):
    """Split image names into training set and validation set.
    """

    trains = list(glob.glob(j(orig_path, 'train/Images/*.png')))
    tests = list(glob.glob(j(orig_path, 'valid/Images/*.png')))
    y = [0] * len(trains)

    train_set, val_set, _, _ = train_test_split(
        trains, y, test_size=val_size)

    return train_set, val_set, tests


def prepare_images(orig_path, dst_path, paths):
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    dst_img_dir = j(dst_path, 'images')
    dst_mask_dir = j(dst_path, 'masks')
    os.mkdir(dst_img_dir)
    os.mkdir(dst_mask_dir)

    for path in paths:
        img_name = os.path.basename(path)
        img_dir = os.path.dirname(path)
        mask_path = j(os.path.dirname(img_dir), "Annotation")

        orig_img_path = path
        dst_img_path = j(dst_img_dir, img_name)
        orig_mask_path = j(mask_path, img_name)
        dst_mask_path = j(dst_mask_dir, img_name)

        # copy original image to destination
        copyfile(orig_img_path, dst_img_path)

        # save binarized mask to destination
        imsave(dst_mask_path, (imread(orig_mask_path) > 0).astype('uint8'))


if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()

    train_set, val_set, test_set = split_train_val_test(
        args.dataset_path, args.val_size)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    train_dir = j(args.output, 'train')
    val_dir = j(args.output, 'val')
    test_dir = j(args.output, 'test')

    prepare_images(args.dataset_path, train_dir, train_set)
    print('Training data is done.')

    prepare_images(args.dataset_path, val_dir, val_set)
    print('Validation data is done.')

    prepare_images(args.dataset_path, test_dir, test_set)
    print('Test data is done.')
