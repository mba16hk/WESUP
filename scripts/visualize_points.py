"""
Script for visualizing point annotation.
"""

import argparse
import csv
import os
import os.path as osp
import numpy as np

import cv2
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.segmentation import mark_boundaries
from joblib import Parallel, delayed

COLORS = (
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255), 
    (227,207,87),
    (102,205,170),
    (152,245,255),
    (255,97,3),
    (104,34,139),
    (255,20,147),
    (24,116,205),
    (89,89,89),
    (238,99,99),
    (238,162,173),
    (192,255,62),
    (78,238,148),
    (74,112,139),
    (238,92,66),
    (255,255,0),
    (255,0,255),
    (255,246,143),
    (238,44,44),
    (0,191,255)
)


parser = argparse.ArgumentParser()
parser.add_argument('point_root', help='Path to point labels directory')
parser.add_argument('-r', '--radius', type=int, default=5, help='Circle radius')
parser.add_argument('-o', '--output',
                    help='Output path to store visualization results')
args = parser.parse_args()

output_dir = args.output or osp.join(args.point_root, 'viz')

if not osp.exists(output_dir):
    os.mkdir(output_dir)

img_dir = osp.join(osp.dirname(args.point_root), 'images')
mask_dir = osp.join(osp.dirname(args.point_root), 'masks')

print(f'Generating dot annotation visualizaion to {output_dir} ...')


def para_func(img_name):
    basename = osp.splitext(img_name)[0]
    img = imread(osp.join(img_dir, img_name))
    mask = imread(osp.join(mask_dir, img_name)) if osp.exists(mask_dir) else None

    # handle PNG files with alpha channel
    if img.shape[-1] == 4:
        img = img[..., :3]

    # mark boundaries if mask is present
    if mask is not None:
        img = (mark_boundaries(img, mask, mode='thick') * 255).astype('uint8')

    csvfile = open(osp.join(args.point_root, f'{basename}.csv'))
    csvreader = csv.reader(csvfile)

    for point in csvreader:
        point = [int(d) for d in point]
        #arguments of cv2.circle(img, centre_coords, circle_radius, circle colour, thickness)
        cv2.circle(img, (point[0], point[1]), args.radius, COLORS[point[2]], -1)

    imsave(osp.join(output_dir, img_name), img, check_contrast=False)
    csvfile.close()


Parallel(n_jobs=os.cpu_count())(delayed(para_func)(img_name) for img_name in tqdm(os.listdir(img_dir)))
