#!/usr/bin/env python3
"""
Segmentation Visualisation Module.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.io import imread
from skimage import color

j = os.path.join

def build_cli_parser():
    parser = argparse.ArgumentParser('Visualisation Function.')
    parser.add_argument('image_path',
     help='Path to image.')
    parser.add_argument('-o', '--output',
     help='Path to output visualisation.')
    return parser

parser = build_cli_parser()
args = parser.parse_args()

path = args.image_path
output_path = args.output

# Read the image
img = img_as_float(imread(path))

# Image segmentation methods
segments_fz = felzenszwalb(img, scale=500, sigma=0.8, min_size=200)
print("Felzenszwalb segmentation complete")
segments_slic = slic(img, n_segments=400, compactness=60, sigma=1,start_label=1)
print("SLIC segmentation complete")
segments_quick = quickshift(img, kernel_size=8, max_dist=100, ratio=0.9)
print("Quickshift segmentation complete")

# 4 square subplots with the same x and y axes
fig, ax = plt.subplots(1, 4, figsize=(11, 11), sharex=True, sharey=True)

# Plot the original image and the corresponding segmented images
ax[0].imshow(img)
ax[0].set_title("Original")
ax[1].imshow(mark_boundaries(img,segments_slic, 
            mode = 'thick', color = (0,0,0),outline_color = (0,0,0)))
ax[1].set_title(f'SLIC: {len(np.unique(segments_slic))} segments')
ax[2].imshow(mark_boundaries(img, segments_fz,
            mode = 'thick', color = (0,0,0),outline_color = (0,0,0)))
ax[2].set_title(f"Felzenszwalb: {len(np.unique(segments_fz))} segments")
ax[3].imshow(mark_boundaries(img, segments_quick, 
            mode = 'thick', color = (0,0,0),outline_color = (0,0,0)))
ax[3].set_title(f'Quickshift: {len(np.unique(segments_quick))} segments')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()
plt.savefig(j(output_path,'Segmentation_Visualisation.pdf'))
print("Visualisation saved.")