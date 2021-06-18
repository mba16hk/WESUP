# -*- coding: utf-8 -*-
"""
Script to Calculate weights for multiclass datasets
"""
from PIL import Image
import cv2
import numpy as np
import os
import glob
import argparse
import itertools
from image_processing import file_extension

j=os.path.join
dirname=os.path.dirname
Image.MAX_IMAGE_PIXELS = 194221750

def build_cli_parser():
    parser = argparse.ArgumentParser('Weights Function.')
    parser.add_argument('dataset_path', help='Path to dataset to crop.')
    parser.add_argument('-o', '--output', default='weights',
                        help='Path to calculated weights')
    return parser


def calculate_weights(dataset_path,destination_path ,filetype) :
    path=j(dataset_path,"masks","*."+filetype)
    
    #find the number of unique classes in the masks
    unique_classes=[]
    for file in glob.glob(path):
        image = Image.open(file)
        unique_classes.append(list(np.unique(image)))
    
    classes= list(itertools.chain(*unique_classes))
    n_classes=len(list(np.unique(classes)))
    
    #script to count the number of pixels containing each of the unique values
    n = np.zeros(n_classes, dtype=np.int64)
    imgsize=0
    for file in glob.glob(path) :
        img = cv2.imread(file)
        h,w,c=img.shape
        imgsize+=(h*w) #find the total pixel number in all masks
        for i in range(0,n_classes):
            x=sum(np.sum(img.astype(int) == i, axis=1))
            n[i]+=x[0]
    
    #Equations taken from Crowdsourcing dataset supplementary material
    ratio= [x / imgsize for x in n]
    weights=[1-x for x in ratio]
    weights[0]=0
    np.savetxt(j(path,"weights.csv"), weights, delimiter=',') 

if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()
    
    #determine the filetyoe of images
    filetype=file_extension(args.dataset_path)
    calculate_weights(args.dataset_path, args.output, filetype)
    print("Class weights successfully calculated.")