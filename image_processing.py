#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to reduce the size of datasets with big images.
"""
from PIL import Image
import numpy as np
import argparse
import sys
import os
import glob
import math
import image_slicer
import shutil
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize
import warnings


warnings.filterwarnings("ignore")
j = os.path.join
split = os.path.splitext
dirname=os.path.dirname
Image.MAX_IMAGE_PIXELS = 194221750


def build_cli_parser():
    parser = argparse.ArgumentParser('Cropping Function.')
    parser.add_argument('dataset_path', help='Path to dataset to crop.')
    parser.add_argument('-o', '--output', default='cropped_data',
     help='Path to output croppped dataset')
    parser.add_argument('-r','--reduce', choices=['t', 'c', 's'],
     help='Type t for tiling, c for cropping, and s for rescaling.')
    parser.add_argument('-T','--target_pixels', type=int, default=200000000,
     help='The maximum height or width of the image above which image rescaling happens.')
    parser.add_argument('-N','--tile_number', type=float , default = 'None',
     help='If tiling, the image will be chopped into NxN tiles. If cropping, the image will be chopped into NxN pixels. If rescaling, N would be any value greater than 0 and less than or equal to 1.')
    return parser

def rescale_images(orig_path, dst_path, filetype, N) :
    output_path = dst_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for directory in os.listdir(orig_path) :
        dst_main_dir = j(output_path, directory)    
        if not os.path.exists(dst_main_dir):
            os.mkdir(dst_main_dir)
        
        if os.path.isdir(j(orig_path,directory)) :
            original_path=j(orig_path,directory)
            
        for path in os.listdir(original_path) :
            if os.path.isdir(j(original_path,path)):
                dst_sub_dir=j(dst_main_dir,path)
                if not os.path.exists(dst_sub_dir):
                    os.mkdir(dst_sub_dir)
                x='*.'+filetype
                for file in glob.glob(j(original_path,path,x)):
                    im= io.imread(file)
                    image_rescaled = resize(im, (im.shape[0]//N, im.shape[1]//N), anti_aliasing=True, 
                                                 mode='constant')
                    filename='resc'+os.path.basename(file)
                    destination=j(dst_sub_dir, filename)
                    io.imsave(destination, image_rescaled)
            else:
                for file in glob.glob(j(original_path,path)):
                    im= io.imread(file)
                    image_rescaled = resize(im, (im.shape[0]//N, im.shape[1]//N), anti_aliasing=True, 
                                                 mode='constant')
                    filename='resc'+os.path.basename(file)
                    destination=j(dst_main_dir, filename)
                    io.imsave(destination, image_rescaled)
                        
#rescales an image in a separate directory
def resize_image_sep_dir(rsc_dir,file, scale): 
    im= io.imread(file)
    scale=round(1/scale)
    rescaled = resize(im, (im.shape[0]//scale, im.shape[1]//scale), 
                      anti_aliasing=True, mode='constant')
    filename='resc'+os.path.basename(file)
    destination=j(rsc_dir, filename)
    io.imsave(destination, rescaled)
    rs_im=destination
    
    return rs_im

#scales and tiles the images based on the input params
def scale_and_tile(target_pixels,file, dst_main_dir,filetype, N):
    im=Image.open(file)
    w, h = im.size
    resc_img_dir= j(dst_main_dir,'rescaled_images')
    if not os.path.exists(resc_img_dir):
        os.mkdir(resc_img_dir)
    if w>target_pixels or h>target_pixels:
        w_adjuster=target_pixels/w
        h_adjuster=target_pixels/h
        scale=(w_adjuster+h_adjuster)/2
        rs_im = resize_image_sep_dir(resc_img_dir, file, scale)
    else:
        rs_im=file
        
    if N==0 :
        im=Image.open(rs_im)
        w, h = im.size
        if w > 400 or h > 400:
            if w > 400:
                Nw=math.ceil(w/400)
            else:
                Nw = 1
            if h > 400:
                Nh=math.ceil(h/400)
            else:
                Nh = 1
            tiles=image_slicer.slice(rs_im,row=Nh,
                                     col=Nw,save= False)
        else: 
            tiles = rs_im
        
    else :
        tiles=image_slicer.slice(rs_im,row=N,
                                     col=N,save= False)
            
    return tiles

#crops images to correct sizes
def crop_images(orig_path, dst_path, filetype, N):
    output_path=dst_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for directory in os.listdir(orig_path) :
        dst_main_dir = j(output_path, directory)    
        if not os.path.exists(dst_main_dir):
            os.mkdir(dst_main_dir)
        
        if os.path.isdir(j(orig_path,directory)) :
            original_path=j(orig_path,directory)

        for path in os.listdir(original_path) :

            if os.path.isdir(j(original_path,path)):
                dst_sub_dir=j(dst_main_dir,path)
                if not os.path.exists(dst_sub_dir):
                    os.mkdir(dst_sub_dir)
                x='*.'+filetype
                for file in glob.glob(j(original_path,path,x)):
                    im=Image.open(file)
                    w, h = im.size
                    change_w=w-N
                    change_h=h-N
                    if (change_w%2==0) :
                        x=(change_w/2)
                        diff_w=np.repeat(x,2)
                    else :
                        x=((change_w+1)/2)
                        diff_w=[x,x-1]

                    if (change_h%2==0) :
                        x=(change_h/2)
                        diff_h=np.repeat(x,2)
                    else :
                        x=((change_h+1)/2)
                        diff_h=[x,x-1]
                    cropped_image=im.crop((diff_w[0], diff_h[0], 
                                           w-diff_w[1], h-diff_h[1]))
                    filename=os.path.basename(file)
                    destination=j(output_path,directory, path, filename)
                    cropped_image.save(destination, filetype)
            else:
                for file in glob.glob(j(original_path,path)):
                    im=Image.open(file)
                    w, h = im.size
                    change_w=w-N
                    change_h=h-N
                    if (change_w%2==0) :
                        x=(change_w/2)
                        diff_w=np.repeat(x,2)
                    else :
                        x=((change_w+1)/2)
                        diff_w=[x,x-1]

                    if (change_h%2==0) :
                        x=(change_h/2)
                        diff_h=np.repeat(x,2)
                    else :
                        x=((change_h+1)/2)
                        diff_h=[x,x-1]
                    cropped_image=im.crop((diff_w[0], diff_h[0], 
                                           w-diff_w[1], h-diff_h[1]))
                    filename=os.path.basename(file)
                    destination=j(output_path,directory, path, filename)
                    cropped_image.save(destination, filetype)

#Tiling function with uses scale_and_tile and resize_img_sep_dir   
def tile_images(orig_path, dst_path, filetype, N, target_pixels) :
    output_path=dst_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    for directory in os.listdir(orig_path):
        dst_main_dir = j(output_path, directory)
        if not os.path.exists(dst_main_dir):
            os.mkdir(dst_main_dir)
        
        if os.path.isdir(j(orig_path,directory)) :
            original_path = j(orig_path,directory)
            
        for path in os.listdir(original_path) :
            if os.path.isdir(j(original_path,path)):
                dst_sub_dir=j(dst_main_dir,path)
                if not os.path.exists(dst_sub_dir):
                    os.mkdir(dst_sub_dir)
                x='*.'+filetype
                for file in glob.glob(j(original_path,path,x)):
                    tiles=scale_and_tile(target_pixels,file, dst_sub_dir, filetype, N)
                    filename=split(os.path.basename(file))
                    output_prefix="sl_" + filename[0]
                    destination= dst_sub_dir
                    if type(tiles) == tuple:
                        image_slicer.save_tiles(tiles, directory=destination, 
                                            prefix=output_prefix, format=filetype)
                    else:
                        destination=j(dst_sub_dir, output_prefix+'.'+filetype)
                        print(destination)
                        img=Image.open(file)  
                        img.save(destination)
                    if os.path.exists(j(dst_sub_dir,"rescaled_images")):
                        shutil.rmtree(j(dst_sub_dir,"rescaled_images"), ignore_errors=True)
            else:
                for file in glob.glob(j(original_path,path)):
                    tiles=scale_and_tile(target_pixels,file, dst_main_dir, filetype, N)
                    filename=split(os.path.basename(file))
                    output_prefix="sl_" + filename[0]
                    destination= dst_main_dir
                    if type(tiles) == tuple:
                        image_slicer.save_tiles(tiles, directory=destination, 
                                            prefix=output_prefix, format=filetype)
                    else:
                        destination=j(dst_main_dir, output_prefix+'.'+filetype)
                        print(destination)
                        img=Image.open(file)  
                        img.save(destination)
                
                if os.path.exists(j(dst_main_dir,"rescaled_images")):
                    shutil.rmtree(j(dst_main_dir,"rescaled_images"), ignore_errors=True)
                        
# Function that determines the file extension of the images in the dataset                        
def file_extension(orig_path) :
    ListFiles = os.walk(orig_path)
    SplitTypes = []
    for walk_output in ListFiles:
        for file_name in walk_output[-1]:
            SplitTypes.append(file_name.split(".")[-1])
    x=list(set(SplitTypes))
    if (len(x)>1) :
        print('DATA PROCESSING ERROR: Data file not prepared properly. Check the output from prepare_*.py')
    elif (len(x)==1) :
        file_ext=x[0]
    return file_ext
            
if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()
    
    if not os.path.exists(args.output) :
        os.mkdir(args.output)
    
    filetype=file_extension(args.dataset_path)
    
    if args.reduce == 't' :
        tile_images(args.dataset_path, args.output, filetype, args.tile_number, args.target_pixels)
        print('Finished Tiling Dataset. Tiled dataset is ready.')
    elif args.reduce == 'c' :
        crop_images(args.dataset_path, args.output, filetype, args.tile_number)
        print('Finished Cropped Dataset. Cropped dataset is ready.')
    elif args.reduce == 's' :
        rescale_images(args.dataset_path, args.output, filetype, args.tile_number)
        print('Finished Downsampling Dataset. Downsampled dataset is ready.')
