import numpy as np
import argparse
import sys
import os
import glob
import cv2

j = os.path.join
dirname=os.path.dirname
basename=os.path.basename

def build_cli_parser():
    parser = argparse.ArgumentParser('Cropping Function.')
    parser.add_argument('dataset_path', help='Path to dataset to crop.')
    parser.add_argument('-p', '--percentage0', type = float, default=2,
     help='Path to output croppped dataset')
    return parser

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

#Remove images with class 0
def rm_class0(directory_path, percentage, filetype):
    path=j(directory_path, "masks", "*."+filetype)
    print("path",path)
    for file in glob.glob(path) :
        print("file",file)
        img = cv2.imread(file)
        h,w,c = img.shape
        imgsize = (h*w) #find the total pixel number the mask
        x=sum(np.sum(img.astype(int) == 0, axis=1)) # Find number of pixels in class 0
        if (x[0]/imgsize) > (percentage/100): #proportion of 0 in mask
            os.remove(file) #remove the mask
            os.remove(j(directory_path,"images", basename(file))) #remove the corresponding image

if __name__ == '__main__':
    parser = build_cli_parser()
    args = parser.parse_args()
    
    filetype=file_extension(args.dataset_path)

    for dir_path in os.listdir(args.dataset_path):
        
        directory_path = j(args.dataset_path, dir_path)
        print("directory path", directory_path)
        rm_class0(directory_path, args.percentage0, filetype)

    print("removed all tiled with class 0 occupying more than", args.percentage0, "percent of the image")