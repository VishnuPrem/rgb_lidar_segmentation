import glob
import os
import PIL.Image
import numpy as np
import cv2
import pdb
import glob

import sys
sys.path.append('../')
from config import *

import configparser

import numpy as np

import torch

def load_class_weights():
    config = configparser.ConfigParser()
    config.sections()
    config.read(ARGS_TRAIN_DIR + "/class_weights.ini" )
    weights_mat = torch.ones([1,NUM_CLASSES])
    weights_mat[0,0] = float(config['ClassWeights']['background'])
    weights_mat[0,1] = float(config['ClassWeights']['Car'])
    weights_mat[0,2] = float(config['ClassWeights']['Pedestrain'])
    weights_mat[0,3] = float(config['ClassWeights']['Bicycle'])

    num_images = float(config['ClassWeights']['num_images'])
 
    return weights_mat.squeeze(), num_images

def calculate_class_weights(labels, n_classes, method = "paszke", c = 1.02):
    idx, counts = np.unique(labels, return_counts=True)
    n_pixels = labels.size
    p_class = np.zeros(n_classes)
    p_class[idx] = counts/n_pixels

    if method == "paszke":
        weights = 1/np.log(c+p_class)
    elif method in {"eigen", "logeigen2"}:
        epsilon = 1e-8
        median = np.median(p_class)
        weights = median/(p_class+epsilon)
        if method == "logeigen2":
            weights = np.log(weights+1)
    else:
        assert False, "Method of weights calculation doesnt exist"
    return weights

def main():
    #train_data_labels = os.path.join(ARGS_TRAIN_DIR,"labels/")
    #label_list = glob.glob(train_data_labels + "*.png")
    with open('/home/neil/squeezeSeg/Camvid/CamVid/train.txt') as f:
            label_list = []
            for fi in f.readlines():
                
                label_list = label_list + ['/home/neil/squeezeSeg/Camvid/CamVid/'+fi.strip().split(' ')[1]]
                
            

    label_stack = []
    print("[Camvid class weights] Building label stack")
    for label_path in label_list:
        label = cv2.imread(label_path)
        sup_labels = label-7

        sup_labels[np.logical_or(sup_labels<=0,sup_labels>=4)]=0
        label = sup_labels[:,:,0]
        
        label_stack.append(label)
    
    label_stack = np.stack(label_stack, axis=0)
    pdb.set_trace()
    print("[Camvid class weights] Calculating class weights")

    weights = calculate_class_weights(label_stack, NUM_CLASSES)
    config = configparser.ConfigParser()
    config['ClassWeights'] = {}
    config['ClassWeights']['background']         = str(weights[0])
    config['ClassWeights']['Car']  = str(weights[1])
    config['ClassWeights']['Pedestrain']           = str(weights[2])
    config['ClassWeights']['Bicycle']              = str(weights[3])

    config['ClassWeights']['num_images']         = str(len(label_list))

    with open(ARGS_TRAIN_DIR + "/class_weights.ini", "w") as configfile:
        config.write(configfile)

    print("[pst class weights] Saved class weights to {}".format(
          ARGS_TRAIN_DIR + "/class_weights.ini")
         )

if __name__ == '__main__':
    main()
    
    
    