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
from os import walk

def load_datastats():
    config = configparser.ConfigParser()
    config.sections()
    config.read(ARGS_TRAIN_DIR + "/data_stats.ini" )
    
    data_stats = np.ones([1,5,2])
    
    data_stats[0,0,:] = [float(i) for i in config['data_stats']['X'].split(',')]
    data_stats[0,1,:] = [float(i) for i in config['data_stats']['Y'].split(',')]
    data_stats[0,2,:] = [float(i) for i in config['data_stats']['Z'].split(',')]
    data_stats[0,3,:] = [float(i) for i in config['data_stats']['I'].split(',')]
    data_stats[0,4,:] = [float(i) for i in config['data_stats']['D'].split(',')]
    

    return data_stats



def load_class_weights():
    config = configparser.ConfigParser()
    config.sections()
    config.read(ARGS_TRAIN_DIR + "/class_weights.ini" )
    
    weights_mat = torch.ones([1,NUM_CLASSES])
    weights_mat[0,0] = float(config['ClassWeights']['background'])
    weights_mat[0,1] = float(config['ClassWeights']['Car'])
    weights_mat[0,2] = float(config['ClassWeights']['Pedestrain'])
    weights_mat[0,3] = float(config['ClassWeights']['Bicycle'])

    return weights_mat.squeeze()


def calculate_mean_var(data_x,data_y,data_z,data_i,data_d):
    mean_x,mean_y,mean_z,mean_i,mean_d = np.mean(data_x[data_x!=-1]), \
                                        np.mean(data_y[data_y!=-1]), \
                                        np.mean(data_z[data_z!=-1]), \
                                        np.mean(data_i[data_i!=-1]), \
                                        np.mean(data_d[data_d!=-1])

    std_x,std_y,std_z,std_i,std_d = np.std(data_x[data_x!=-1]), \
                                        np.std(data_y[data_y!=-1]), \
                                        np.std(data_z[data_z!=-1]), \
                                        np.std(data_i[data_i!=-1]), \
                                        np.std(data_d[data_d!=-1])
    stats = np.zeros((5,2))
    stats[0,:] = [mean_x,std_x]
    stats[1,:] = [mean_y,std_y]
    stats[2,:] = [mean_z,std_z]
    stats[3,:] = [mean_i,std_i]
    stats[4,:] = [mean_d,std_d]

    return stats

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

    image_list=[]
    for (dirpath, dirnames, filenames) in walk('/home/neil/cis_522/squeezeSeg/data/train/'):
        for file in filenames:
            image_list+=[os.path.join(dirpath,file)]      
    


    data_x=[]
    data_y=[]
    data_z=[]
    data_d=[]
    data_i=[]
    data_l=[]
    print("[Camvid class weights] Building label stack")
    for data_path in image_list:
        data = np.load(data_path)
        
        data_x.append(data[:,:,0])
        data_y.append(data[:,:,1])
        data_z.append(data[:,:,2])
        data_i.append(data[:,:,3])
        data_d.append(data[:,:,4])
        data_l.append(data[:,:,5])
        

    
    data_x = np.stack(data_x, axis=0)
    data_y = np.stack(data_y, axis=0)
    data_z = np.stack(data_z, axis=0)
    data_i = np.stack(data_i, axis=0)
    data_d = np.stack(data_d, axis=0)
    data_l = np.stack(data_l, axis=0).astype(int)
    data_l[data_l==-1]=0
    
    
    print("[Camvid class weights] Calculating class weights")

    weights = calculate_class_weights(data_l, 4)
    stats = calculate_mean_var(data_x,data_y,data_z,data_i,data_d)
    
    config = configparser.ConfigParser()
    config['ClassWeights'] = {}
    config['ClassWeights']['background']         = str(weights[0])
    config['ClassWeights']['Car']  = str(weights[1])
    config['ClassWeights']['Pedestrain']           = str(weights[2])
    config['ClassWeights']['Bicycle']              = str(weights[3])

    
    config_stats = configparser.ConfigParser()
    config_stats['data_stats'] = {}
    config_stats['data_stats']['X'] = ','.join([str(i) for i in stats[0,:]])
    config_stats['data_stats']['Y'] = ','.join([str(i) for i in stats[1,:]])
    config_stats['data_stats']['Z'] = ','.join([str(i) for i in stats[2,:]])
    config_stats['data_stats']['I'] = ','.join([str(i) for i in stats[3,:]])
    config_stats['data_stats']['D'] = ','.join([str(i) for i in stats[4,:]])



    with open(ARGS_TRAIN_DIR + "/class_weights.ini", "w") as configfile:
        config.write(configfile)
    with open(ARGS_TRAIN_DIR + "/data_stats.ini", "w") as configfile:
        config_stats.write(configfile)

    print("[pst class weights] Saved class weights to {}".format(
          ARGS_TRAIN_DIR + "class_weights.ini")
         )

if __name__ == '__main__':
    main()
    
    
    