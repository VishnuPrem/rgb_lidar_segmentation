
ARGS_ROOT = '/data/Docker_Codebase/cis_522/squeezeSeg/'
ROOT_DIR = '/data/Docker_Data/kitti_data/' 

ARGS_MODEL_NAME = 'efficientunet'
ARGS_MODEL = 'effnet/'

ARGS_SAVE_DIR = ARGS_ROOT +'Saved_model/' + ARGS_MODEL + 'Inference/'

ARGS_INFERENCE_MODEL = ARGS_ROOT + 'Saved_model/' + ARGS_MODEL + 'model_best.pth'
NUM_CLASSES = 4

ARGS_INPUT_TYPE_1 = 'XYZDI'
ARGS_INPUT_TYPE_2 = 'DIRGB'


ARGS_VAL_BATCH_SIZE = 1
ARGS_CUDA=True

ARGS_NUM_WORKERS = 5


import numpy as np
from easydict import EasyDict as edict

data_dict = edict()
data_dict.CLASSES = ['unkwon', 'car', 'pedestrian', 'cyclist']
data_dict.CHANNELS = ARGS_INPUT_TYPE_1
data_dict.NUM_CLASSES = NUM_CLASSES
# data_dict.NUM_CLASSES = 9
data_dict.CLS_2_ID = dict(zip(data_dict.CLASSES, range(len(data_dict.CLASSES))))
data_dict.CLS_LOSS_WEIGHT = np.array([1/15.0, 1.0, 10.0, 10.0])
data_dict.CLS_COLOR_MAP = np.array([
	[0.00, 0.00, 0.00],
	[0.12, 0.56, 0.37],
	[0.66, 0.55, 0.71],
	[0.58, 0.72, 0.88]])

data_dict.AZIMUTH_LEVEL = 512
data_dict.ZENITH_LEVEL = 64

data_dict.LCN_HEIGHT = 3
data_dict.LCN_WIDTH = 5
data_dict.RCRF_ITER = 3
data_dict.BILATERAL_THETA_A = np.array([.9, .9, .6, .6])
data_dict.BILATERAL_THETA_R = np.array([.015, .015, .01, .01])
data_dict.BI_FILTER_COEFF = 0.1
data_dict.ANG_THETA_A = np.array([.9, .9, .6, .6])
data_dict.ANG_FILTER_COEFF = 0.02

data_dict.CLS_LOSS_COEF = 15.0

data_dict.DATA_AUGMENTATION = False
data_dict.RANDOM_FLIPPING = False

# x, y, z, intensity, distance for Normalization
data_dict.INPUT_MEAN = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
data_dict.INPUT_STD = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])
