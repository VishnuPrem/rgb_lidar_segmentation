ROOT_DIR = '/home/neil/squeezeSeg/Camvid/CamVid/'
IMG_WIDTH = 512

NUM_CLASSES=4
ARGS_NUM_WORKERS = 10
ARGS_BATCH_SIZE=2 

ARGS_MODEL = 'SqueezeSeg/'
ARGS_SAVE_DIR = '/home/neil/squeezeSeg/Saved_model/' 
ARGS_TRAIN_DIR = '/home/neil/squeezeSeg/'
ARGS_CUDA = True

OPT_LEARNING_RATE_INIT 	= 5e-4

OPT_BETAS 		= (0.9, 0.999)
OPT_EPS_LOW 		= 1e-08
OPT_WEIGHT_DECAY 	= 1e-4

ARGS_NUM_EPOCHS = 1000