# RGB-Lidar Semantic Segmentation

This code was trained and tested in the following enviorenment

python = 3.6.5
torch = 1.1.0
torchvision = 0.3.0 
PIL = 6.1.0
numpy = 1.15.0

### Example results
| ResFcn_34 Output  |           
|-------------------------|
|![result](/report/2d_res34fcn.gif) | 
|  Ground truths |
|-------------------------|
| ![result](/report/2d_ground_truth.gif) |

### Download Data and trained model weights
The data[Lidar + RGB] was converted to PGM representation taking the form of a 64x512 image.
There are 8 such channels, XYZ D[Depth] I[Intensity] RGB

This data is stored in npy format, details of the data generations can be found on [here](https://github.com/arsjindal/Segmentation-on-Point-Cloud)

To download data and trained model weights use
```
./get_data.sh
```
This will also install the efficientnet_pytorch library with pretrained weights


Remember to change the following in config.py before training:
```
ROOT_DIR: Data path 
ARGS_ROOT: Path pointing to code directory
```

### Config File

Tune hyper parameters, path files, input channels.

```
config.py
```

You only need to change ARGS_ROOT to point to the repo
#### Model Specific Parameters
```
ARGS_MODEL_NAME: model name as declared in models/ directory
ARGS_MODEL: directory in which to save best model weights

ARGS_INPUT_TYPE_1: channels that correspond to the input of the first part of the network
ARGS_INPUT_TYPE_2: channels that correspond to the input of the second part of the network

ARGS_TRAIN_BATCH_SIZE: Batch size for training
ARGS_LEARNING_RATE_INIT: Initial Learning rate
ARGS_NUM_EPOCHS: Number of epoch to train the model
```

### Data Stats

To extract stats of data in ROOT_DIR run

```
python3 calculate_weights.py
```
after navigating into utils

The following files will be created:
```
data_stats.ini : which contains the mean and variance of train data for XYZDI
class_weights.ini : which contains the paszke weights for class imbalance
```

### Training models

After making the changes to the config file, run

```
CUDA_VISIBLE_DEVICES=0 python3 train.py
```


### Inference on Models in the report

Run the following:
```
python3 inference_table.py
```

The output should look like this
![result](/report/class_iou.png)

### Inference on trained models
Change the following variables in infer_config.py

```
ARGS_ROOT: Data path 
ROOT_DIR: Path pointing to code directory

ARGS_MODEL_NAME: model name as declared in models/
ARGS_MODEL: directory in which to save best model weights

ARGS_INPUT_TYPE_1: channels that correspond to the input of the first part of the network
ARGS_INPUT_TYPE_2: channels that correspond to the input of the second part of the network
ARGS_VAL_BATCH_SIZE: Batch size for inference
ARGS_NUM_WORKERS: number of CPU cores to use during evaluation 
```

To run inference:
```
python3 infer_single.py
```

