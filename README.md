# SqueezeSeg for RGB-Lidar Semantic Segmentation

### Data Stats

To extract stats of data in 'data/' run

'''
python3 calculate_weights.py
'''
after navigating into utils

### Config File

Tune hyper parameters, path files, input channels.

'''
config.py
'''

You only need to change ARGS_ROOT to point to the repo

### Data for training

Save the data to folder 'data/train/'