###############################################################
# Network definition for Fully Convolutional Network FCN-32
#                             April 2020
#           Ravi Teja | University of Pennsylvania
###############################################################


import numpy as np
import torch
import torch.optim as optim
import os

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self,data_dict):

    n_class = 4
    super(Net, self).__init__()

    """First Conv layer 
       input - 8*64*512 
       output - 64*64*256"""
    self.conv1_1 = nn.Conv2d(8,64,3,padding = 1) #64*512
    self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)   #64*512
    self.maxp1 = torch.nn.MaxPool2d((1,2),(1,2),ceil_mode = True)

    """Second Conv layer 
       input - 64*64*256 
       output - 128*64*128"""
    self.conv2_1 = nn.Conv2d(64,128,3,padding = 1) 
    self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
    self.maxp2 = torch.nn.MaxPool2d((1,2),(1,2),ceil_mode = True)

    """Third Conv layer 
       input - 128*64*128 
       output - 256*64*64 """
    self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
    self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
    self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
    self.maxp3 = torch.nn.MaxPool2d((1,2),(1,2),ceil_mode = True)

    """Fourth Conv layer 
       input - 256*64*64 
       output - 512*32*32"""
    self.conv4_1 = nn.Conv2d(256,512,3,padding = 1)
    self.conv4_2 = nn.Conv2d(512,512,3,padding = 1)
    self.conv4_3 = nn.Conv2d(512,512,3,padding = 1)
    self.maxp4 = torch.nn.MaxPool2d((2,2),(2,2),ceil_mode = True)

    """Fifth Conv layer 
       input - 512*32*32 
       output - 512*16*16 """
    self.conv5_1 = nn.Conv2d(512,512,3,padding = 1)
    self.conv5_2 = nn.Conv2d(512,512,3,padding = 1)
    self.conv5_3 = nn.Conv2d(512,512,3,padding = 1)
    self.maxp5 = torch.nn.MaxPool2d((2,2),(2,2),ceil_mode = True)

    """Sixth Conv layer 
       input - 512*16*16 
       output - 4096*16*16"""
    self.conv6 = nn.Conv2d(512,4096,3,padding = 1) # (10,10) , k_size = 7, no padding
    self.drop6 = nn.Dropout2d()

    """Senventh Conv layer 
       input - 4096*16*16 
       output - 4096*16*16"""
    self.conv7 = nn.Conv2d(4096,4096,3,padding = 1) #(10,10), k_size = 1, no padding
    self.drop7 = nn.Dropout2d()

    """Final Conv layer 
       input - 4096*16*16 
       output - 4*16*16"""
    self.conv8 = nn.Conv2d(4096,n_class,1) # (10,10), k_size  = 1, no padding

    """ConvT layer 
       input - 4*16*16 
       output - 4*512*512"""
    self.convT = nn.ConvTranspose2d(4,4,(8,64),stride = (4,30), padding =  (2,1), bias = False)    # 64,45,padding = 1
    self.initialize_weights()

  def initialize_weights(self):
    for m in self.modules():
      if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias,0)

  def forward(self, x,x2,mask):
    
    x = self.maxp1( F.relu ( self.conv1_2 ( F.relu ( self.conv1_1(x) ) ) ) )
    x = self.maxp2( F.relu ( self.conv2_2 ( F.relu ( self.conv2_1(x) ) ) ) )
    x = self.maxp3( F.relu ( self.conv3_3 ( F.relu ( self.conv3_2 ( F.relu ( self.conv3_1(x) ) ) ) ) ) )
    x = self.maxp4( F.relu ( self.conv4_3 ( F.relu ( self.conv4_2 ( F.relu ( self.conv4_1(x) ) ) ) ) ) )
    x = self.maxp5( F.relu ( self.conv5_3 ( F.relu ( self.conv5_2 ( F.relu ( self.conv5_1(x) ) ) ) ) ) )
    x = self.drop6( F.relu ( self.conv6(x) ) )
    x = self.drop7( F.relu ( self.conv7(x) ) )
    x = self.conv8(x)
    x = self.convT(x)
    x = x[:,:,:,:]
    

    return x


def unit_test():
  model = Net()
  z  = torch.zeros((1,8,64,512))
  x = model(z)
  print(x.shape)

if __name__ == '__main__':
    unit_test()
