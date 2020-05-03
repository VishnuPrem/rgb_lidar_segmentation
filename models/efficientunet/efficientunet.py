###############################################################
# Network definition for MobilenetV2 based encoder with a Unet architecture
#                             April 2020
#           Neil Rodrigues | University of Pennsylvania
###############################################################


import pdb
from efficientnet_pytorch import EfficientNet 

import torch
import torch.nn as nn
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class Net(nn.Module):
    def __init__(self, data_dict):
        super(Net,self).__init__()
        self.data_dict = data_dict
        self.base_model = EfficientNet.from_pretrained('efficientnet-b4',in_channels=5) 
        
        #self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_layers = list(self.base_model.children())
        self.base_layers[0].in_channels=5
        #pdb.set_trace()
        #self.base_layers = self.base_layers[0]
        
        #print(self.base_layers[0])
        #self.base_layers[0] = nn.Conv2d(CHANNELS,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)

        #self.base_layers[0][0] = nn.Conv2d(len(data_dict.CHANNELS),32,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        #print(self.base_layers[0])
        self.layer0 = nn.Sequential(*self.base_layers[:2],*self.base_layers[2][0:2]) # size=(N, 64/16, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(24,24, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[2][2:6]) # size=(N, 64/24, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(32, 32, 1, 0)
        self.layer2 = nn.Sequential(*self.base_layers[2][6:10]) # size=(N, 128/32, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(56, 56, 1, 0)
        self.layer3 = nn.Sequential(*self.base_layers[2][10:22])  # size=(N, 256/64, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(160, 160, 1, 0)
        self.layer4 = nn.Sequential(*self.base_layers[2][22:],*self.base_layers[3:5])  # size=(N, 512/1280, x.H/32, x.W/32)
        
        self.layer4_1x1 = convrelu(1792, 1792, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(160 + 1792, 1280, 3, 1)
        self.conv_up2 = convrelu(56 + 1280, 512, 3, 1)
        self.conv_up1 = convrelu(32 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(24 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(len(data_dict.CHANNELS), 16, 3, 1)
        self.conv_original_size1 = convrelu(16, 16, 3, 1)
        self.conv_original_size2 = convrelu(16 + 128, 16, 3, 1)

        self.conv_last = nn.Conv2d(16, data_dict.NUM_CLASSES, 1)

    def forward(self, input, x2 ,mask):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        #print('x_original',x_original.shape, 'layer0',layer0.shape)
        layer1 = self.layer1(layer0)
        
        layer2 = self.layer2(layer1)
        
        layer3 = self.layer3(layer2)
        
        layer4 = self.layer4(layer3)
        
        layer4 = self.layer4_1x1(layer4)
        #pdb.set_trace()
        #print('lowest feature size',layer4.shape)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)

        #print('first concat',x.shape,layer3.shape)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)

        layer2 = self.layer2_1x1(layer2)
        #print('second concat',x.shape,layer2.shape)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        #print('third concat',x.shape,layer1.shape)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        #print('fourth concat',x.shape,layer0.shape)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)

        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

def unit_test():
    from easydict import EasyDict as edict
    data_dict=edict()
    num_minibatch = 2
    num_channels = 5
    rgb = torch.randn(num_minibatch, num_channels, 64, 512).cuda(0)
    data_dict.NUM_CLASSES=4
    data_dict.CHANNELS="RGBDT"
    rtf_net = Net(data_dict).cuda(0)
    
    output=rtf_net(rgb,rgb,rgb)
    print(output.shape)
    

if __name__ == '__main__':
    unit_test()
