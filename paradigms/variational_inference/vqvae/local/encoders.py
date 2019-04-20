import numpy as np
import os, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


sys.path.append('/home/srallaba/development/repos/falkon/')
import src.nn.layers as falcon_layers

print_flag = 0

class DownsamplingEncoder(nn.Module):
    """
        Input: (N, samples_i) numeric tensor
        Output: (N, samples_o, channels) numeric tensor
    """
    def __init__(self, channels):
        super().__init__()

        self.convs_wide = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        prev_channels = 1
        total_scale = 1
        pad_left = 0
        self.skips = []
        ksz = 4
        dilation_factor = 1
        convs_wide = {}
        convs_1x1 = {}
        strides = [2,2,2,1,2,1,2,1,2,1]
        self.layer_specs = []

        #### Lets share the parameters
        for stride in strides:

            keystring = str(stride) + '_' + str(prev_channels)

            if keystring  in convs_wide.keys() and keystring in convs_1x1.keys():
                conv_wide = convs_wide[keystring]
                conv_1x1 = convs_1x1[keystring]
                self.convs_wide.append(conv_wide)
                self.convs_1x1.append(conv_1x1)
            else:
                conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor)
                wsize = 2.967 / math.sqrt(ksz * prev_channels)
                conv_wide.weight.data.uniform_(-wsize, wsize)
                conv_wide.bias.data.zero_()
                self.convs_wide.append(conv_wide)
                convs_wide[keystring] = conv_wide

                conv_1x1 = nn.Conv1d(channels, channels, 1)
                conv_1x1.bias.data.zero_()
                self.convs_1x1.append(conv_1x1)
                convs_1x1[keystring] = conv_1x1

            prev_channels = channels
            skip = (ksz - stride) * dilation_factor
            pad_left += total_scale * skip
            self.skips.append(skip)
            total_scale *= stride
            self.layer_specs.append((stride, ksz, dilation_factor))
        
        self.pad_left = pad_left
        self.total_scale = total_scale

        self.final_conv_0 = nn.Conv1d(channels, channels, 1)
        self.final_conv_0.bias.data.zero_()
        self.final_conv_1 = nn.Conv1d(channels, channels, 1)
        
        if print_flag:
           print("   Layer: Lengths of self.convs_wide: ", len(self.convs_wide)) 

    def gating_function(self, conv_wide, conv_1x1, x):
        x = conv_wide(x)
        x_a, x_b = x.split(x.size(1) // 2, dim=1)
        x = torch.tanh(x_a) * torch.sigmoid(x_b)
        x = conv_1x1(x)
        return x

    def forward(self, samples):
        x = samples.transpose(1,2)
        if print_flag:
           print("   Layer: Shape of x input to the downsampling encoder: ", x.shape)

        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            if print_flag:       
               print("   Layer: Going through loop ", i) 
            conv_wide, conv_1x1, layer_specs, skip = stuff
            stride, ksz, dilation_factor = layer_specs
           
            x_gating = self.gating_function(conv_wide, conv_1x1, x)

            if i == 0:
                x = x_gating
            else:
                x = x_gating + x[:, :, skip:skip+x_gating.size(2)*stride].view(x.size(0), x_gating.size(1), x_gating.size(2), -1)[:, :, :, -1]

        x = self.final_conv_1(F.relu(self.final_conv_0(x)))
        return x.transpose(1,2)





