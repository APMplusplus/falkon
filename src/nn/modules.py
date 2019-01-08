import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
from layers import *


import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
from layers import *


class residualconvmodule(nn.Module):

    def __init__(self,  in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(residualconvmodule, self).__init__()

        self.conv = self.weightnorm_conv1d( in_channels, out_channels, kernel_size, stride, padding, dilation )
        self.prefinal_fc = SequenceWise(nn.Linear(256, 256))



    def weightnorm_conv1d(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        dropout = 0
        std_mul = 1.0
        m = AdvancedConv1d(in_channels,out_channels, kernel_size=kernel_size, stride=stride, padding = padding, dilation = dilation)
        std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
        m.weight.data.normal_(mean=0, std=std)
        m.bias.data.zero_()
        return nn.utils.weight_norm(m)

    def clear_buffer(self):
        self.conv.clear_buffer()


    def forward(self, x, c, g=None):
        return self._forward(x, c, g, False)


    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)


    def _forward(self,x, c, g, incremental_flag):

        residual = x        

        # Feed to the module
        if incremental_flag:
           if print_flag:
              print("   Module: The shape of residual in the module is ", residual.shape, " and that of x is ", x.shape) 
           assert residual.shape[1] == x.shape[1]
           x = F.relu(self.conv.incremental_forward(x))

        else:
           x = F.relu(self.conv(x))
           x = x.transpose(1,2)
           # Causal
           x = x[:,:residual.shape[2],:] 

        if print_flag:
           print("   Module: The shape of residual in the module is ", residual.shape)
           print("   Module: Shape of x after residual convs is ", x.shape)
           print("   Module: Shape of x before prefinal fc is ", x.shape)
 
        x = self.prefinal_fc(x)

        if incremental_flag:
           pass
        else:
           x = x.transpose(1,2)

        if print_flag:
           print("   Module: Shape of x right before adding the residual and the residual: ", x.shape, residual.shape)
        assert x.shape == residual.shape

        x = (x + residual) * math.sqrt(0.5)

        return x


