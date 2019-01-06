import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
from layers import *

print_flag = 0



class residualconvmodule(nn.Module):

    def __init__(self,  in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(residualconvmodule, self).__init__()

        self.conv = self.weightnorm_conv1d( in_channels, out_channels, kernel_size, stride, padding, dilation )

        self.local_fca = SequenceWise(nn.Linear(60, 32))
        self.local_fcb = SequenceWise(nn.Linear(60, 32))
        self.prefinal_fc = SequenceWise(nn.Linear(32, 64))

        self.nlayers = 1
        self.nhid = 128


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
           print_flag = 0
           if print_flag:
              print("   Module: The shape of residual in the module is ", residual.shape) 
           assert residual.shape[1] == x.shape[1]
           x = F.relu(self.conv.incremental_forward(x))

        else:
           print_flag = 0
           #x = x.transpose(1,2)
           x = F.relu(self.conv(x))
           x = x.transpose(1,2)
           # Causal
           x = x[:,:residual.shape[2],:] 

        if print_flag:
           print("   Module: The shape of residual in the module is ", residual.shape)
           print("   Module: Shape of x after residual convs is ", x.shape)
           print("   Module: Shape of c in the module is ", c.shape)

        # Expert and Gating
        a,b = x.split(x.shape[-1] // 2, dim = -1)
 
        # Local conditioning
        ca = self.local_fca(c.double())
        cb = self.local_fcb(c.double())
        if print_flag:
           print("   Module: Shape of ca and cb in the module: ", ca.shape, cb.shape)

        a, b = a + ca, b + cb

        # Combine
        x = torch.tanh(a) * torch.sigmoid(b)
        if print_flag:
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

