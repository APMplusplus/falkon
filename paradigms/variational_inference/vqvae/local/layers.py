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

class Conv1dplusplus(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearized_weight)

    def incremental_forward(self, input):
        # input: (B, T, C)
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')

        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]

        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                #print("Input buffer is None")
                self.input_buffer = input.new(bsz, kw + (kw - 1) * (dilation - 1), input.size(2))
                self.input_buffer.zero_()
            else:
                # shift buffer
                #print("Shifting input buffer")
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
                #print("     Layer: Dilation of this layer is ", dilation, " and the number of time steps in the layer: ", self.input_buffer.shape[1])
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
        if print_flag:
           print("Shape of input and the weight: ", input.shape, weight.shape)
        output = F.linear(input.view(bsz, -1), weight, self.bias)
        return output.view(bsz, 1, -1)


    def clear_buffer(self):
        self.input_buffer = None

    def _clear_linearized_weight(self, *args):
        self._linearized_weight = None

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            if self.weight.size() == (self.out_channels, self.in_channels, kw):
                weight = self.weight.transpose(1, 2).contiguous()
            else:
                # fairseq.modules.conv_tbc.ConvTBC
                weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight



class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps = x.size(0), x.size(1)
        x = x.contiguous()
        x = x.view(batch_size * time_steps, -1)
        x = self.module(x)
        x = x.contiguous()
        x = x.view(batch_size, time_steps, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


#https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
#https://discuss.pytorch.org/t/custom-binarization-layer-with-straight-through-estimator-gives-error/4539/5
class straight_through(torch.autograd.Function):

     @staticmethod
     def forward(ctx, input):
         #print("Shape of input to the quantizer: ", input.shape)
         ctx.save_for_backward(input)
         #print("Shape of output from the quantizer: ", out.shape)
         return input

     @staticmethod
     def backward(ctx, grad_output):
         input, = ctx.saved_tensors
         grad_output[input>1]=0
         grad_output[input<-1]=0
         return grad_output



class baseline_model(nn.Module):

      def __init__(self):
          super(baseline_model, self).__init__()


class VectorQuantizer(baseline_model):

     def __init__(self, num_classes, dimensions):
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Parameter(torch.rand(num_classes,dimensions))
        self.activation = straight_through.apply

     def forward(self, encoded):
          bsz = encoded.shape[0]
          T = encoded.shape[1]
          dims = encoded.shape[2]
          encoded = encoded.reshape(bsz*T, dims)
          ## Loop over batch. (Cant you code better?)
          index_batch = []
          for chunk in encoded:
             c = (chunk - self.embedding).norm(dim=1)
             index_batch.append(torch.argmin(c))
          index_batch = torch.stack(index_batch, dim=0)  
          quantized_values = torch.stack([self.embedding[k] for k in index_batch], dim=0)
          activated_values =  self.activation(quantized_values)
          return activated_values.reshape(bsz, T, dims)
        
