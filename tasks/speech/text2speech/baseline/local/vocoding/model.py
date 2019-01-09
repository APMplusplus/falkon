import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
from modules import *
from layers import *
import numpy as np
from utils import *

print_flag = 0
frame_period= 16

class cnnmodel(nn.Module): 

    def __init__(self):
        super(cnnmodel, self).__init__()

        self.embedding = nn.Embedding(259, 128)
        self.encoder_fc = SequenceWise(nn.Linear(128, 64))
        self.encoder_dropout = nn.Dropout(0.3)
        self.kernel_size = 3
        self.stride = 1

        layers = 24
        stacks = 4
        layers_per_stack = layers // stacks

        self.conv_modules = nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            self.padding = int((self.kernel_size - 1) * dilation)
            conv = residualconvmodule(64,64, self.kernel_size, self.stride, self.padding,dilation)
            self.conv_modules.append(conv)

        self.final_fc1 = SequenceWise(nn.Linear(64, 512))
        self.final_fc2 = SequenceWise(nn.Linear(512, 259))
        
        self.nlayers = 1
        self.nhid = 128

    def encode(self, x, teacher_forcing_ratio):
        x = self.embedding(x.long())
        if len(x.shape) < 3:
           x = x.unsqueeze(1)
        x = F.relu(self.encoder_fc(x))
        if teacher_forcing_ratio > 0.1:
           #print("Dropping out")
           x = self.encoder_dropout(x)
        return x

    def upsample_ccoeffs(self, c, frame_period=16):
        if print_flag:
           print("Shape of ccoeffs in upsampling routine is ", c.shape)
        c = c.transpose(1,2)
        c = F.interpolate(c, size=[c.shape[-1]*frame_period])
        c = c.transpose(1,2)
        if print_flag:
           print("Shape of ccoeffs after upsampling is ", c.shape)
        return c #[:,:-1,:]


    def forward(self,x, c, tf=1):


       # Do something about the wav
       x = self.encode(x.long(), 1.0)

       # Do something about the ccoeffs
       c = self.upsample_ccoeffs(c)       

       # Feed to Decoder
       x = x.transpose(1,2)
       for module in self.conv_modules:
          x = F.relu(module(x, c))

       x = x.transpose(1,2)

       x = F.relu(self.final_fc1(x))
       x = self.final_fc2(x)

       return x[:,:-1,:]


    def clear_buffers(self):

       for module in self.conv_modules:
           module.clear_buffer()

    def forward_incremental(self,x, c, gen_flag = 0):

       self.clear_buffers()
       print(c)
       # Get the length to predict
       max_length = c.shape[1] * frame_period
       print("Max Length is ", max_length)
       #max_length = 8000
       bsz = c.shape[0]
       if print_flag:
          print("  Model: Shape of x and c in the model: ", x.shape, c.shape, " and the max length is  ", max_length)

       x = c.new(bsz,1)
       a = 0
       x.fill_(a)

       outputs = []
       samples = []

       # Do something about the ccoeffs
       c = self.upsample_ccoeffs(c)    
       if print_flag:
           print("  Model: Shape of upsampled c is ", c.shape)
           #sys.exit()

       for i in range(max_length-1):

          # Do something about the wav
          x = self.encode(x.long(), 0.0)

          # Feed to Decoder
          ct = c[:,i,:].unsqueeze(1)
          if print_flag:
             print("  Model: Shape of ct in the model is ", ct.shape)

          assert len(x.shape) == 3
          
          for module in self.conv_modules:
             x = F.relu(module.incremental_forward(x, ct))

          if print_flag:
             print("  Model: Shape of input to the final fc in forward_incremental of model is ", x.shape, " and the time steps: ", i )      

          x = F.relu(self.final_fc1(x))
          x = self.final_fc2(x)
          #x += sample_gumbel(x.size(), out=x.data.new())
          probs = F.softmax(x.view(bsz, -1), dim=1)
          predicted = torch.max(x.view(bsz, -1), dim=1)[1]
          #print("Shape of probs: ", probs.shape)
          sample = np.random.choice(np.arange(259), p = probs.view(-1).data.cpu().numpy())
          #print(sample, predicted.shape)
          #assert sample.shape == predicted.shape
          predicted = np.array([sample])
          #v, predicted = torch.max(x,2)
          #print("I picked ", i, predicted)
          sample_onehotk = x.new(x.shape[0], 259)
          sample_onehotk.zero_()
          sample_onehotk[:,predicted] = 1
          #outputs.append(predicted)
          outputs.append(x)
          #samples.append(predicted)
          samples.append(sample_onehotk)
          x = torch.LongTensor(predicted).cuda()

          '''
          # Sample the next input
          probs = F.softmax(x.view(bsz, -1), dim=1)
          #print("Shape of probs is ", probs.shape)
          #print(sample)
          #import sys
          #sys.exit()
          
          
          #print("Not using gumbel noise") 
          #x += sample_gumbel(x.size(), out=x.data.new())
          #sample = torch.max(x.view(bsz, -1), dim=1)[1]
          
          sample = np.random.choice(np.arange(259), p = probs.view(-1).data.cpu().numpy())
          sample_copy = sample
          x = x.view(bsz,-1)
          x.zero_()
          #print("Shape of x: ", x.shape)
          x[:,sample_copy] = 1
          outputs += [x.data]
          sample = torch.LongTensor(np.array([sample]))
          sample = sample.unsqueeze(0)
          sample = sample.cuda()
          x = sample
          print("Picked sample: ", sample_copy)
          
          
          if print_flag:
             print("  Model: Shape of next input in forward_incremental of model is ", x.shape )
   
          '''

       outputs = torch.stack(outputs)
       samples = torch.stack(samples)
       if gen_flag:
          return samples
       return outputs



