import numpy as np
import os, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

## Locations
FALCON_DIR = os.environ.get('FALCON_DIR')
BASE_DIR = os.environ.get('base_dir')
DATA_DIR = os.environ.get('data_dir')
assert ( all (['FALCON_DIR', 'BASE_DIR', 'DATA_DIR']) is not None)

FEATS_DIR = BASE_DIR + '/feats/world_feats_20msec'
ETC_DIR = BASE_DIR + '/etc'

sys.path.append(FALCON_DIR)
import src.nn.layers as layers

import torch.nn.functional as F
import random
import math


# Input: World ccoeffs of dim 60 Shape (B,T,C)
# Output: Logits of Shape (B,3)
class baseline_model(nn.Module):

      def __init__(self):
          super(baseline_model, self).__init__()

class baseline_lstm(baseline_model):

        def __init__(self):
          super(baseline_lstm, self).__init__()

          self.encoder_fc = layers.SequenceWise(nn.Linear(60, 32))
          self.encoder_dropout = layers.SequenceWise(nn.Dropout(0.7))

          self.seq_model = nn.LSTM(32, 64, 1, bidirectional=True, batch_first=True)

          self.final_fc = nn.Linear(32, 3)

        def forward(self, c):

           x = self.encoder_fc(c)
           x = self.encoder_dropout(x)

           x, (c,h) = self.seq_model(x, None)
           hidden_left , hidden_right = h[0,:,:], h[1,:,:]
           hidden = torch.cat((hidden_left, hidden_right),1)
           x = self.final_fc(hidden)
           return x

        def forward_eval(self, c):

           x = self.encoder_fc(c)

           x, (c,h) = self.seq_model(x, None)
           hidden_left , hidden_right = h[0,:,:], h[1,:,:]
           hidden = torch.cat((hidden_left, hidden_right),1)
           x = self.final_fc(hidden)
           return x


class attentionlstm(baseline_lstm):

        def __init__(self):
           super(attentionlstm, self).__init__()

           self.attention_w = nn.Linear(128, 32)
           self.attention_u = nn.Parameter(torch.randn((32,16)))

        def attend(self, A):

           assert len(A.shape) == 3
           batch_size = A.shape[0]
           #print("Shape of A: ", A.shape)

           # Multiply
           alpha = F.tanh(self.attention_w(A))
           alpha = F.softmax(alpha, dim = -1)
           #print("Shape of alpha: ", alpha.shape) 

           # Sum
           #beta = torch.bmm(alpha, self.attention_u)
           beta = torch.sum(alpha, dim = 1)

           # Return
           #print("Shape of beta is ", beta.shape)
           return beta

        def forward(self, c):

           x = self.encoder_fc(c)
           x = self.encoder_dropout(x)

           x, (c,h) = self.seq_model(x, None)
           weighted_representation = self.attend(x)
           #print("Shape of weighted representation from attention is ", weighted_representation.shape)
           x = self.final_fc(weighted_representation)
           return x

        def forward_eval(self, c):

           x = self.encoder_fc(c)

           x, (c,h) = self.seq_model(x, None)
           weighted_representation = self.attend(x)
           x = self.final_fc(weighted_representation)

           return x


