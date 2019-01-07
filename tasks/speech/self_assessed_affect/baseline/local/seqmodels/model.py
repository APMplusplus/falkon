import numpy as np
import os, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

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
          self.encoder_dropout = layers.SequenceWise(nn.Dropout(0.3))

          self.seq_model = nn.LSTM(32, 16, 2, bidirectional=True, batch_first=True)

          self.final_fc = nn.Linear(32, 3)

        def forward(self, c):

           x = self.encoder_fc(c)
           x = self.encoder_dropout(x)

           x, _ = self.seq_model(x, None)
           x = self.final_fc(x)

           return x[:,-1,:]

