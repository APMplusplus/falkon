import torch
import numpy as np
import os, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

class baseline_model(nn.Module):

      def __init__(self):
          super(baseline_model, self).__init__()

class baseline_mlp(baseline_model):

       def __init__(self):
          super(baseline_mlp, self).__init__()

          self.encoder_fc = nn.Linear(240, 120)
          self.decoder_fc = nn.Linear(120, 40)

       def forward(self, x):
          x = torch.tanh(self.encoder_fc(x))
          x = self.decoder_fc(x)
          return x 
