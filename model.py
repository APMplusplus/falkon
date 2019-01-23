import torch
import numpy as np
import os, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F

class baseline_model(nn.Module):

      def __init__(self):
          super(baseline_model, self).__init__()

# Input: (B, 240) Output: (B, 40)
class baseline_mlp(baseline_model):

       def __init__(self):
          super(baseline_mlp, self).__init__()

          self.encoder_fc = nn.Linear(240, 120)
          self.decoder_fc = nn.Linear(120, 40)

       def forward(self, x):
          x = torch.tanh(self.encoder_fc(x))
          x = self.decoder_fc(x)
          return x 

# Input: (B, 240) Output: (B, 40)
class baseline_cnn(baseline_model):

       def __init__(self):
          super(baseline_cnn, self).__init__()

          self.encoder_fc = nn.Linear(240, 240)
          self.cnn = nn.Conv1d(1, 40, kernel_size=240) 
          self.decoder_fc = nn.Linear(40, 40)

       def forward(self, x):
          x = torch.tanh(self.encoder_fc(x))
          x = x.unsqueeze(1) 
          x = F.relu(self.cnn(x))
          x = x.squeeze(-1)
          x = self.decoder_fc(x)
          return x 