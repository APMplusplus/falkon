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

print_flag = 0

# Input: Phones of Shape(B,T1) and World ccoeffs of dim 60 Shape (B,T2,C)
# Output: ccoeffs of Shape (B,T2,C)
class baseline_model(nn.Module):

      def __init__(self):
          super(baseline_model, self).__init__()

class baseline_lstm(baseline_model):

        def __init__(self, vocab_size):
          super(baseline_lstm, self).__init__()

          self.vocab_size = vocab_size
          self.embed_size = 128
          
          # Encoder
          self.encoder_embedding = nn.Embedding(self.vocab_size, self.embed_size)
          self.seq_model = nn.LSTM(128, 64, 1, batch_first=True)

          # Decoder
          self.decoder_initialfc = nn.Linear(64, 128)
          self.decoder_lstm = nn.LSTM(128, 66, 1, batch_first=True)
          self.final_fc = nn.Linear(66, 128)

        def encoder(self, x):
            
           # Do something about the phones
           x = self.encoder_embedding(x)

           # Pass through some sequence model
           x, (c,h) = self.seq_model(x, None)

           return c[0]
       
        def decoder(self, encoder_output, ccoeffs):
                
            # Loop through the length of decoder
            max_steps = ccoeffs.shape[1]
            outputs = []
            initial_input = encoder_output.new(encoder_output.shape[0],1)
            x = initial_input.zero_()
            x = self.encoder_embedding(x.long())
            assert len(x.shape) == 3
            hidden = None
            for i in range(max_steps):
                
                assert x.shape[1] == 1
                x, hidden = self.decoder_lstm(x, hidden)
                o = x.squeeze(1)
                outputs.append(o)
                
                # Modify shape for next input
                x = self.final_fc(x)
                
            decoder_outputs = torch.stack(outputs, 1)
            return decoder_outputs
       
        def forward(self, x, c):
            encoder_output = self.encoder(x)
            if print_flag:
                print("Shape of encoder lstm output: ", encoder_output.shape)
            
            decoder_output =  self.decoder(encoder_output, c)
            if print_flag:
                print("Shape of decoder lstm output: ", decoder_output.shape)
                
            return decoder_output
        


class attentionlstm(baseline_lstm):

        def __init__(self, vocab_size):
           super(attentionlstm, self).__init__(vocab_size)

           # Attention
           self.attention_w = nn.Linear(64, 32)
           self.attention_u = nn.Parameter(torch.randn((32,16)))
           self.attention_fc = nn.Linear(128, 64)
           
           self.vocab_size = vocab_size
           self.embed_size = 128
          
           # Encoder
           self.encoder_embedding = nn.Embedding(self.vocab_size, self.embed_size)
           self.seq_model = nn.LSTM(128, 64, 1, batch_first=True)

           # Decoder
           self.decoder_initialfc = nn.Linear(64, 128)
           self.decoder_lstm = nn.LSTM(32, 66, 1, batch_first=True)
           self.final_fc = nn.Linear(66, 128)

        def encoder(self, x):
            
           # Do something about the phones
           x = self.encoder_embedding(x)

           # Pass through some sequence model
           x, (c,h) = self.seq_model(x, None)

           return x

        def decoder(self, encoder_output, ccoeffs):
                
            # Loop through the length of decoder
            max_steps = ccoeffs.shape[1]
            outputs = []
            initial_input = encoder_output.new(encoder_output.shape[0],1)
            input = initial_input.zero_()
            x = self.encoder_embedding(input.long())
            assert len(x.shape) == 3
            hidden = None
            for i in range(max_steps):
                
                assert x.shape[1] == 1
                decoder_lstm_input = self.attend(x, encoder_output)
                decoder_lstm_input = decoder_lstm_input.unsqueeze(1)
                #print("Shape of decoder lstm input is ", decoder_lstm_input.shape)
                
                assert decoder_lstm_input.shape[1] == 1
                x, hidden = self.decoder_lstm(decoder_lstm_input, hidden)
                
                o = x.squeeze(1)
                outputs.append(o)
                
                # Modify shape for next input
                x = self.final_fc(x)
                
            decoder_outputs = torch.stack(outputs, 1)
            return decoder_outputs
              
       
        def attend(self, a, A):

           a = self.attention_fc(a)
           if print_flag:
               print("Shapes of a and A are: ", a.shape, A.shape)

           A = a * A     
           assert len(A.shape) == 3
           batch_size = A.shape[0]
           #print("Shape of A: ", A.shape)

           # Multiply
           alpha = torch.tanh(self.attention_w(A))
           alpha = F.softmax(alpha, dim = -1)
           #print("Shape of alpha: ", alpha.shape) 

           # Sum
           #beta = torch.bmm(alpha, self.attention_u)
           beta = torch.sum(alpha, dim = 1)

           # Return
           #print("Shape of beta is ", beta.shape)
           return beta
       

        def forward(self, x, c):
            encoder_output = self.encoder(x)
            if print_flag:
                print("Shape of encoder lstm output: ", encoder_output.shape)
            
            decoder_output =  self.decoder(encoder_output, c)
            if print_flag:
                print("Shape of decoder lstm output: ", decoder_output.shape)
                
            return decoder_output
