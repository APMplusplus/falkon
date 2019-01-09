import numpy as np
import os, sys
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.io import wavfile as wf

def load_data(feats_path, file, k):
   max = 0
   f= open(file)
   feats_array = []
   wav_array = []
   ctr = 0
   for line in f:
    if ctr < k:
     ctr += 1
     line = line.split('\n')[0].split('|')
     wav = np.load(feats_path + '/' + line[0])
     feats = np.load(feats_path + '/' + line[1])
     length = len(feats)
     max_val = np.max(wav)
     if max_val > max:
        max = max_val
        #print(wav)
     feats_array.append(feats)
     wav_array.append(wav)
     if ctr % 1000 == 1:
         print("Processed ", ctr, " files")
   print ("Maximum label value is ", max)
   return np.array(feats_array), np.array(wav_array)


def make_charmap(charset):
    # Create the inverse character map
    return {c: i for i, c in enumerate(charset)}

def make_intmap(charset):
    # Create the inverse character map
    return {i: c for i, c in enumerate(charset)}

def map_characters(utterances, charmap):
    # Convert transcripts to ints
    ints = [np.array([charmap[c] for c in u], np.int32) for u in utterances]
    return ints

def build_charset(utterances):

    # Create a character set
    chars = set(itertools.chain.from_iterable(utterances))
    chars = list(chars)
    chars.sort()
    return chars

class arctic_dataset(Dataset):
      def __init__(self, ccoeffs_array, wav_array):
          self.ccoeffs_array = ccoeffs_array
          self.wav_array = wav_array

      def __getitem__(self, index):
           return self.ccoeffs_array[index], self.wav_array[index]

      def __len__(self):
           return len(self.ccoeffs_array)



def mulaw(x, mu=256):
   return _sign(x) * _log1p(mu * _abs(x)) / _log1p(mu)

def mulaw_quantize(x, mu=256):
   y = mulaw(x, mu)
   return _asint((y + 1) / 2 * mu)

def _sign(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sign(x) if isnumpy or isscalar else x.sign()


def _log1p(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.log1p(x) if isnumpy or isscalar else x.log1p()


def _abs(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.abs(x) if isnumpy or isscalar else x.abs()


def _asint(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else x.long()


def _asfloat(x):
    # ugly wrapper to support torch/numpy arrays
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else x.float()


def inv_mulaw(y, mu=256):
    return _sign(y) * (1.0 / mu) * ((1.0 + mu)**_abs(y) - 1.0)


def inv_mulaw_quantize(y, mu=256):
  y = 2 * _asfloat(y) / mu - 1
  
  return ( inv_mulaw(y,mu)) 

def quantize_wavfile(file):
   fs, A = wf.read(file)

   x_1 = (A/32768.0).astype(np.float32) 

   y_1 = mulaw_quantize(x_1,256)

   return y_1


def sample_gumbel(shape, eps=1e-10, out=None):
   """
   Sample from Gumbel(0, 1)
   based on
   https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
   (MIT license)
   """
   U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
   return - torch.log(eps - torch.log(U + eps))

def gumbel_argmax(logits, dim):
   # Draw from a multinomial distribution efficiently
   #print("Shape of gumbel input: ", logits.shape)
   return logits + sample_gumbel(logits.size(), out=logits.data.new())
   return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]

# Avasaram ledura yedava
class SequenceCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, inputs, targets):

        # (B,T,C) 
        losses = self.criterion(inputs, targets)
        return losses.sum()/inputs.shape[0]


def ensure_frameperiod(mel, sig, factor=80):
   length = mel.shape[0]
   l = len(sig)

   if float(factor * length) == l:
      return sig, mel
   else:
      num_samples = factor * length
      if num_samples > l:
         difference = int((num_samples - l))
         for k in range(difference):
           sig =  np.append(sig, sig[-1])
         return sig, mel

      elif num_samples < l:
         difference = int((l - num_samples))
         return sig[:len(sig)-difference], mel
         
      else:
         print("This is hard")
         sys.exit()         
     
 
def read_pmfile(file):
   f = open(file)
   lines = f.readlines()
   timestamp_array = []
   for i, line in enumerate(lines):
       if i > 9:
          pitch_mark = line.split('\n')[0].split()[0]
          timestamp_array.append(pitch_mark)
   return timestamp_array

