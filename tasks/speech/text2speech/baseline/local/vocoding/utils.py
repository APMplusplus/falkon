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
from nltk.translate.bleu_score import *

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

# Avasaram ledura 
class SequenceCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def forward(self, inputs, targets):

        # (B,T,C) 
        losses = self.criterion(inputs, targets)
        return losses.sum()/inputs.shape[0]

def cut_slack(sig, mel, factor, T=8000, frame_period=256):

    if len(sig) == T:
       return sig, mel
    elif len(sig) > T:
       difference_samples = int(len(sig) - T)
       difference_frames = int(difference_samples / frame_period)
       #print("  In util: difference_samples, difference_frames: ", difference_samples, difference_frames)
       if difference_frames < 1:
          difference_frames = 1
       return ensure_frameperiod(mel[:-difference_frames,:], sig[:-difference_samples], factor)

def ensure_frameperiod(mel, sig, factor=80, T = 8000):
   length = mel.shape[0]
   l = len(sig)
   #print("  In util: Shapes of mel and sig are :", mel.shape, sig.shape, length, l, factor)
   if float(factor * length) == l:

      # Check if max time steps requirement is met 
      #sig, mel = cut_slack(sig, mel, factor)
      return sig, mel
   else:
      num_samples = factor * length
      if num_samples > T:
         #print("  In util: Number of samples after upsampling , ", num_samples, " will be greater than T ", T )
         # We dont want this 
         mel = mel[:-1,:]     
         #print("  In util: adjusted shapes are: ", mel.shape, sig.shape)
         return ensure_frameperiod(mel, sig, factor, T) 

      elif num_samples < T:
         # We dont want this case
         #difference = int((l - num_samples))
         #return sig[:len(sig)-difference], mel
         # Repeat the last frame of mel and check the condition
         #print("  In util: Number of samples after upsampling , ", num_samples, " will be less than T ", T )
         #a = mel[-1,:]
         #a = np.expand_dims(a,axis=0)
         #mel = np.append(mel,a,axis=0)
         difference = int((T - num_samples))
         for i in range(difference):
             sig = np.append(sig, 0)
         #print("  In util: adjusted shapes are: ", mel.shape, sig.shape)    
         return ensure_frameperiod(mel, sig, factor, T)

      else:
         print("This is hard")
         sys.exit()         
     
def ensure_frameperiod_pm(mel, sig, factor=80):
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

# https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/wavenet.py
def receptive_field_size(total_layers, num_cycles, kernel_size,
                         dilation=lambda x: 2**x):
    """
    layers = [ 10, 12]
    stacks = [2] 
    kernel_sizes = [3]

    for layer in layers:
       for stack in stacks:
         for kernel_size in kernel_sizes:
            print("Receptive field with ", layer, " layers ", " and ", stack, " stacks with kernel size ", kernel_size, " is ", receptive_field_size(layer, stack, kernel_size))
       print('\n')
      
    """
    assert total_layers % num_cycles == 0
    layers_per_cycle = total_layers // num_cycles
    dilations = [dilation(i % layers_per_cycle) for i in range(total_layers)]
    print("Dilations are: ", dilations)
    return (kernel_size - 1) * sum(dilations) + 1



      

def return_utterance_bleu(original_signal, sampled_signal, **kwargs):
    chencherry = SmoothingFunction()
    bleu_score_utterance = sentence_bleu([original_signal], sampled_signal, smoothing_function=chencherry.method1, **kwargs)
    return bleu_score_utterance

