'''
Script for baseline implementation of Vector Quantized Variational AutoEncoders
This is for speech generation. For now, its implemented on Voice Conversion 2018 dataset
'''

## Imports
import os, sys
import logging

from utils import *
from models import baseline_vqvae

## Flags and locations
FALCON_DIR = os.environ.get('FALCON_DIR')
data_dir = '/home/srallaba/development/repos/falkon/paradigms/variational_inference/vqvae/data/data_vcc2018_multispk/'
print_flag = 1

## Dataloaders and stuff
train_loader, val_loader = get_dataloaders(data_dir)

## Train loop
def train(model):

   for (audio, mel, spk) in train_loader:
       if torch.cuda.is_available():
          audio, mel, spk = audio.cuda(), mel.cuda(), spk.cuda()
       quantized_output = model(audio)
       if print_flag:
          print("Shape of quantized output: ", quantized_output.shape)
          print("Shape of audio: ", audio.shape) 


## Eval loop

## Main thingy
model = baseline_vqvae()
if torch.cuda.is_available():
   model.cuda()

train(model)

