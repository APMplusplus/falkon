'''

Script for baseline implementation of Vector Quantized Variational AutoEncoders
This is for speech generation. For now, its implemented on Voice Conversion 2018 dataset

'''

## Imports
import os, sys
import logging

from utils import *
from models import *

## Flags and locations
FALCON_DIR = os.environ.get('FALCON_DIR')
data_dir = '/home/srallaba/development/repos/falkon/paradigms/Variational_Inference/data/data_vcc2018_multispk/'
print_flag = 1

## Dataloaders and stuff
train_loader, val_loader = get_dataloaders(data_dir)

## Train loop
def train(model):

   for (audio, mel, spk) in train_loader:
       encoder_output = model(audio)
       print("Shape of encoder output: ", encoder_output.shape)

## Eval loop

## Main thingy
model = baseline_vqvae()
train(model)

