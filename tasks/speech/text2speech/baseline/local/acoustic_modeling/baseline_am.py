import numpy as np
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import *
import time
from collections import defaultdict
from utils import *

## Locations
FALCON_DIR = os.environ.get('FALCON_DIR')
BASE_DIR = os.environ.get('base_dir')
DATA_DIR = os.environ.get('data_dir')
EXP_DIR = os.environ.get('exp_dir')
assert ( all (['FALCON_DIR', 'BASE_DIR', 'DATA_DIR', 'EXP_DIR']) is not None)

ETC_DIR = BASE_DIR + '/etc'

sys.path.append(FALCON_DIR)
from src.nn import logger as l

## Flags and variables - This is not the best way to code log file since we want log file to get appended when reloading model
exp_name = 'exp_baseline'
exp_dir = EXP_DIR + '/' + exp_name
if not os.path.exists(exp_dir):
   os.mkdir(exp_dir)
   os.mkdir(exp_dir + '/logs')
   os.mkdir(exp_dir + '/models')
# This is just a command line utility
logfile_name = exp_dir + '/logs/log_' + exp_name
g = open(logfile_name, 'w')
g.close()
# This is for visualization
logger = l.Logger(exp_dir + '/logs/' + exp_name)
model_name = exp_dir + '/models/model_' + exp_name + '_'
max_timesteps = 100
max_epochs = 10
updates = 0
plot_flag = 1
write_intermediate_flag = 0


class text2speech_am_dataset(Dataset):
    
    def __init__(self, tdd_file = ETC_DIR + '/tdd.train', feats_dir='../feats/rms-arctic_5msec'):

        self.tdd_file = tdd_file
        self.feats_dir = feats_dir
        self.phones_array = []
        self.feats_array = [] 
        f = open(self.tdd_file)
        for line in f:
          line = line.split('\n')[0]
          fname = line.split()[0]
          feats_fname = feats_dir + '/' + fname + '.ccoeffs_ascii'
          feats = np.loadtxt(feats_fname)
          self.feats_array.append(feats)
          phones = line.split()[1:]
          self.phones_array.append(phones)

    def __getitem__(self, index):
          return self.feats_array[index], self.phones_array[index]

    def __len__(self):
           return len(self.phones_array)

tdd_file = ETC_DIR + '/tdd.phseq.train'
feats_dir = BASE_DIR + '/feats/rms-arctic_5msec'
train_set = text2speech_am_dataset(tdd_file, feats_dir)
train_loader = DataLoader(train_set,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4
                          )

for i, d in enumerate(train_loader):
    print(i)
