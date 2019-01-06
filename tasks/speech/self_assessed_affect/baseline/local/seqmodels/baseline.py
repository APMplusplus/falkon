import numpy as np
import os, sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

## Locations
FALCON_DIR = os.environ.get('FALCON_DIR')
BASE_DIR = os.environ.get('base_dir')
DATA_DIR = os.environ.get('data_dir')
assert ( all (['FALCON_DIR', 'BASE_DIR', 'DATA_DIR']) is not None)

FEATS_DIR = BASE_DIR + '/feats/world_feats_20msec'
ETC_DIR = BASE_DIR + '/etc'

sys.path.append(FALCON_DIR)
from src.nn import *

## Flags


## Data loaders 

class selfassessed_dataset(Dataset):

    def __init__(self, tdd_file = ETC_DIR + '/filenames.train.tdd', wav_dir = DATA_DIR + '/wav', ccoeffs_dir= BASE_DIR +'/feats/world_feats_20msec'):

        self.tdd_file = tdd_file
        self.wav_dir = wav_dir
        self.mfcc_dir = ccoeffs_dir
        self.filenames_array = []
        f = open(self.tdd_file)
        for line in f:
          line = line.split('\n')[0]
          fname = line.split()[0]
          self.filenames_array.append(fname)

    def __getitem__(self, index):
          return self.filenames_array[index]

    def __len__(self):
           return len(self.filenames_array)


tdd_file = ETC_DIR + '/filenames.train.tdd'
wav_dir = DATA_DIR + '/wav'
ccoeffs_dir= BASE_DIR +'/feats/world_feats_20msec'
train_set = selfassessed_dataset(tdd_file, wav_dir, ccoeffs_dir)
train_loader = DataLoader(train_set,
                          batch_size=4,
                          shuffle=True,
                          num_workers=4
                          )

tdd_file = ETC_DIR + '/filenames.val.tdd'
wav_dir = DATA_DIR + '/wav'
ccoeffs_dir= BASE_DIR +'/feats/world_feats_20msec'
val_set = selfassessed_dataset(tdd_file, wav_dir, ccoeffs_dir)
val_loader = DataLoader(val_set,
                          batch_size=1,
                          shuffle=True,
                          num_workers=1
                          )


## Model


## Val and Train



