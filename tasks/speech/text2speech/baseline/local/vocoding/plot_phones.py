import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import *
from model_pmnet import *
import time
import sys, os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import soundfile as sf


## Locations
FALCON_DIR = os.environ.get('FALCON_DIR')
BASE_DIR = os.environ.get('base_dir')
DATA_DIR = os.environ.get('data_dir')
EXP_DIR = os.environ.get('exp_dir')
assert ( all (['FALCON_DIR', 'BASE_DIR', 'DATA_DIR', 'EXP_DIR']) is not None)

ETC_DIR = BASE_DIR + '/etc'
pm_dir = DATA_DIR + '/pm'

sys.path.append(FALCON_DIR)
from src.nn import logger as l

## Flags and stuff 
plot_flag = 1
# Add visualization for model like https://github.com/szagoruyko/functional-zoo/blob/master/resnet-18-export.ipynb


class arctic_dataset(Dataset):

    def __init__(self, tdd_file= '../etc/txt.done.data', wav_dir = '../wav', ccoeffs_dir='../mfcc_ascii'):
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

wav_dir = '/home/srallaba/development/cmu_us_rms/segments/ax/wav/'
ccoeffs_dir = '/home/srallaba/development/cmu_us_rms/segments/ax/ccoeffs/'
plot_dir = wav_dir + '../plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
    
train_set = arctic_dataset(BASE_DIR + '/etc/tdd.phones.train',wav_dir, ccoeffs_dir)
train_loader = DataLoader(train_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4
                          )


# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html





def plot_phones():
  for i, files in enumerate(train_loader):
    for file in files:
       #print(file) 
       
       wav_file = wav_dir + '/' + file + '.npy'
       x = np.load(wav_file)
       x_quantized = quantize_wav(x)
       
       if plot_flag:
       
              axes = plt.gca()
              axes.set_ylim([-0.9,0.9])
                            
              assert len(x) == len(x_quantized)
              
              plt.subplot(211)
              plt.plot(x)
              
              plt.subplot(212)
              plt.plot(x_quantized)
              
              plt.savefig(plot_dir + '/plot_' + 'phoneid_' + str(i).zfill(4) + '.png')
              plt.close()


plot_phones()
  
