import argparse
import numpy as np
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import baseline_lstm
import time
from collections import defaultdict
from utils import *
import pickle
import torch.nn.functional as F


## Locations
FALCON_DIR = os.environ.get('FALCON_DIR')
BASE_DIR = os.environ.get('base_dir') #if args.base_dir == '' else args.base_dir
DATA_DIR = os.environ.get('data_dir')
EXP_DIR = os.environ.get('exp_dir')
FEATS_DIR = os.environ.get('feats_dir')
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
model_name = exp_dir + '/models/model_' + exp_name
max_timesteps = 100
max_epochs = 100
updates = 0
plot_flag = 1
write_intermediate_flag = 1
log_flag = 1
fnames_array = []

class enhancement_dataset(Dataset):

    def __init__(self, tdd_file, src_dir, tgt_dir):

        self.tdd_file = tdd_file
        self.src_array = []
        self.tgt_array = []
        f = open(self.tdd_file)
        for line in f:
          line = line.split('\n')[0]  # removes trailing '\n' in line
          fname = line.split()[0]
          fnames_array.append(fname)
          src_fname =  src_dir + '/' + fname + '.mfcc'
          tgt_fname = tgt_dir + '/' + fname + '.mfcc' 
          src_feats = np.loadtxt(src_fname)
          tgt_feats = np.loadtxt(tgt_fname)
          self.src_array.append(src_feats)
          self.tgt_array.append(tgt_feats)

    def __getitem__(self, index):
          return self.src_array[index], self.tgt_array[index]

    def __len__(self):
           return len(self.src_array)


def collate_fn_chopping(batch):

    input_lengths = [len(x[0]) for x in batch]
    min_input_len = np.min(input_lengths)


    a = np.array( [ x[0][:min_input_len]  for x in batch ], dtype=np.float)
    b = np.array( [ x[1][:min_input_len]  for x in batch ], dtype=np.float)
    a_batch = torch.FloatTensor(a)
    b_batch = torch.FloatTensor(b)
    return a_batch, b_batch

src_dir = '/home/srallaba/projects/speech_enhancement/feats/cleaned_clean_28spk'
tgt_dir = '/home/srallaba/projects/speech_enhancement/feats/cleaned_noisy_28spk'

tdd_file = ETC_DIR + '/tdd.train'
train_set = enhancement_dataset(tdd_file, src_dir, tgt_dir)
train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_chopping
                          )

tdd_file = ETC_DIR + '/tdd.test'
val_set = enhancement_dataset(tdd_file, src_dir, tgt_dir)
val_loader = DataLoader(val_set,
                          batch_size=16,
                          shuffle=False,
                          num_workers=1,
                          collate_fn=collate_fn_chopping
                          )


## Model
model = baseline_lstm()
print(model)
if torch.cuda.is_available():
   model.cuda()
criterion = nn.MSELoss()
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = optimizer_sgd
updates = 0


def train():
  model.train()
  optimizer.zero_grad()
  start_time = time.time()
  l = 0
  global updates
  for i, (ccoeffs,labels) in enumerate(train_loader):
    updates += 1

    inputs = torch.FloatTensor(ccoeffs)
    targets = torch.FloatTensor(labels)
    inputs, targets = Variable(inputs), Variable(targets)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()

    logits = model(inputs)
    optimizer.zero_grad()
    loss = criterion(logits, targets)
    l += loss.item() 
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
  
    # This 100 cannot be hardcoded
    if i % 100 == 1 and write_intermediate_flag:
       g = open(logfile_name, 'a')
       g.write("  Train loss after " + str(updates) +  " batches: " + str(l/(i+1)) + ". It took  " + str(time.time() - start_time) + '\n')
       g.close()

    if log_flag:
            # Log the scalars
            logger.scalar_summary('Train Loss', l * 1.0 / (i+1) , updates) 

  return l/(i+1)  


def main():
  for epoch in range(max_epochs):
    epoch_start_time = time.time()
    train_loss = train()
    val_loss = 0
    g = open(logfile_name,'a')
    g.write("Train loss after epoch " + str(epoch) + ' ' + str(train_loss)  + " and the val loss: " + str(val_loss) + ' It took ' +  str(time.time() - epoch_start_time) +  '\n')
    g.close()

    fname = model_name + '_epoch_' + str(epoch).zfill(3) + '.pth'
    with open(fname, 'wb') as f:
      torch.save(model, f)

def debug():
   val()

main()    
#debug()
