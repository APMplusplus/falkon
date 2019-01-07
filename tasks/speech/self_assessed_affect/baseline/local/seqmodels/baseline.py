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

## Locations
FALCON_DIR = os.environ.get('FALCON_DIR')
BASE_DIR = os.environ.get('base_dir')
DATA_DIR = os.environ.get('data_dir')
EXP_DIR = os.environ.get('exp_dir')
assert ( all (['FALCON_DIR', 'BASE_DIR', 'DATA_DIR', 'EXP_DIR']) is not None)

FEATS_DIR = BASE_DIR + '/feats/world_feats_20msec'
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
max_epochs = 100
updates = 0
plot_flag = 1
write_intermediate_flag = 1


## Data loaders

class selfassessed_dataset(Dataset):

    def __init__(self, tdd_file = ETC_DIR + '/filenames.train.tdd', wav_dir = DATA_DIR + '/wav', ccoeffs_dir= BASE_DIR +'/feats/world_feats_20msec'):

        self.tdd_file = tdd_file
        self.wav_dir = wav_dir
        self.mfcc_dir = ccoeffs_dir
        self.filenames_array = []
        self.labels_array = []
        f = open(self.tdd_file)
        fnames = f.readlines()
        self.label_file = self.tdd_file.replace('filenames', 'labels')
        g = open(self.label_file)
        labels = g.readlines()
        for (fname, label) in list(zip(fnames, labels)):
          fname = fname.split('\n')[0]
          fname = fname.split()[0]
          self.filenames_array.append(fname)
          label = label.split('\n')[0]
          self.labels_array.append(label.split()[1])

    def __getitem__(self, index):
          return self.filenames_array[index], self.labels_array[index]

    def __len__(self):
           return len(self.filenames_array)


tdd_file = ETC_DIR + '/filenames.train.tdd'
wav_dir = DATA_DIR + '/wav'
ccoeffs_dir= BASE_DIR +'/feats/world_feats_20msec'
train_set = selfassessed_dataset(tdd_file, wav_dir, ccoeffs_dir)
train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4
                          )

tdd_file = ETC_DIR + '/filenames.val.tdd'
wav_dir = DATA_DIR + '/wav'
ccoeffs_dir= BASE_DIR +'/feats/world_feats_20msec'
val_set = selfassessed_dataset(tdd_file, wav_dir, ccoeffs_dir)
val_loader = DataLoader(val_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=1
                          )


## Model
model = baseline_lstm()
print(model)
if torch.cuda.is_available():
   model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = optimizer_adam
updates = 0
label_dict = defaultdict(int)


## Val and Train
def val():
  model.eval()
  with torch.no_grad():
    l = 0
    global updates
    for i, data in enumerate(val_loader):
      updates += 1
      inputs_batch = []
      targets_batch = []
      files, labels = data[0], data[1]
      for (file,label) in list(zip(files, labels)):

        ccoeffs_file = FEATS_DIR + '/' + file + '.ccoeffs_ascii'
        ccoeffs = np.loadtxt(ccoeffs_file, usecols=range(1,61))

        start_frame = np.random.randint(len(ccoeffs) - max_timesteps)
        end_frame = start_frame + max_timesteps
        c = ccoeffs[start_frame:end_frame]   
        label_int = label_dict[label]
        inputs_batch.append(c)
        targets_batch.append(label_int) 

      inputs = torch.FloatTensor(inputs_batch)
      targets = torch.LongTensor(targets_batch)
      inputs, targets = Variable(inputs), Variable(targets)
      if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()

      logits = model(inputs)
      loss = criterion(logits, targets)
      l += loss.item()
  return l/(i+1)


def train():
  model.train()
  optimizer.zero_grad()
  start_time = time.time()
  l = 0
  global updates
  for i, data in enumerate(train_loader):
    updates += 1
    inputs_batch = []
    targets_batch = []
    files, labels = data[0], data[1]
    for (file,label) in list(zip(files, labels)):

        ccoeffs_file = FEATS_DIR + '/' + file + '.ccoeffs_ascii'
        ccoeffs = np.loadtxt(ccoeffs_file, usecols=range(1,61))

        start_frame = np.random.randint(len(ccoeffs) - max_timesteps)
        end_frame = start_frame + max_timesteps
        c = ccoeffs[start_frame:end_frame]   
        label_int = label_dict[label]
        inputs_batch.append(c)
        targets_batch.append(label_int) 

    inputs = torch.FloatTensor(inputs_batch)
    targets = torch.LongTensor(targets_batch)
    inputs, targets = Variable(inputs), Variable(targets)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()

    logits = model(inputs)
    optimizer.zero_grad()
    loss = criterion(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
    l += loss.item()
  
    # This 10 cannot be hardcoded
    if i % 10 == 1 and write_intermediate_flag:
       g = open(logfile_name, 'a')
       g.write("  Train loss after " + str(updates) +  " batches: " + str(l/(i+1)) + ". It took  " + str(time.time() - start_time) + '\n')
       g.close()

  return l/(i+1)  


def main():
  for epoch in range(max_epochs):
    epoch_start_time = time.time()
    train_loss = train()
    val_loss = val()

    g = open(logfile_name,'a')
    g.write("Train loss after epoch " + str(epoch) + ' ' + str(train_loss)  + " and the val loss: " + str(val_loss) + ' It took ' +  str(time.time() - epoch_start_time) + '\n')
    g.close()

main()    
