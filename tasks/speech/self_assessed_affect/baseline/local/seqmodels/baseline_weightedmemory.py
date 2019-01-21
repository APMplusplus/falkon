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
label_dict = defaultdict(int, l=0, m=1,h=2)
log_flag = 1

## Data loaders

class selfassessed_dataset(Dataset):

    def __init__(self, tdd_file = ETC_DIR + '/filenames.train.tdd', wav_dir = DATA_DIR + '/wav', ccoeffs_dir= BASE_DIR +'/feats/world_feats_20msec'):

        self.tdd_file = tdd_file
        self.wav_dir = wav_dir
        self.mfcc_dir = ccoeffs_dir
        self.ccoeffs_array = []
        self.labels_array = []
        f = open(self.tdd_file)
        fnames = f.readlines()
        self.label_file = self.tdd_file.replace('filenames', 'labels')
        g = open(self.label_file)
        labels = g.readlines()
        for (fname, label) in list(zip(fnames, labels)):
          fname = fname.split('\n')[0]
          fname = fname.split()[0]
          ccoeffs_fname = ccoeffs_dir + '/' + fname + '.ccoeffs_ascii'
          ccoeffs = np.loadtxt(ccoeffs_fname, usecols=range(1,61))
          self.ccoeffs_array.append(ccoeffs)
          label = label.split('\n')[0]
          self.labels_array.append(label.split()[1])

    def __getitem__(self, index):
          return self.ccoeffs_array[index], self.labels_array[index]

    def __len__(self):
           return len(self.labels_array)


def collate_fn_chopping(batch):
    input_lengths = [len(x[0]) for x in batch]
    min_input_len = np.min(input_lengths)

    a = np.array( [ x[0][:min_input_len]  for x in batch ], dtype=np.float)
    b = np.array( [ label_dict[x[1]]  for x in batch ], dtype=np.int)
    a_batch = torch.FloatTensor(a)
    b_batch = torch.LongTensor(b)
    return a_batch, b_batch


tdd_file = ETC_DIR + '/filenames.train.tdd'
wav_dir = DATA_DIR + '/wav'
ccoeffs_dir= BASE_DIR +'/feats/world_feats_20msec'
train_set = selfassessed_dataset(tdd_file, wav_dir, ccoeffs_dir)
train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_chopping
                          )

tdd_file = ETC_DIR + '/filenames.val.tdd'
wav_dir = DATA_DIR + '/wav'
ccoeffs_dir= BASE_DIR +'/feats/world_feats_20msec'
val_set = selfassessed_dataset(tdd_file, wav_dir, ccoeffs_dir)
val_loader = DataLoader(val_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=1,
                          collate_fn=collate_fn_chopping
                          )


## Model
model = attentionlstm()
print(model)
if torch.cuda.is_available():
   model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = optimizer_adam
updates = 0


## Val and Train
def val(epoch):
  model.eval()
  with torch.no_grad():
    l = 0
    global updates
    y_true = []
    y_pred = []
    for i, (ccoeffs, labels) in enumerate(val_loader):

      inputs = torch.FloatTensor(ccoeffs)
      targets = torch.LongTensor(labels)
      inputs, targets = Variable(inputs), Variable(targets)
      if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()

      logits = model.forward_eval(inputs)
      loss = criterion(logits, targets)
      y_true.append(targets)
      y_pred.append(logits)
      l += loss.item()

  predicteds = return_classes(logits)
  recall = get_metrics(predicteds, targets)
  print("Unweighted Recall for the validation set:  ", recall)
  
  if log_flag:
       logger.scalar_summary('Dev UAR', recall , epoch)     
       
  return l/(i+1)


def train():
  model.train()
  optimizer.zero_grad()
  start_time = time.time()
  l = 0
  global updates
  for i, (ccoeffs,labels) in enumerate(train_loader):
    updates += 1

    inputs = torch.FloatTensor(ccoeffs)
    targets = torch.LongTensor(labels)
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
    val_loss = val(epoch)

    g = open(logfile_name,'a')
    g.write("Train loss after epoch " + str(epoch) + ' ' + str(train_loss)  + " and the val loss: " + str(val_loss) + ' It took ' +  str(time.time() - epoch_start_time) + '\n')
    g.close()

def debug():
   val()

main()    
#debug()
