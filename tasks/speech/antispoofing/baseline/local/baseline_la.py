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

## Locations
FALCON_DIR = os.environ.get('FALCON_DIR')
BASE_DIR = os.environ.get('base_dir')
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
model_name = exp_dir + '/models/model_' + exp_name + '_'
max_timesteps = 100
max_epochs = 10
updates = 0
plot_flag = 1
write_intermediate_flag = 1
label_dict = defaultdict(int, bonafide=0, spoof=1)
int2label = {i:w for w,i in label_dict.items()}

fnames_array = []

class antispoofing_dataset(Dataset):
    
    def __init__(self, tdd_file = ETC_DIR + '/tdd.la.train', feats_dir=FEATS_DIR):

        self.tdd_file = tdd_file
        self.feats_dir = feats_dir
        self.labels_array = []
        self.feats_array = [] 
        f = open(self.tdd_file)
        for line in f:
          line = line.split('\n')[0]
          fname = line.split()[0]
          fnames_array.append(fname)
          feats_fname = feats_dir + '/' + fname + '.npz'
          feats = np.load(feats_fname)
          feats = feats['arr_0']
          self.feats_array.append(feats)
          label = line.split()[1]
          self.labels_array.append(label)

    def __getitem__(self, index):
          return self.feats_array[index], self.labels_array[index]

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

tdd_file = ETC_DIR + '/tdd.la.train'
train_set = antispoofing_dataset(tdd_file)
train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_chopping
                          )

tdd_file = ETC_DIR + '/tdd.la.dev'
val_set = antispoofing_dataset(tdd_file)
val_loader = DataLoader(val_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=1,
                          collate_fn=collate_fn_chopping
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

def val():
  model_name = exp_dir + '/models/' + '/model_exp_baseline__epoch_008.pth'
  with open(model_name, 'rb') as f:
      model = torch.load(f)
  model.eval()
  g = open('t', 'w')
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
      vals, predicteds = return_valsnclasses(logits)
      g.write(str(fnames_array[i]) + ' - ' + str(int2label[predicteds.item()]) + ' ' + str(vals.item()) + '\n')

      if i % 300 == 1:
         print("Processed ", i, " files")

  #recall = get_eer(predicteds, targets)
  #print("EER for the validation set:  ", recall)
  g.close()
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
    #val_loss = val()
    val_loss = 0
    g = open(logfile_name,'a')
    g.write("Train loss after epoch " + str(epoch) + ' ' + str(train_loss)  + " and the val loss: " + str(val_loss) + ' It took ' +  str(time.time() - epoch_start_time) + '\n')
    g.close()

    fname = model_name + '_epoch_' + str(epoch).zfill(3) + '.pth'
    with open(fname, 'wb') as f:
      torch.save(model, f)

def debug():
   val()

#main()    
debug()
