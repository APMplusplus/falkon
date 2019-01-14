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
log_flag = 1
write_intermediate_flag = 0
phones_dict =  defaultdict(lambda: len(phones_dict))
phones_dict[0]
print_flag = 0

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
          phones = [ phones_dict[k] for k in line.split()[1:]]
          self.phones_array.append(np.array(phones))

    def __getitem__(self, index):
          return self.feats_array[index], self.phones_array[index]

    def __len__(self):
           return len(self.phones_array)

def collate_fn_padding(batch):
    feats_lengths = [len(x[0]) for x in batch]
    phones_lengths = [len(x[1]) for x in batch]
    max_feats_len = np.max(feats_lengths)
    max_phones_len = np.max(phones_lengths)
    
    
    a = np.array( [ _pad_ccoeffs(x[0], max_feats_len)  for x in batch ], dtype=np.float)
    b = np.array( [ _pad(x[1], max_phones_len)  for x in batch ], dtype=np.int)
    a_batch = torch.FloatTensor(a)
    b_batch = torch.LongTensor(b)

    return a_batch, b_batch

def _pad(seq, max_len):
    if seq.shape[0] < max_len:
        return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)
    else:
       mid_point = int(seq.shape[0]/2.0)
       seq = seq[mid_point - int(max_len/2): mid_point - int(max_len/2) + max_len]
       return seq
       #return seq[mid_point - int(max_len/2): mid_point + int(max_len/2)] 


def _pad_ccoeffs(seq, max_len):

    if seq.shape[0] < max_len:
       kk = np.zeros((max_len-seq.shape[0], seq.shape[1]), dtype='float32')
       return np.concatenate((seq,kk),axis = 0)
        
    else:
       mid_point = int(seq.shape[0]/2.0)
       return seq[mid_point - int(max_len/2): mid_point - int(max_len/2) + max_len] 



tdd_file = ETC_DIR + '/tdd.phseq.train'
feats_dir = BASE_DIR + '/feats/rms_arctic_5msec'
train_set = text2speech_am_dataset(tdd_file, feats_dir)
train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_padding
                          )

tdd_file = ETC_DIR + '/tdd.phseq.test'
val_set = text2speech_am_dataset(tdd_file, feats_dir)
val_loader = DataLoader(val_set,
                          batch_size=4,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_padding
                          )

#for i, (feats, phones) in enumerate(train_loader):
#    print(i, feats.shape, phones.shape)

## Model
model = attentionlstm(len(phones_dict))
print(model)
if torch.cuda.is_available():
   model.cuda()
criterion = nn.MSELoss()
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = optimizer_adam
updates = 0



def val():
 model.eval()
 l = 0
 with torch.no_grad(): 
   for i, (ccoeffs,phones) in enumerate(val_loader):

    ccoeffs = torch.FloatTensor(ccoeffs)
    phones = torch.LongTensor(phones)
    ccoeffs, phones = Variable(ccoeffs), Variable(phones)
    if torch.cuda.is_available():
        ccoeffs = ccoeffs.cuda()
        phones = phones.cuda()

    ccoeffs_predicted = model(phones, ccoeffs)
    optimizer.zero_grad()
    loss = criterion(ccoeffs_predicted, ccoeffs)
    l += loss.item()
    
    if log_flag:
       logger.scalar_summary('Val Loss', l * 1.0 / (i+1) , updates)  
    
 return l/(i+1)

def train():
  model.train()
  optimizer.zero_grad()
  start_time = time.time()
  l = 0
  global updates
  for i, (ccoeffs,phones) in enumerate(train_loader):
    updates += 1

    ccoeffs = torch.FloatTensor(ccoeffs)
    phones = torch.LongTensor(phones)
    ccoeffs, phones = Variable(ccoeffs), Variable(phones)
    if torch.cuda.is_available():
        ccoeffs = ccoeffs.cuda()
        phones = phones.cuda()

    ccoeffs_predicted = model(phones, ccoeffs)
    if print_flag:
        print("Shape of ccoeffs and ccoeffs_predicted: ", ccoeffs.shape, ccoeffs_predicted.shape)
    optimizer.zero_grad()
    loss = criterion(ccoeffs_predicted, ccoeffs)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
    l += loss.item()
    
    if i % 10 == 1:
        print(" Train Loss after processing ", updates, " batches: ", l/(i+1))
        
    if log_flag:
       logger.scalar_summary('Train Loss', l * 1.0 / (i+1) , updates)           
    
  return l/(i+1)


for epoch in range(max_epochs):
    epoch_start_time = time.time()
    train_loss = train()
    val_loss = val()
    g = open(logfile_name,'a')
    g.write("Train loss after epoch " + str(epoch) + ' ' + str(train_loss)  + " and the val loss: " + str(val_loss) + ' It took ' +  str(time.time() - epoch_start_time) + '\n')
    g.close()
    
    if epoch % 10 == 1:
     fname = model_name + '_epoch_' + str(epoch).zfill(3) + '.pth'
     with open(fname, 'wb') as f:
      torch.save(model, f)