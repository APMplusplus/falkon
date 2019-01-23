import numpy as np
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from collections import defaultdict
from utils import *
from model import *
from scipy.io import wavfile as wf

max_epochs = 100
frame_period = 400
frame_shift = 160

class arctic_dataset_wavNccoeffs(Dataset):

    def __init__(self, tdd_file, wav_dir, ccoeffs_dir):
        self.tdd_file = tdd_file
        self.wav_dir = wav_dir
        self.ccoeffs_dir = ccoeffs_dir
        self.feats_array = []
        self.samples_array = []
        f = open(self.tdd_file)
        for line in f:
          line = line.split('\n')[0]
          wav_fname = self.wav_dir + '/' + line.split()[0] + '.wav'
          ccoeffs_fname = self.ccoeffs_dir + '/' + line.split()[0] +  '.feats.mfcc.clean'
          #a = np.loadtxt(ccoeffs_fname)
          self.feats_array.append(ccoeffs_fname)
          #fs, A = wf.read(wav_fname)
          self.samples_array.append(wav_fname)


    def __getitem__(self, index):
          return self.samples_array[index], self.feats_array[index]

    def __len__(self):
           return len(self.samples_array)


tdd_file = 'tdd'
wav_dir = 'wav/'
ccoeffs_dir = '/home/srallaba/development/kitchens/kitchen_sincnet/feats_dumped'
train_set = arctic_dataset_wavNccoeffs(tdd_file, wav_dir, ccoeffs_dir)
train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4
                          )

updates = 0
model = baseline_mlp()
print(model)
if torch.cuda.is_available():
   model.cuda()
criterion = nn.MSELoss()
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = optimizer_adam


def train():
    start_time = time.time()
    l = 0
    global updates
    for i, (samples, ccoeffs) in enumerate(train_loader):
        updates += 1
        x_batch = []
        c_batch = []
        for (wfname, cfname) in list(zip(samples, ccoeffs)):
            c = np.loadtxt(cfname)
            fs, x = wf.read(wfname)
            x_array, c_array = return_frames(x, c, 400, 160)
            idx = np.random.choice(np.arange(len(x_array)))
            x = x_array[idx]
            c = c_array[idx]
            assert len(x) > 0
            x_batch.append(x)
            c_batch.append(c)

        x,y = x_batch, c_batch
        x = torch.FloatTensor(x)
        c = torch.FloatTensor(c)
        x, c = Variable(x), Variable(c)
        if torch.cuda.is_available():
          x = x.cuda()
          c = c.cuda()

        c_hat = model(x)
        loss = criterion(c_hat, c)
        l += loss.item()

    return l/(i+1)

for epoch in range(max_epochs):         
    train_loss = train()
    print("Train loss after epoch ", epoch, " is: ", train_loss)
