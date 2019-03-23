import os
os.environ["CUDA_VISIBLE_DEVICIES"]="1"
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict
import numpy as np
import random
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import sys
from sklearn.metrics import classification_report

### Help:  Piazza posts: https://piazza.com/class/j9xdaodf6p1443?cid=2774
num_classes = 3
input_dim = 50
hidden = 256
batch_size = 4
print_flag = 1

# Process labels
labels_file = '/home1/srallaba/challenges/compare2018/ComParE2018_SelfAssessedAffect/lab/ComParE2018_SelfAssessedAffect.tsv'
labels = {}
ids = ['l','m','h']
f = open(labels_file)
cnt = 0 
for line in f:
  if cnt == 0:
    cnt+= 1
  else:
    line = line.split('\n')[0].split()
    fname = line[0].split('.')[0]
    lbl = ids.index(line[1])
    labels[fname] = lbl


# Process the dev
print("Processing Dev")
f = open('files.devel')
devel_input_array = []
devel_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file = '../feats/soundnet/' + line + '.npz'
    A = np.load(input_file, encoding='latin1')
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    devel_input_array.append(inp.astype(np.float32))
    devel_output_array.append(labels[line])

np.save('dev_input.npy', devel_input_array)
np.save('dev_output.npy', devel_output_array)


# Process the dev
print("Processing Train")
f = open('files.train')
train_input_array = []
train_output_array = []
for line in f:
    line = line.split('\n')[0]
    input_file =  '../feats/soundnet/' + line  + '.npz'
    A = np.load(input_file, encoding='latin1')
    a = A['arr_0']
    inp = np.mean(a[4],axis=0)
    print(inp.shape)
    train_input_array.append(inp.astype(np.float32))
    train_output_array.append(labels[line.split('.')[0]])

np.save('train_input.npy', train_input_array)
np.save('train_output.npy', train_output_array)


train_input_array = np.load('train_input.npy')
train_output_array = np.load('train_output.npy')
devel_input_array = np.load('dev_input.npy')
devel_output_array = np.load('dev_output.npy')


class COMPARE(Dataset):

    def __init__(self, A,B):
      
        self.input = A
        self.output = B
        #print(B)
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict

trainset = COMPARE(train_input_array,train_output_array)
devset = COMPARE(devel_input_array,devel_output_array)
train_loader = data_utils.DataLoader(trainset, batch_size=batch_size, shuffle=True)
dev_loader = data_utils.DataLoader(devset, batch_size=1, shuffle=False)


class kenyon_fc(nn.Linear):
  
     def __init__(self, *args, **kwargs):
        super(kenyon_fc, self).__init__(*args, **kwargs)
        
     def forward(self, x):
        y_batch = torch.zeros(x.shape[0],self.weight.shape[0])
        for i, k in enumerate(x):
          k = k.unsqueeze(0)
          y = k * self.weight
          y_top5 = torch.sum(torch.topk(y,5,dim=1)[0], dim=1)
          y_batch[i] = y_top5
        return y_batch


class encoder(nn.Module):
     
    def __init__(self, input_dim, hidden_dim):
       super(encoder, self).__init__()
       self.fc1 = nn.Linear(input_dim, hidden_dim)
       self.fc2 = nn.Linear(kenyon_dim, 3)
       self.fc_kenyon = kenyon_fc(hidden_dim, kenyon_dim)

    def forward(self, x):

       x = torch.tanh(self.fc1(x))
       x = self.fc_kenyon(x)
       return self.fc2(x)

embedding_dim = 3
hidden_dim = 128
kenyon_dim = 2000
vocab_size = 3
target_size = vocab_size
input_dim = 512
print("The target size is ", target_size)
baseline_encoder = encoder(input_dim, hidden_dim)
if torch.cuda.is_available():
   baseline_encoder + baseline_encoder.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(baseline_encoder.parameters()) , lr=0.01)
objective = nn.CrossEntropyLoss()

def train():
 total_loss = 0
 for ctr, t in enumerate(train_loader):
    a,b = t
    #print("Shape of input, output: ", a.shape, b.shape)
    #print("Type of input, output is ", a.data.type(), b.data.type())
    if torch.cuda.is_available():
       a,b = a.cuda(), b.cuda()
    pred = baseline_encoder(a.float())
    #print("Shape of encoder output:", pred.shape)
    #print("Batch Done")
    loss = criterion(pred, b)
    total_loss += loss.cpu().data.numpy()
    #if ctr % 100 == 1:
    #    print("Loss after ", ctr, "batches: ", total_loss/(ctr+1))
    optimizer.zero_grad()
    loss.backward()       
    optimizer.step()
 print("Train Loss is: ", total_loss/ctr) 
 test()
 print('\n')

def test():
 baseline_encoder.eval()
 total_loss = 0
 ytrue = []
 ypred = []
 for ctr, t in enumerate(dev_loader):
    a,b = t
    ytrue.append(b.data.numpy()[0])
    if torch.cuda.is_available():
       a,b = a.cuda(), b.cuda()
    pred = baseline_encoder(a.float())
    loss = criterion(pred, b)
    total_loss += loss.cpu().data.numpy()
    prediction = np.argmax(pred.cpu().data.numpy())
    ypred.append(prediction)
    #if ctr % 200 == 1:
    #   print ("Prediction, Original: ", prediction, b.cpu().data.numpy())
  
 #print(ytrue[0:10], ypred[0:10])
 print(classification_report(ytrue, ypred))
 print("Test Loss is: ", total_loss/ctr) 
 baseline_encoder.train()

for epoch in range(10):
    print("Running epoch ", epoch)
    train()
