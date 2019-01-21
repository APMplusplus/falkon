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
exp_name = 'exp_phones'
exp_dir = EXP_DIR + '/' + exp_name
if not os.path.exists(exp_dir):
   os.mkdir(exp_dir)
   os.mkdir(exp_dir + '/logs')
   os.mkdir(exp_dir + '/models')
   os.mkdir(exp_dir + '/plots')
# This is just a command line utility
logfile_name = exp_dir + '/logs/log_' + exp_name
print(logfile_name)
g = open(logfile_name, 'w')
g.close()
model_dir = exp_dir + '/models'
plot_dir = exp_dir + '/plots'
# This is for visualization
logger = l.Logger(exp_dir + '/logs/' + exp_name)
model_name = exp_dir + '/models/model_' + exp_name + '_'
max_epochs = 1000
updates = 0
log_flag = 1
debug_flag = 0
wav_dir = DATA_DIR + '/wav'
ccoeffs_dir = BASE_DIR + '/feats/rms_arctic_1msec'
print_flag = 1
debug_flag = 0
start_time = time.time()
write_intermediate_flag = 1
model_name = exp_dir + '/models/model_' + exp_name + '_'
log_flag = 1
plot_flag = 1
save_intermediate_flag = 0
updates = 0
max_timesteps = 8000
frame_period = 16

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

train_set = arctic_dataset(BASE_DIR + '/etc/tdd.phones.train',wav_dir, ccoeffs_dir)
train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4
                          )

val_set = arctic_dataset(BASE_DIR + '/etc/tdd.phones.test', wav_dir, ccoeffs_dir)
val_loader = DataLoader(val_set,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4
                          )


# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html



model = cnnmodel()
#model.double()
print(model)

if torch.cuda.is_available():
   model.cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0) # This is not ok but lets continue for now. use reduce=false though. 
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer = optimizer_adam





def val(partial_flag = 1):
 model.eval()
 l = 0
 with torch.no_grad():
    x_batch = []
    c_batch = []
    for i, file in enumerate(val_loader):
       file = file[0]
       print(file) 
       wav_file = wav_dir + '/' + file + '.npy'
       x = np.load(wav_file)
       mfcc_file = ccoeffs_dir + '/' + file + '.npy' 
       c = np.load(mfcc_file)
       x , c = ensure_frameperiod_pm(c , x, frame_period)
       c = c[:,1:61]
       
       if 10 > 2:
      
          
           
          x = torch.LongTensor(x)
          c = torch.FloatTensor(c)
          x, c = Variable(x), Variable(c)
          if torch.cuda.is_available():
             x = x.cuda()
             c = c.cuda()
          x = x.unsqueeze(0)
          c = c.unsqueeze(0)
          #print("Shape of x and c ", x.shape, c.shape)
          x_hat = model.forward_incremental(x, c, 1)
          x = x[:,1:]
          
          loss = criterion(x_hat.contiguous().view(-1,259), x.contiguous().view(-1))
          l += loss.item()
          #print("Shapes of original and predicted files are: ", x.shape, x_hat.shape, " and the loss is ", loss.item())

          if log_flag:
            # Log the scalars
            logger.scalar_summary('Val Loss', l * 1.0 / (i+1) , updates)   
          
          if plot_flag:
              axes = plt.gca()
              axes.set_ylim([-0.9,0.9])
              
              #x_hat = x_hat.reshape(x_hat.shape[0]*x_hat.shape[1], x_hat.shape[2])
              x_hat = torch.max(x_hat,2)[1]
              x_hat = x_hat.cpu().numpy()
              #print("Shape of x_hat: ", x_hat.shape)
              #x_hat = torch.max(x_hat,1)[1].cpu().numpy()
              #print("Shape of x_hat: ", x_hat.shape)
              x = x.data[0].cpu().numpy()
              
              x = inv_mulaw_quantize(x)
              x_hat = inv_mulaw_quantize(x_hat)
              
              assert len(x) == len(x_hat)
              
              plt.subplot(211)
              plt.plot(x)
              
              plt.subplot(212)
              plt.plot(x_hat)
              
              plt.savefig(plot_dir + '/plot_step_' + str(updates).zfill(4)  + '_phoneid_' + str(i).zfill(4) + '.png')
              plt.close()
              

          if partial_flag == 1:
             return l/(i+1)
          
          elif partial_flag == 2 and i == 20:   
            return l/(i+1)
        
       return l/(i+1)  
           
    if plot_flag:
       return
    return x[0], x_hat


def train():
  model.train()
  optimizer.zero_grad()
  start_time = time.time()
  l = 0
  global updates 
  for i, files in enumerate(train_loader):
    #print(i, files)
    updates += 1
    x_batch = []
    c_batch = []
    #print(updates)
    for file in files:
       #print(file) 
       
       wav_file = wav_dir + '/' + file + '.npy'
       mfcc_file = ccoeffs_dir + '/' + file + '.npy'
       x = np.load(wav_file)
       c = np.load(mfcc_file)
       #print(len(c), len(x))
       x, c = ensure_frameperiod_pm(c, x, 16)
       assert len(c) * frame_period == len(x)
       
       if len(x) > 0:
         #print(x.shape, c.shape)  
         x_batch.append(x)
         c_batch.append(c[:, 1:61])
       else:
          continue 
         
    x,c = x_batch, c_batch
    x = torch.LongTensor(x)
    c = torch.FloatTensor(c)
    x,c = Variable(x), Variable(c)

    if torch.cuda.is_available():
        x = x.cuda()
        c = c.cuda()

    x[:,0] = 0
    x_hat = model(x, c)
    x = x[:,1:]

    optimizer.zero_grad()
    loss = criterion(x_hat.contiguous().view(-1,259), x.contiguous().view(-1))
    #print(x_hat.shape, x.shape) 
    #loss = F.kl_div(x_hat.contiguous().view(-1,259).cpu(), x.contiguous().view(-1).cpu().float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()

    l += loss.item()

    if i % 100 == 1 and write_intermediate_flag:
       g = open(logfile_name, 'a')
       g.write("  Train loss after " + str(updates) +  " batches: " + str(l/(i+1)) + ". It took  " + str(time.time() - start_time) + '\n')
       g.close()
       #val(1)
       #model.train()

    if log_flag:
     
       # Log the scalars
       logger.scalar_summary('Train Loss', l * 1.0 / (i+1) , updates)       

       '''
       # Log the gradients
       for tag, value in model.named_parameters():
            #print(tag)
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), updates)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), updates)
       '''
       #print("Logging stuff")

  return l/(i+1)


def main():
  for epoch in range(max_epochs):
   epoch_start_time = time.time()
   train_loss = train()
   if 10 > 2: #epoch % 2 == 1 :
      val_loss = val(2)
      print(val_loss)
      g = open(logfile_name,'a')
      g.write("Train loss after epoch " + str(epoch) + ' ' + str(train_loss)  + " and the val loss: " + str(val_loss) + ' It took ' +  str(time.time() - epoch_start_time) + '\n')
      g.close()
   else:
      g = open(logfile_name,'a')
      g.write("Train loss after epoch " + str(epoch) + ' ' + str(train_loss)  + ' It took ' +  str(time.time() - epoch_start_time) + '\n')
      g.close() 
       

   if epoch % 10 == 1:
     fname = model_name + '_epoch_' + str(epoch).zfill(3) + '.pth'
     with open(fname, 'wb') as f:
      torch.save(model, f)

def debug():
    val(2)
    
  
main()  
#debug()    
#generate()
