import numpy as np
from keras.utils import to_categorical
import torch
from sklearn.metrics import *

def return_frames(arr, ccoeffs, period=25, shift=10):

   x_array = []
   c_array = []
   num_frames = int((len(arr)- period)/shift) + 1
   #print("The number of frames from signal: ", num_frames, " and that from the ccoeffs: ", len(ccoeffs))
   assert num_frames == len(ccoeffs)
   start_sample = 0
   for i in range(num_frames):
       end_sample = start_sample + period
       #print("Start sample, end sample: ", start_sample, end_sample)
       start_sample += shift
       x_array.append(arr[start_sample:end_sample])
       c_array.append(ccoeffs[i])  
   return x_array, c_array

#A = np.linspace(1,100,100)
#return_frames(A)

def get_onehotk_tensor(A, num_classes = 2):
   a = A.cpu().numpy()
   #print(a.shape)
   a = to_categorical(a)
   #print(a.shape)
   A = torch.FloatTensor(a).cuda()
   #print(A.shape)
   return A


# Utility to return predictions
def return_classes(logits, dim=-1):
   #print(logits.shape)
   _, predicted = torch.max(logits,dim)    
   #print(predicted.shape)
   return predicted 

def get_metrics(predicteds, targets):
   print(classification_report(targets, predicteds))
   return recall_score(predicteds, targets,average='macro')
