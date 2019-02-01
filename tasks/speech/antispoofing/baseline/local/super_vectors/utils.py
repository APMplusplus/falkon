from sklearn.metrics import *
import torch
import numpy as np
from keras.utils import to_categorical
import sys

# Utility to return predictions
def return_classes(logits, dim=-1):
   #print(logits.shape)
   _, predicted = torch.max(logits,dim)    
   #print(predicted.shape)
   return predicted 

def return_valsnclasses(logits, dim=-1):
   #print(logits.shape)
   vals, predicted = torch.max(logits,dim)    
   #print(predicted.shape)
   return vals, predicted 

def get_metrics(predicteds, targets):
   print(classification_report(targets, predicteds))
   fpr, tpr, threshold = roc_curve(targets, predicteds, pos_label=1)
   EER = threshold[np.argmin(np.absolute(tpr-fpr))]
   print("EER is ", EER)
   return recall_score(predicteds, targets, average='macro'), \
          precision_score(predicteds, targets, average='macro'), \
          f1_score(predicteds, targets, average='macro'), \
          accuracy_score(predicteds, targets)


def get_eer(predicted_tensor, target_tensor):
   predicteds = predicted_tensor.cpu().numpy()
   targets = target_tensor.cpu().numpy()
   fpr, tpr, threshold = roc_curve(targets, predicteds, pos_label=1)
   #print(fpr, tpr)
   EER = threshold[np.argmin(np.absolute(tpr-fpr))]
   return EER

def get_onehotk_tensor(A, num_classes = 2):
   a = A.cpu().numpy()
   #print(a.shape)
   a = to_categorical(a)
   #print(a.shape)
   A = torch.FloatTensor(a).cuda()
   #print(A.shape)
   return A
