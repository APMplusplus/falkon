from sklearn.metrics import *
import torch
import numpy as np

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

def get_metrics(predicted_tensor, target_tensor):
   predicteds = predicted_tensor.cpu().numpy()
   targets = target_tensor.cpu().numpy()
   #print(predicteds, targets)
   return recall_score(predicteds, targets,average='macro')

def get_eer(predicted_tensor, target_tensor):
   predicteds = predicted_tensor.cpu().numpy()
   targets = target_tensor.cpu().numpy()
   fpr, tpr, threshold = roc_curve(targets, predicteds, pos_label=1)
   #print(fpr, tpr)
   EER = threshold[np.argmin(np.absolute(tpr-fpr))]
   return EER
