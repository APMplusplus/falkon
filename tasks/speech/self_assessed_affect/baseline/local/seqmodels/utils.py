from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.utils import to_categorical
import sys

# Utility to return predictions
def return_classes(logits, dim=-1):
   #print(logits.shape)
   _, predicted = torch.max(logits,dim)    
   #print(predicted.shape)
   return predicted 

def get_metrics(predicted_tensor, target_tensor):
   predicteds = predicted_tensor
   targets = target_tensor
   print(classification_report(predicteds, targets))
   return recall_score(predicteds, targets,average='macro')

# Utility to return predictions when using MoL loss
def return_classes_MoL(logits, dim=-1):
   mixtures = logits[:,:3]
   means = logits[:,3:]
   #print("Shapes of mixtures and means: ", mixtures.shape, means.shape)
   assert mixtures.shape[-1] == 3
   assert means.shape[-1] == 3

   _, predicted_mixtures = torch.max(mixtures,dim)    
   predicted_mixtures_onehotk = get_onehotk_tensor(predicted_mixtures, 3)
   #print("Shape of predicted_mixtures_onehotk: ", predicted_mixtures_onehotk.shape, predicted_mixtures_onehotk)
   assert predicted_mixtures_onehotk.shape[-1] == 3
   means = torch.sum(means * predicted_mixtures_onehotk, dim = -1)
   #print(means)
   #sys.exit()
   return predicted_mixtures

class logistic_loss(nn.Module):

    def __init__(self):
        super(logistic_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()

    # Input are 6 dimensional. Targets are three dimensional
    def forward(self, input, target):
        #print("Shapes of input and target: ", input.shape, target.shape)
        mixture_components = input[:,:3]
        logits = input[:,3:]
        #print("Shapes of logits and target: ", logits.shape, target.shape)
        target_onehotk = get_onehotk_tensor(target)
        ce_loss = self.criterion(logits, target_onehotk)
        component_softmax = F.log_softmax(mixture_components, -1)
        #print("CE Loss: ", ce_loss)
        #print("Shape of component_softmax: ", component_softmax.shape)
        return ce_loss + -1.0 * torch.sum(log_sum_exp(component_softmax))


def get_onehotk_tensor(A, num_classes = 2):
   a = A.cpu().numpy()
   #print(a.shape)
   a = to_categorical(a, num_classes)
   #print(a.shape)
   A = torch.FloatTensor(a).cuda()
   #print(A.shape)
   return A

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

