from sklearn.metrics import recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility to return predictions
def return_classes(logits, dim=-1):
   #print(logits.shape)
   _, predicted = torch.max(logits,dim)    
   #print(predicted.shape)
   return predicted 

def get_metrics(predicted_tensor, target_tensor):
   predicteds = predicted_tensor
   targets = target_tensor
   return recall_score(predicteds, targets,average='macro')

# Utility to return predictions when using MoL loss
def return_classes_MoL(logits, dim=-1):
   logits = logits[:,:2]
   _, predicted = torch.max(logits,dim)    
   return predicted 

class logistic_loss(nn.Module):

    def __init__(self):
        super(logistic_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    # Input are 6 dimensional. Targets are three dimensional
    def forward(self, input, target):
        #print("Shapes of input and target: ", input.shape, target.shape)
        mixture_components = input[:,:3]
        logits = input[:,3:]
        #print("Shapes of logits and target: ", logits.shape, target.shape)
        ce_loss = self.criterion(logits, target)
        component_softmax = F.log_softmax(mixture_components, -1)
        #print("CE Loss: ", ce_loss)
        #print("Shape of component_softmax: ", component_softmax.shape)
        return ce_loss + -1.0 * torch.sum(log_sum_exp(component_softmax))



def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

