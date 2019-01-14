from sklearn.metrics import recall_score
import torch

# Utility to return predictions
def return_classes(logits, dim=-1):
   #print(logits.shape)
   _, predicted = torch.max(logits,dim)    
   #print(predicted.shape)
   return predicted 

def get_metrics(predicted_tensor, target_tensor):
   predicteds = predicted_tensor.cpu().numpy()
   targets = target_tensor.cpu().numpy()
   #print(predicteds, targets)
   return recall_score(predicteds, targets,average='macro')

def get_eer(predicted_tensor, target_tensor):
   predicteds = predicted_tensor.cpu().numpy()
   targets = target_tensor.cpu().numpy()
   fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
   EER = threshold(np.argmin(abs(tpr-fpr)))
   return EER
