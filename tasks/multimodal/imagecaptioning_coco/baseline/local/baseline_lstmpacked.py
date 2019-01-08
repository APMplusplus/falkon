import torch
import torch.nn as nn
import numpy as np
import os, sys
import pickle
from utils import get_loader 
from model import *
import time
from torch.nn.utils.rnn import *

## Locations
FALCON_DIR = os.environ.get('FALCON_DIR')
BASE_DIR = os.environ.get('base_dir')
DATA_DIR = os.environ.get('data_dir')
EXP_DIR = os.environ.get('exp_dir')
assert ( all (['FALCON_DIR', 'BASE_DIR', 'DATA_DIR', 'EXP_DIR']) is not None)

FEATS_DIR = BASE_DIR + '/feats/world_feats_20msec'
ETC_DIR = BASE_DIR + '/etc'

sys.path.append(FALCON_DIR)
from src.nn import logger as l

## Flags
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
max_epochs = 100
updates = 0
log_flag = 1
debug_flag = 0

## Files
vocab_file = DATA_DIR + '/vocab.pkl'
imageid2captions_train_file = DATA_DIR + '/imageid2captions.pkl'
imageid2features_train_file = DATA_DIR  + '/imageid2features.pkl'
imageid2captions_val_file = DATA_DIR + '/imageid2captions_val.pkl'
imageid2features_val_file = DATA_DIR + '/imageid2features_val.pkl'

## Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load the files
with open(vocab_file, 'rb') as f:
    vocab = pickle.load(f)

with open(imageid2captions_train_file, 'rb') as f:
    imageid2captions_train = pickle.load(f)

with open(imageid2features_train_file, 'rb') as f:
    imageid2features_train = pickle.load(f)

with open(imageid2captions_val_file, 'rb') as f:
    imageid2captions_val = pickle.load(f)

with open(imageid2features_val_file, 'rb') as f:
    imageid2features_val = pickle.load(f)


## Dataloaders
train_loader = get_loader(i2f_dict=imageid2features_train,
                             i2c_dict=imageid2captions_train,
                             vocab=vocab,
                             transform=None,
                             batch_size=32,
                             shuffle=True,
                             num_workers=4)

val_loader = get_loader(i2f_dict=imageid2features_val,
                             i2c_dict=imageid2captions_val,
                             vocab=vocab,
                             transform=None,
                             batch_size=1,
                             shuffle=True,
                             num_workers=4)

## Model and stuff
feature_size = 2048
embed_size = 256
hidden_size = 128
model = CaptionRNN(feature_size,embed_size, hidden_size,len(vocab),2).to(device)
print(model)
criterion = nn.CrossEntropyLoss(ignore_index=0)
params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr = 0.001)
updates = 0

def generate(features, captions):
    
    features = features.unsqueeze(0)
    captions = captions.unsqueeze(0)
    
    # Generate an caption from the image
    sampled_ids = model.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    #print("Shape of sampled ids during generation: ", sampled_ids.shape, sampled_ids)

    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        #print("Word Id is: ", word_id)
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
             break
    predicted_sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print ("I predicted: ", predicted_sentence)
    
    sampled_ids = captions
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    
    # Convert word_ids to words
    original_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        original_caption.append(word)
        if word == '<end>':
             break
    original_sentence = ' '.join(original_caption)
    
    # Print out the image and the generated caption
    print ("Original Sentence: ", original_sentence)
    bleu_score_sentence = return_sentence_bleu(original_caption, sampled_caption)
    print("Smoothed BLEU score of a sample sentence is ", bleu_score_sentence)
    print('\n')
    
    return


## Validation
def val(partial_flag= 1):
    model.eval()
    l = 0
    for i, (features, captions, lengths, image_names) in enumerate(val_loader):
                  
            features = features.to(device)
            captions = captions.to(device)
            outputs = model.sample(features,return_logits=1)
            bsz = features.shape[0]
            outputs = outputs[:captions.shape[1],:,:]
            outputs = outputs.squeeze(1)
            #print("Shape of outputs and captions: ", outputs.shape, captions.shape)
            loss = criterion(outputs,captions.reshape(captions.shape[0]*captions.shape[1]))
            l += loss.item()
            
            if i == 1 and partial_flag:
               #print("  Val loop: After ", i, " batches, loss: ", l/(i+1))
               output = model.sample(features)
               sampled_ids = torch.max(outputs,dim=1)[1]
               sampled_ids = sampled_ids.cpu().numpy()
               sampled_caption = []
               for word_id in sampled_ids:
                  word = vocab.idx2word[word_id]
                  sampled_caption.append(word)
                  if word == '<end>':
                     break
               sentence = ' '.join(sampled_caption)
               predicted_caption = sampled_caption
               
    
               # Print out the image and the generated caption
               print ("  Val mein Predicted: ", sentence)
               
               outputs = captions[0,:]
               sampled_ids = outputs
               sampled_ids = sampled_ids.cpu().numpy()
               sampled_caption = []
               for word_id in sampled_ids:
                  word = vocab.idx2word[word_id]
                  sampled_caption.append(word)
                  if word == '<end>':
                   break
               sentence = ' '.join(sampled_caption)
               #generate(features, captions)
               original_caption = sampled_caption
               # Print out the image and the generated caption
               print ("  Val mein Original: ", sentence)
               bleu_score_sentence = return_sentence_bleu(original_caption, predicted_caption)
               if log_flag:
                   logger.scalar_summary('Dev BLEU', float(bleu_score_sentence)*100 , updates)     
               print("Smoothed BLEU score of a sample sentence is ", float(bleu_score_sentence)*100)
               print('\n')
               return l/(i+1)
     
    return l/(i+1)
    
## Train 
def train():
    model.train()
    global updates 
    l = 0
    for i, (features, captions, lengths, image_names) in enumerate(train_loader):
            
            updates += 1
                        
            features = features.to(device)
            captions = captions.to(device)
            outputs = model(features, captions, lengths)
            bsz = features.shape[0]
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            #print("Shape of outputs and captions: ", outputs.shape, captions.shape)
            loss = criterion(outputs,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            l += loss.item()
            
            if debug_flag and i% 300 == 1:
               captions = captions_bkp
               outputs = pad_packed_sequence(outputs_bkp,batch_first=True)
               captions = captions[0,:]
               outputs = outputs[0,:,:]
               sampled_ids = torch.max(outputs,dim=1)[1]
               sampled_ids = sampled_ids.cpu().numpy()
               sampled_caption = []
               for word_id in sampled_ids:
                  word = vocab.idx2word[word_id]
                  sampled_caption.append(word)
                  if word == '<end>':
                     break
               sentence = ' '.join(sampled_caption)
    
               # Print out the image and the generated caption
               print ("  Train mein Predicted: ", sentence)

               outputs = captions
               sampled_ids = outputs
               sampled_ids = sampled_ids.cpu().numpy()
               sampled_caption = []
               for word_id in sampled_ids:
                  word = vocab.idx2word[word_id]
                  sampled_caption.append(word)
                  if word == '<end>':
                   break
               sentence = ' '.join(sampled_caption)
    
               # Print out the image and the generated caption
               print ("  Train mein Original: ", sentence)
                

     
    return l/(i+1)
    
    
    
## Main Loop
for epoch in range(max_epochs):
    epoch_start_time = time.time()
    train_loss = train()
    val_loss = val()
    g = open(logfile_name, 'a')
    g.write("Epoch: " +  str(epoch).zfill(3) + " Train Loss: " +  str(train_loss) +  " Val Loss: " +  str(val_loss)  +  " Time per epoch: " + str(time.time() - epoch_start_time) + " seconds" + '\n')    
    g.close()   
 

