import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import sys
from layers import *
from modules import *
import numpy as np
from utils import *

class CaptionRNN(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=100):
        """Set the hyper-parameters and build the layers."""
        super(CaptionRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

        self.image_linear = nn.Linear(feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        features = self.bn(self.image_linear(features))
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        #print("Shape of hiddens: ", hiddens.shape)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None, return_logits=0):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        logits = []
        features = self.bn(self.image_linear(features))
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            #print(i)
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            logits.append(outputs)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        logits = torch.stack(logits)
        if return_logits:
          return logits
        return sampled_ids

class CaptionCNN(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(CaptionCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        layers = 12
        stacks = 4
        layers_per_stack = layers // stacks
        self.kernel_size = 3
        self.stride = 1
        self.vocab_size = vocab_size
           

        self.conv_modules = nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            self.padding = int((self.kernel_size - 1) * dilation)
            conv = residualconvmodule(embed_size,embed_size, self.kernel_size, self.stride, self.padding,dilation)
            self.conv_modules.append(conv)


        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

        self.image_linear = nn.Linear(feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        self.final_fc1 = SequenceWise(nn.Linear(256, 512))
        self.final_fc2 = SequenceWise(nn.Linear(512, vocab_size))


        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        features = self.bn(self.image_linear(features))

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        c = None
        x = embeddings.transpose(1,2)
        for module in self.conv_modules:
          x = F.relu(module(x, c))
        x = x.transpose(1,2)

        x = F.relu(self.final_fc1(x))
        x = self.final_fc2(x)

        return x
        return x[:,:-1,:]
    
    def clear_buffers(self):

       for module in self.conv_modules:
           module.clear_buffer()

    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""

        outputs = []
        features = self.bn(self.image_linear(features))
        inputs = features.unsqueeze(1)
        bsz = inputs.shape[0]
        
        self.clear_buffers()
        c = None
        x = inputs
        for i in range(self.max_seg_length):
            module_count = 0
            for module in self.conv_modules:
                module_count += 1
                #print("  Module: Feeding into the module number: ", module_count)
                x = F.relu(module.incremental_forward(x, c))
            x = F.relu(self.final_fc1(x))
            x = self.final_fc2(x)
            #print("Shape of x", x.shape)
            print("The value of x is ", x)
            v, predicted = torch.max(x,2)
            outputs.append(predicted[0])
            
        outputs = torch.stack(outputs, 1)
        return outputs

class CaptionSingleCNN(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=60):
        """Set the hyper-parameters and build the layers."""
        super(CaptionSingleCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.kernel_size = 5
        self.stride = 1
        self.vocab_size = vocab_size
        self.padding = self.kernel_size - 1  # Causal
        self.conv = AdvancedConv1d(embed_size, hidden_size, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation = 1)

        self.max_seg_length = max_seq_length

        self.image_linear = nn.Linear(feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        # https://github.com/Erlemar/cs231n_self/blob/master/assignment2/cs231n/layers.py#L116
        self.final_fc1 = SequenceWise(nn.Linear(hidden_size, 64))
        self.final_fc2 = SequenceWise(nn.Linear(64, vocab_size))

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        features = self.bn(self.image_linear(features)) # (32, 256)
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # (32,17,256)
        #print("Shape of embeddings: ", embeddings.shape) 
        x = embeddings.transpose(1,2)
        x = F.relu(self.conv(x))
        x = x.transpose(1,2) # (32,19,256)
        x = x[:,:-self.padding,:] # Causal
        #print("Shape of x: ", x.shape)

        x = F.relu(self.final_fc1(x))
        x = self.final_fc2(x)
        #print("Shape of x: ", x.shape)
        return x[:,:-1,:]

    def clear_buffers(self):
        self.conv.clear_buffer()


    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""

        outputs = []
        features = self.bn(self.image_linear(features))
        inputs = features.unsqueeze(1)
        bsz = inputs.shape[0]

        self.clear_buffers()
        x = inputs

        for i in range(self.max_seg_length):
            assert len(x.shape) == 3
            assert x.shape[1] == 1
            
            x = F.relu(self.conv.incremental_forward(x))
            x = F.relu(self.final_fc1(x))
            x = self.final_fc2(x)

            x = x.squeeze(1) # (bsz, vocab_size)
            _, predicted = x.max(1) 

            outputs.append(predicted)
            x = self.embed(predicted)
            x = x.unsqueeze(0)

        outputs = torch.stack(outputs, 1)
        
        return outputs



class CaptionSingleRNN(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=60):
        """Set the hyper-parameters and build the layers."""
        super(CaptionSingleRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.vocab_size = vocab_size
        self.seqmodel = nn.LSTM(embed_size, hidden_size, 2, batch_first=True)

        self.max_seg_length = max_seq_length

        self.image_linear = nn.Linear(feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.final_fc1 = SequenceWise(nn.Linear(hidden_size, vocab_size))
        

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        features = self.bn(self.image_linear(features))
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        c = None
        x = embeddings
        x, c = self.seqmodel(x,c)
        x = self.final_fc1(x)
        return x[:,:-1,:]

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""

        outputs = []
        features = self.bn(self.image_linear(features))
        inputs = features.unsqueeze(1)
        c = None
        x = inputs
        
        for i in range(self.max_seg_length):

            assert x.shape[1] == 1
            assert len(x.shape) == 3
            #print(x.shape)

            x, c = self.seqmodel(x, c)
            x = self.final_fc1(x)
            
            x = x.squeeze(1) # (bsz, vocab_size)
            _, predicted = x.max(1) 

            outputs.append(predicted)
            x = self.embed(predicted)
            x = x.unsqueeze(0)

        outputs = torch.stack(outputs,1)
        #outputs = outputs.contiguous().view(features.shape[0]*self.max_seg_length,-1)
        return outputs    

    def sample_old(self, features, states=None, return_logits=0):
        """Generate captions for given image features using greedy search."""

        outputs = []
        logits = []
        features = self.bn(self.image_linear(features))
        inputs = features.unsqueeze(1)
        c = None
        x = inputs
        
        for i in range(self.max_seg_length):

            assert x.shape[1] == 1
            assert len(x.shape) == 3

            x, c = self.seqmodel(x, c)
            x = self.final_fc1(x)

            probs = F.softmax(x.view(1,-1),1).detach().cpu().numpy()

            predicted = np.random.choice(np.arange(self.vocab_size), p=probs[0])
            predicted = torch.LongTensor(np.array([predicted])).cuda()

            logits.append(x.squeeze(0))
            predicted = predicted.unsqueeze(1)
            outputs.append(predicted.squeeze(1))
            x = self.embed(predicted)

        outputs = torch.stack(outputs,1)
        logits = torch.stack(logits)

        if return_logits:
            #print("Returning logits")
            return logits
        return outputs


