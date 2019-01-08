import nltk
import pickle
import argparse
from collections import Counter
import sys
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os, sys
import pickle
import numpy as np
import nltk
import argparse
import torch
import numpy as np
import torch.nn as nn
from nltk.translate.bleu_score import *


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, i2f_dict, i2c_dict, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            i2f_dict: image id to feature map
            i2c_dict: image id to caption map
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.ids = i2f_dict.keys()
        self.image_name = []
        self.features = []
        self.captions = []

        for i, id in enumerate(self.ids):
            self.image_name.append(id)
            self.features.append(i2f_dict[id])
            self.captions.append(i2c_dict[id])
        
            if (i % 1000) == 0:
                print("[{}/{}] curated.".format(i+1, len(self.ids)))

        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        caption = self.captions[index]
        img_feature = self.features[index]
        img_name = self.image_name[index]

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        image = torch.Tensor(img_feature)
        return image, target, img_name

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, image_name).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption, image_name). 
            - image: torch tensor of shape (features_dim).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, features_dim).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, image_names = zip(*data)

    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    
    return images, targets, lengths, image_names


def get_loader(i2f_dict, i2c_dict, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(i2f_dict=i2f_dict,
                       i2c_dict=i2c_dict,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths, image_names) for each iteration.
    # images: a tensor of shape (batch_size, feature_lenght).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(pkl_file, threshold):
    """Build a simple vocabulary wrapper."""
    with open(pkl_file, 'rb') as f:
        imageid2captions = pickle.load(f)

    counter = Counter()
    ids = imageid2captions.keys()
    for i, id in enumerate(ids):
        caption = str(imageid2captions[id])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def sample_gumbel(shape, eps=1e-10, out=None):
   U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
   return - torch.log(eps - torch.log(U + eps))

def gumbel_argmax(logits, dim):
   # Draw from a multinomial distribution efficiently
   #print("Shape of gumbel input: ", logits.shape)
   return logits + sample_gumbel(logits.size(), out=logits.data.new())
   #sys.exit()
   return torch.max(logits + sample_gumbel(logits.size(), out=logits.data.new()), dim)[1]


def return_sentence_bleu(original_caption, sampled_caption):
    chencherry = SmoothingFunction()
    bleu_score_sentence = sentence_bleu([original_caption], sampled_caption, smoothing_function=chencherry.method1)
    
    return bleu_score_sentence
