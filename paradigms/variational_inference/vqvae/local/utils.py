import numpy as np
import multiprocessing as mp
from collections import defaultdict
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# https://stackoverflow.com/questions/17864466/flatten-a-list-of-strings-and-lists-of-strings-and-lists-in-python
def flattern(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flattern(i))
        else: rt.append(i)
    return rt


class vcc2018_dataset(Dataset):

    def __init__(self, files_array, data_dir, files2speakers_dict):

       self.files_array = files_array
       self.data_dir = data_dir
       self.files2speakers_dict = files2speakers_dict
       self.max_timesteps = 2000
       self.frame_period = 256

    def __getitem__(self, index):
          audio = np.load(self.data_dir + '/' + self.files_array[index])
          mel = np.load(self.data_dir + '/' + self.files_array[index].replace('audio', 'mel'))
          speaker = self.files2speakers_dict[self.files_array[index]]
          start_sample = np.random.randint(len(audio) - 500 - self.max_timesteps)
          end_sample = start_sample + self.max_timesteps
          start_frame = int(start_sample / self.frame_period )
          end_frame = int(end_sample / self.frame_period )
          end_frame = start_frame + 8

          x = audio[start_sample:end_sample]
          c = mel[start_frame:end_frame]
          return x, c, int(speaker)

    def __len__(self):
           return len(self.files_array)


def collate_fn(batch):
    audio_batch = [x[0] for x in batch]
    mel_batch = [x[1] for x in batch]
    speakers = [x[2] for x in batch]

    audio_batch = torch.FloatTensor(audio_batch)
    mel_batch = torch.FloatTensor(mel_batch)   
    spk_batch = torch.LongTensor(np.array(speakers))

    return audio_batch, mel_batch, spk_batch

def get_dataloaders(data_dir):

   speakers_files, files2speakers_dict = get_speakers_files(data_dir)
   test_speakers = ['10', '11']
   train_speakers = ['1', '2', '3','4', '5','6','7','8','9']
   all_speakers = list(speakers_files.keys())

   train_files = [speakers_files[spk][:-5] for spk in all_speakers]
   test_files = [speakers_files[spk][-5:] for spk in all_speakers]

   train_files = flattern(train_files)
   test_files = flattern(test_files)

   train_set = vcc2018_dataset(train_files, data_dir, files2speakers_dict)
   train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn = collate_fn
                          )

   val_set = vcc2018_dataset(test_files, data_dir, files2speakers_dict)
   val_loader = DataLoader(val_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=4,
                          collate_fn = collate_fn
                          )

   return train_loader, val_loader


def get_speakers_files(data_dir):

    speakers_files_dict = {}
    files2speakers_dict = {}
    train_file = data_dir + '/train.txt'
    f = open(train_file)
    for line in f:
        contents = line.split('\n')[0].split('|')
        speaker_id = contents[-1]
        files2speakers_dict[contents[0]] = speaker_id
        if speaker_id in speakers_files_dict.keys():
           speakers_files_dict[speaker_id].append(contents[0])
        else:
           speakers_files_dict[speaker_id] = []

    return speakers_files_dict, files2speakers_dict

