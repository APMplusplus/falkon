import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *

sys.path.append('/home/srallaba/development/repos/falkon/')
import src.nn.layers as falcon_layers


class baseline_vqvae(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = DownsamplingEncoder(256)

    def forward(self, x):
        print("  Model: Shape of input to the model: ", x.shape)
        x = x.unsqueeze(-1)
        encoded = self.encoder(x)
        print("  Model: Shape of output from the encoder: ", encoded.shape)
        return self.encoder(x)


