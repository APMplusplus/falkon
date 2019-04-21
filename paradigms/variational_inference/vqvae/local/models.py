import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import *
from layers import VectorQuantizer
from decoders import *

sys.path.append('/home/srallaba/development/repos/falkon/')
import src.nn.layers as falcon_layers

print_flag = 0

class baseline_vqvae(nn.Module):

    def __init__(self):
        super().__init__()

        ### Encoder
        self.dimensions_encoder = 256
        self.encoder = DownsamplingEncoder(self.dimensions_encoder)

        ### Vector Quantization
        self.dimensions_vq = self.dimensions_encoder
        self.num_classes_vq = 64
        self.quantizer = VectorQuantizer(self.num_classes_vq, self.dimensions_vq)

        ### Decoder
        self.decoder = wavenet_baseline()

    def forward(self, x, mel):

        ### Encoder
        if print_flag:
           print("  Model: Shape of input to the model: ", x.shape)
        x = x.unsqueeze(-1)
        encoded = self.encoder(x)
        if print_flag:
           print("  Model: Shape of output from the encoder: ", encoded.shape)


        ### Quantization
        latents = self.quantizer(encoded)
        if print_flag:
           print("  Model: Shape of output from the quantizer: ", latents.shape)

        ### Decoder
        y_hat = self.decoder( x, latents)
 
        return y_hat

