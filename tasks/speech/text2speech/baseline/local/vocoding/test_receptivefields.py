from utils import *

layers = [ 12]
stacks = [2, 4] 
kernel_sizes = [3]

for layer in layers:
   for stack in stacks:
     for kernel_size in kernel_sizes:
       print("Receptive field with ", layer, " layers ", " and ", stack, " stacks with kernel size ", kernel_size, " is ", receptive_field_size(layer, stack, kernel_size))
   print('\n')

