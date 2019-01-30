import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
    
        self.fc1 = nn.Sequential( \
            nn.Linear(128, 64), nn.ReLU())
        self.fc2 = nn.Sequential( \
            nn.Linear(64, 32), nn.ReLU())
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

    def forward_eval(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x