import torch
from torch import nn
class NN(nn.Module):
     def __init__(self):
         super(NN, self).__init__()
         self.layer1 = nn.Sequential(
             nn.Linear(85, 256),
             nn.BatchNorm1d(256),
             nn.ReLU(),
         )
         self.layer2 = nn.Sequential(
             nn.Linear(256, 128),
             nn.BatchNorm1d(128),
             nn.ReLU(),
         )
         self.decoder = nn.Sequential(
            nn.Linear(128, 83),
         )

     def forward(self, x):
         x = self.layer1(x)
         x = self.layer2(x)
         y = self.decoder(x)
         return y

