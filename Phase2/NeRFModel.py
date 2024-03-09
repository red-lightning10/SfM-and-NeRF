import torch
import torch.nn as nn
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(self):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        self.layer1 = nn.Sequential(
            nn.Linear(3 + 3*2*6, 256),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256+39, 256),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(128,4),
        )
    def pos_enc(self, x, L):
        #############################
        # Implement position encoding here
        #############################
        y = []
        y.append(x)
        for i in range(L):
            y.append(torch.sin((2.0**i) * x))
            y.append(torch.cos((2.0**i) * x))
        y = torch.concat(y, axis =-1)
        return y

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################
        x = self.l1(gamma)  #39-256
        x = self.l2(x)      #256-256
        x = self.l2(x)      #256-256
        x = self.l2(x)      #256-256
        x = self.l3(torch.concat([x , gamma], axis=-1))      #256+39-256
        x = self.l2(x)      #256-256
        x = self.l2(x)      #256-256
        x = self.l4(x)      #256-128
        x = self.l5(x)
        return output
