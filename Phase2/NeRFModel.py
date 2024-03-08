import torch
import torch.nn as nn
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################
        self.layer1 = nn.Sequential(
            nn.Linear(3 + 3*2*6, 256),
            nn.Relu(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256,256),
            nn.Relu(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256+39, 256),
            nn.Relu(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(256,128),
            nn.Relu(),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(128,4),
        )
    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################

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
