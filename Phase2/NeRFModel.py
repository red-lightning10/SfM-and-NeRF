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
            nn.Linear(3 + 3*2*6, 128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128+39, 128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(64,4),
        )
    def pos_enc(self, points, num_higer_freqs=6):
        #############################
        # Implement position encoding here
        #############################
        encoding = [points]
        frequency_bands = 2.0 * torch.linspace(0.0,5.0,6,dtype=points.dtype,device=points.device, )
        print(f"frequency{frequency_bands}")
        for fequency in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(points * fequency))
        encoding_points = torch.cat(encoding, dim=-1)
        return encoding_points

    def forward(self, gamma):
        #############################
        # network structure
        #############################
        x = self.layer1(gamma)  #39-256
        x = self.layer2(x)      #256-256
        x = self.layer2(x)      #256-256
        x = self.layer2(x)      #256-256
        x = self.layer3(torch.concat([x , gamma], axis=-1))      #256+39-256
        x = self.layer2(x)      #256-256
        x = self.layer2(x)      #256-256
        x = self.layer4(x)      #256-128
        x = self.layer5(x)
        return x
class model_tinyNeRF(nn.Module):
    def __init__(self):
        super(model_tinyNeRF,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(3 + 3*2*6, 128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128,4),
        )
    def pos_enc(self, tensor, num_higher_freqs=6):
        #############################
        # Implement position encoding here
        #############################
        ps = [tensor]
        for i in range(num_higher_freqs):
            ps.append(torch.sin((2.0**i) * tensor))
            ps.append(torch.cos((2.0**i) * tensor))
        ps = torch.concat(ps, axis =-1)
        return ps

    def forward(self, gamma):
        x = self.layer1(gamma)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
