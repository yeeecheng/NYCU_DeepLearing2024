# implement SCCNet model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# reference paper: https://ieeexplore.ieee.org/document/8716937

# square activation
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.square(x)


class SCCNet(nn.Module):

    def __init__(self, numClasses= 4, timeSample= 438, C= 22, Nu= 22, Nc= 20, Nt= 1, dropoutRate= 0.5):
        super(SCCNet, self).__init__()
        
        # spatial component analysis
        self.first_conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= Nu, kernel_size= (C, Nt)),
            nn.BatchNorm2d(num_features= Nu),
            nn.Dropout(p= dropoutRate),
        )

        self.second_conv2d_block = nn.Sequential(
            nn.Conv2d(in_channels= 22 - C + 1, out_channels= Nc, kernel_size= (Nu, 12), padding= (0, 6)),
            nn.BatchNorm2d(num_features= Nc),
            SquareLayer(),
            nn.Dropout(p= dropoutRate),
        )

        self.pooling_block = nn.Sequential(
            nn.AvgPool2d(kernel_size= (1, 62), stride= (1,12)),
        )

        self.softmax_block = nn.Sequential(
            nn.Linear(Nc * math.ceil((timeSample- 62 + 1) / 12), numClasses)
        )

    def forward(self, x):

        # [B, 1, 22, 438]
        x = self.first_conv2d_block(x)
        # [B, Nu, 22 - C + 1, 438]
        x = x.permute(0, 2, 1, 3)
        # [B, 22 - C + 1, Nu, 438]
        x = self.second_conv2d_block(x)
        # [B, Nc, 1, 438]
        x = self.pooling_block(x)
        # [B, Nc, 32]
        x = x.view(x.size(0), -1)
        # [B, Nc * 32]
        x = self.softmax_block(x) 
        return x
    
    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass