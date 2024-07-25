import torch 
import torch.nn as nn


class Resnet34_UNet(nn.Module):
    def __init__(self):
        super(Resnet34_UNet, self).__init__()
        pass

    def forward(self, x):
        return x