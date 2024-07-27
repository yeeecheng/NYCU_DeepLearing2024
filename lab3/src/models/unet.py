import torch 
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= 3, padding= 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
            nn.Conv2d(out_channels, out_channels, kernel_size= 3, padding= 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace= True),
        )

    def forward(self, x):
        return self.conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()

        self.down_sampling = nn.Sequential(
            nn.MaxPool2d(kernel_size= 2, stride= 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_sampling(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()

        self.de_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 2, stride= 2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, cropped_x):
        x = self.de_conv(x)
        x = torch.cat([cropped_x, x], dim= 1)
        return self.conv(x)

class UNet(nn.Module):

    def __init__(self, num_channels, num_classes):
        super(UNet, self).__init__()
        

        self.contracting1 = DoubleConv(num_channels, 64)
        self.contracting2 = DownSampling(64, 128)
        self.contracting3 = DownSampling(128, 256)
        self.contracting4 = DownSampling(256, 512)
        # connect between contracting path and expansive path
        self.contracting5 = DownSampling(512, 1024)
        self.expansive1 = UpSampling(1024, 512)
        self.expansive2 = UpSampling(512, 256)
        self.expansive3 = UpSampling(256, 128)
        self.expansive4 = UpSampling(128, 64)
        self.output = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size= 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        c1 = self.contracting1(x)
        c2 = self.contracting2(c1)
        c3 = self.contracting3(c2)
        c4 = self.contracting4(c3)
        c5 = self.contracting5(c4)
        x = self.expansive1(c5, c4)
        x = self.expansive2(x, c3)
        x = self.expansive3(x, c2)
        x = self.expansive4(x, c1)
        return self.output(x)
