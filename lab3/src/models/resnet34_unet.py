import torch 
import torch.nn as nn
import torch.nn.functional as F

#https://meetonfriday.com/posts/fb19d450/
#https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet34
#https://github.com/zhoudaxia233/PyTorch-Unet/blob/master/resnet_unet.py
class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride= 1, shortcut= None):
        super(Residual_Block, self).__init__()

        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= stride, padding= 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias= False),
            nn.BatchNorm2d(out_channels),
        )
        self.ac = nn.ReLU()
        self.shortcut = shortcut

    def forward(self, x):
        out = self.residual_conv(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return self.ac(out)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= (3, 3), padding= (1, 1), bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size= (3, 3), padding= (1, 1), bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)
    
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()

        self.de_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 2, stride= 2)

    def forward(self, x):
        return self.de_conv(x)

def concat(x, cropped_x):

    diff_x = cropped_x.shape[2] - x.shape[2]
    diff_y = cropped_x.shape[3] - x.shape[3]
    # padding input shape with dim1 of (diff_x // 2, diff_x - diff_x // 2) and dim2 of (diff_y // 2, diff_y - diff_y // 2) 
    x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
    # concat cropped_x and x
    return torch.cat([cropped_x, x], dim= 1)


class ResNet34_UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ResNet34_UNet, self).__init__()
        
        # encoder
        self.resnet34_conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size= 7, stride= 2, padding= 3, bias= False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 1, stride= 1, bias= False),
            nn.BatchNorm2d(64)
        )
        self.resnet34_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),
            Residual_Block(in_channels= 64, out_channels= 64, stride= 1, shortcut= shortcut2),
            *[Residual_Block(in_channels= 64, out_channels= 64) for _ in range(2)]
        )
        shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 1, stride= 2, bias= False),
            nn.BatchNorm2d(128)
        )
        self.resnet34_conv3 = nn.Sequential(
            Residual_Block(in_channels= 64, out_channels= 128, stride= 2, shortcut= shortcut3),
            *[Residual_Block(in_channels= 128, out_channels= 128) for _ in range(3)]
        )
        shortcut4 = nn.Sequential(
            nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= 1, stride= 2, bias= False),
            nn.BatchNorm2d(256)
        )
        self.resnet34_conv4 = nn.Sequential(
            Residual_Block(in_channels= 128, out_channels= 256, stride= 2, shortcut= shortcut4),
            *[Residual_Block(in_channels= 256, out_channels= 256) for _ in range(5)]
        )
        shortcut5 = nn.Sequential(
            nn.Conv2d(in_channels= 256, out_channels= 512, kernel_size= 1, stride= 2, bias= False),
            nn.BatchNorm2d(512)
        )
        self.resnet34_conv5 = nn.Sequential(
            Residual_Block(in_channels= 256, out_channels= 512, stride= 2, shortcut= shortcut5),
            *[Residual_Block(in_channels=512, out_channels= 512) for _ in range(2)]
        )

        # decoder
        self.unet_up_conv1 = UpSampling(512, 512)
        # cropped + expansive_path 
        self.unet_double_conv1 = DoubleConv(512 + 256, 512)
        self.unet_up_conv2 = UpSampling(512, 256)
        self.unet_double_conv2 = DoubleConv(256 + 128, 256)
        self.unet_up_conv3 = UpSampling(256, 128)
        self.unet_double_conv3 = DoubleConv(128 + 64, 128)
        self.unet_up_conv4 = UpSampling(128, 64)
        self.unet_double_conv4 = DoubleConv(64 + 64, 64)
        self.unet_up_conv5 = UpSampling(64, 32)
        self.output = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size= (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        # encoder
        e1 = self.resnet34_conv1(x)
        e2 = self.resnet34_conv2(e1)
        e3 = self.resnet34_conv3(e2)
        e4 = self.resnet34_conv4(e3)
        e5 = self.resnet34_conv5(e4)
        
        # decoder
        d1 = concat(self.unet_up_conv1(e5), e4)
        x = self.unet_double_conv1(d1)

        d2 = concat(self.unet_up_conv2(x), e3)
        x = self.unet_double_conv2(d2)

        d3 = concat(self.unet_up_conv3(x), e2)
        x = self.unet_double_conv3(d3)

        d4 = concat(self.unet_up_conv4(x), e1)
        x = self.unet_double_conv4(d4)

        d5 = self.unet_up_conv5(x)
        x = self.output(d5)
        
        return x
    
