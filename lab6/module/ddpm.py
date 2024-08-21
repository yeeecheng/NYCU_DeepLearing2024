import torch
from torch import nn
from torch.nn import functional as F 
from diffusers import UNet2DConditionModel, UNet2DModel


class DDPM(nn.Module):
    
    def __init__(self, num_classes= 3, class_emb_size= 24):
        super().__init__()
        self.UNet = UNet2DModel(
            sample_size= 64,
            in_channels= 3 + class_emb_size,
            out_channels= 3,
            layers_per_block= 2,
            block_out_channels= (64, 128, 128, 256),
            down_block_types= (
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types= (
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        """
        x: noisy_x,
        t: time steps,
        class_labels: y
        """
        b, c, w, h = x.shape
        class_cond = class_labels.view(b, class_labels.shape[1], 1, 1).expand(b, class_labels.shape[1], w, h)
        # new input is x and class cond concatenated together along dim 1
        net_input = torch.cat((x, class_cond), 1)
        # Feed new input to the UNet alongside the time step and return the prediction
        return self.UNet(net_input, t).sample


class CondDDPM(nn.Module):

    def __init__(self, num_classes=3, class_emb_size=24):
        super().__init__()

        self.UNet = UNet2DConditionModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            encoder_hid_dim = 64
        )

        self.encoder = nn.Linear(class_emb_size, 64 * 64)

    def forward(self, x, t, class_labels):
        """
        x: noisy_x,
        t: time steps,
        class_labels: y
        """
        b, c, w, h = x.shape
        
        class_cond = self.encoder(class_labels).view(b, 64, 64)
        # class_cond = class_labels.view(b, class_labels.shape[1], 1, 1).expand(b, class_labels.shape[1], w, h)
        # print(class_cond.shape)
        # Feed input to the UNet alongside the time step and class labels, and return the prediction
        return self.UNet(x, t, class_cond).sample