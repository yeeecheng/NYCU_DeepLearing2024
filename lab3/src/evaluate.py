import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *

def evaluate(net, val_dataloader, device):
    

    criterion = nn.CrossEntropyLoss()
    net.eval()

    batch_val_loss = []
    batch_val_acc = []
    with torch.no_grad():

        for batch in tqdm(val_dataloader):

            imgs, masks = batch["image"], batch["mask"]
            masks_pred = net(imgs.to(device))
            loss = criterion(masks_pred, masks.to(device))
            acc = dice_score( torch.argmax(masks_pred, dim=1),masks.to(device))
            
            batch_val_acc.append(acc)
            batch_val_loss.append(loss.item())
        val_acc = sum(batch_val_acc) / len(batch_val_acc)
        val_loss = sum(batch_val_loss) / len(batch_val_loss)
        return val_acc, val_loss

 