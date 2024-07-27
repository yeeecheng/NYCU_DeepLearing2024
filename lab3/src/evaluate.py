import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *

def evaluate(net, val_dataloader, device):
    

    criterion = nn.BCELoss()
    net.eval()

    batch_val_loss = []
    batch_val_acc = []
    with torch.no_grad():

        for batch in tqdm(val_dataloader):

            imgs, masks = batch["image"], batch["mask"]
            masks_pred = net(imgs.to(device)).squeeze(1)
            loss = criterion(masks_pred, masks.to(device).float())
            acc = dice_score(masks_pred, masks.to(device).float())
            
            batch_val_acc.append(acc)
            batch_val_loss.append(loss.item())
        val_acc = sum(batch_val_acc) / len(batch_val_acc)
        val_loss = sum(batch_val_loss) / len(batch_val_loss)
        return val_acc, val_loss

 