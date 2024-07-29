import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *

def evaluate(net, val_dataloader, device):
    
    criterion = nn.BCELoss()
    batch_val_loss = []
    batch_val_dice_score = []

    with torch.no_grad():

        net.eval()
        for batch in tqdm(val_dataloader):

            imgs, masks = batch["image"].to(device), batch["mask"].to(device)
            masks_pred = net(imgs.to(device)).squeeze(1)
            val_loss = criterion(masks_pred, masks.float()) + dice_loss(masks_pred, masks.float())
            dice_score = cal_dice_score(masks_pred, masks.float())
            
            batch_val_dice_score.append(dice_score)
            batch_val_loss.append(val_loss.item())
        val_dice_score = sum(batch_val_dice_score) / len(batch_val_dice_score)
        val_loss = sum(batch_val_loss) / len(batch_val_loss)
        return val_dice_score, val_loss

 