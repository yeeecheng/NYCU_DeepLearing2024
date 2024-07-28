import argparse
from oxford_pet import *
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import *
from models.resnet34_unet import *
from utils import *

from evaluate import *
import torch.nn.functional as F

def train(args):
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    NET = args.net
    PRE_TRAIN_MODEL = args.model 
    DATA_PATH = args.data_path    

    # Data preprocessing 
    train_dataset = load_dataset(DATA_PATH, "train")
    val_dataset = load_dataset(DATA_PATH, "valid")
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle= True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle= False)    


    model = UNet(3, 1).to(device) if NET == "UNet" else ResNet34_UNet(3, 1).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= LR)

    prev_epochs = 0
    if PRE_TRAIN_MODEL is not None:
        checkpoint = torch.load(PRE_TRAIN_MODEL)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epochs = checkpoint['epoch']
    
    EPOCHS += prev_epochs
    
    best_dice_score = 0
    history = {"train_loss": [], "train_dice_score": [], "val_loss": [], "val_dice_score": []}

    
    for epoch in range(prev_epochs + 1, EPOCHS + 1):
        
        model.train()
        batch_train_loss = []
        batch_train_dice_score = []

        for batch in tqdm(train_dataloader):

            imgs, masks = batch["image"].to(device), batch["mask"].to(device)
            # batch, 1, W, H -> batch, W, H

            masks_pred = model(imgs).squeeze(1)
            loss = criterion(masks_pred, masks.float()) + dice_loss(masks_pred, masks.float())
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            dice_score = cal_dice_score(masks_pred, masks.float())


            batch_train_dice_score.append(dice_score)
            batch_train_loss.append(loss.item())


        train_dice_score = sum(batch_train_dice_score) / len(batch_train_dice_score)
        train_loss = sum(batch_train_loss) / len(batch_train_loss)
        history["train_dice_score"].append(train_dice_score)
        history["train_loss"].append(train_loss)
        

        val_dice_score, val_loss = evaluate(model, val_dataloader, device)
        history["val_dice_score"].append(val_dice_score)
        history["val_loss"].append(val_loss)

        print(f"[ Train | {epoch:03d}/{EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_dice_score:.5f} [ Validation | {epoch:03d}/{EPOCHS:03d} ] val_loss = {val_loss:.5f}, val_acc = {val_dice_score:.5f}")

        if val_dice_score > best_dice_score:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save({'model_state_dict': model.state_dict(), 
                        'epoch': epoch, 
                        'optimizer_state_dict': optimizer.state_dict()}, "./best.pth")
            best_dice_score = val_dice_score
        draw_history(history)
    # draw model history 
    draw_history(history, True)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--net', '-n', type=str, default="UNet")
    parser.add_argument('--model', default= None, help='path to the stored model weight')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)