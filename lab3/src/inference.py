import argparse
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import *
from models.resnet34_unet import *
from tqdm import tqdm
from oxford_pet import * 
from utils import *

def inference(args):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    NET = args.net
    DATA_PATH = args.data_path
    BATCH_SIZE = args.batch_size
    PRE_TRAIN_MODEL = args.model    

    test_dataset = SimpleOxfordPetDataset(DATA_PATH, mode= "test")
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle= False)

    if NET == "UNet":
        model = UNet(3, 2).to(device)
        model.load_state_dict(torch.load(PRE_TRAIN_MODEL))

        criterion = nn.CrossEntropyLoss()

    elif NET == "ResNet34_UNet":
        model = ResNet34_UNet(3, 2).to(device)
        model.load_state_dict(torch.load(PRE_TRAIN_MODEL))
        criterion = nn.CrossEntropyLoss()
    
    model.eval()

    batch_test_loss = []
    batch_test_acc = []
    with torch.no_grad():

        for batch in tqdm(test_dataloader):

            imgs, masks = batch["image"], batch["mask"]
            masks_pred = model(imgs.to(device))
            loss = criterion(masks_pred, masks.to(device))
            acc = dice_score(torch.argmax(masks_pred, dim= 1), masks.to(device))
            
            batch_test_acc.append(acc)
            batch_test_loss.append(loss.item())
        test_acc = sum(batch_test_acc) / len(batch_test_acc)
        test_loss = sum(batch_test_loss) / len(batch_test_loss)
        
        print(f"[ Test ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")




def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--net', '-n', type=str, default="UNet")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    inference(args)