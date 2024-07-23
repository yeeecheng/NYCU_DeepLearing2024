# implement your testing script here
import argparse
import torch
from Dataloader import MIBCI2aDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model.SCCNet import SCCNet
from utils import * 
from tqdm import tqdm


def test(args):
    
    device = get_device()
    # get dataloader 
    test_dataloader = get_dataloader(method= args.method, phase= "test", batch_size= args.batch_size)
    model = SCCNet(C= args.channel, Nu= args.Nu, Nc= args.Nc, Nt= args.Nt, dropoutRate= args.dropoutRate).to(device)
    # load pre-trained model
    if args.pre_trained != "":
        model.load_state_dict(torch.load(args.pre_trained))
    model.eval()

    with torch.no_grad():

        criterion = nn.CrossEntropyLoss()
        batch_test_acc = []
        batch_test_loss = []

        for batch in tqdm(test_dataloader):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            y_pred = model(features)

            loss = criterion(y_pred, labels.long())
            acc = (y_pred.argmax(dim= -1) == labels).float().mean()

            batch_test_acc.append(acc)
            batch_test_loss.append(loss.item())
        test_acc = sum(batch_test_acc) / len(batch_test_acc)
        test_loss = sum(batch_test_loss) / len(batch_test_loss)

        print(f"[ Test ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")

def parse_args():
    
    parse = argparse.ArgumentParser()
    parse.add_argument("-m", "--method", required= True, type= str)
    parse.add_argument("--epochs", default= 10000, type= int)
    parse.add_argument("--batch_size", default= 32, type= int)
    parse.add_argument("--l2", default= 0.0001, type= float)
    parse.add_argument("--lr", default= 0.001, type= float)
    parse.add_argument("-p", "--pre_trained", default= "./best.ckpt", type= str)
    parse.add_argument("-c", "--num_classes", default= 4, type= int)
    parse.add_argument("-C", "--channel", default= 22, type= int)
    parse.add_argument("--Nu", default= 22, type= int)
    parse.add_argument("--Nc", default= 20, type= int)
    parse.add_argument("--Nt", default= 1, type= int)
    parse.add_argument("--dropoutRate", default= 0.5, type= int)
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    test(args)