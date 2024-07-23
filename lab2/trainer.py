# implement your training script here
import argparse
import torch


import torch.nn as nn
from model.SCCNet import SCCNet
from utils import * 
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(args):

    # check current device is cpu or gpu
    device = get_device()
    # processing dataset to dataloader
    train_dataloader = get_dataloader(method= args.method, phase= "train", batch_size= args.batch_size)
    test_dataloader = get_dataloader(method= args.method, phase= "test", batch_size= args.batch_size)

    model = SCCNet(C= args.channel, Nu= args.Nu, Nc= args.Nc, Nt= args.Nt, dropoutRate= args.dropoutRate).to(device)

    # pre-trained model
    if args.pre_trained != "":
        model.load_state_dict(torch.load(args.pre_trained, weights_only= True))
    # check model total parameters
    show_model(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, weight_decay= args.l2)

    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(args.epochs):

        model.train()
        batch_train_loss = []
        batch_train_acc = []

        for batch in tqdm(train_dataloader):

            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            y_pred = model(features)

            loss = criterion(y_pred, labels.long())
            # back propagation
            loss.backward()
            # update weight
            optimizer.step()

            acc = (y_pred.argmax(dim= -1) == labels).float().mean()

            batch_train_acc.append(acc)
            batch_train_loss.append(loss.item())

        train_acc = sum(batch_train_acc) / len(batch_train_acc)
        train_loss = sum(batch_train_loss) / len(batch_train_loss)
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)


        model.eval()

        batch_test_loss = []
        batch_test_acc = []
        
        with torch.no_grad():

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
            history["test_acc"].append(test_acc)
            history["test_loss"].append(test_loss)

        print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f} [ Test | {epoch + 1:03d}/{args.epochs:03d} ] loss = {test_loss:.5f}, test_acc = {test_acc:.5f}")

        if test_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), "./best.ckpt")
            best_acc = test_acc

    # draw model history 
    draw_history(history)


def parse_args():
    
    parse = argparse.ArgumentParser()
    parse.add_argument("-m", "--method", required= True, type= str)
    parse.add_argument("--epochs", default= 10000, type= int)
    parse.add_argument("--batch_size", default= 16, type= int)
    parse.add_argument("--l2", default= 0.0001, type= float)
    parse.add_argument("--lr", default= 0.001, type= float)
    parse.add_argument("-p", "--pre_trained", default= "", type= str)
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
    train(args)

