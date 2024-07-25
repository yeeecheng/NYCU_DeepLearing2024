import argparse
from oxford_pet import *
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import *
from utils import *
from torchvision import transforms
from evaluate import *

def train(args):
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_transforms = transforms.Compose([
        transforms.ElasticTransform(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees= 15),
    ])

    train_dataset = SimpleOxfordPetDataset(args.data_path, mode= "train", transform= train_transforms)
    val_dataset = SimpleOxfordPetDataset(args.data_path, mode= "valid")

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle= True)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle= False)    

    model = UNet(3, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= LR, momentum= 0.99)

    best_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}


    for epoch in range(EPOCHS):
        
        model.train()
        batch_train_loss = []
        batch_train_acc = []

        for batch in tqdm(train_dataloader):

            imgs, masks = batch["image"], batch["mask"]
            optimizer.zero_grad()
            # batch, channel (num_classes), W, H
            masks_pred = model(imgs.to(device))
            loss = criterion(masks_pred, masks.to(device))

            loss.backward()
            optimizer.step()
            # torch.argmax() -> find the optimal in all masks_pred classes
            acc = dice_score(torch.argmax(masks_pred, dim=1), masks.to(device))

            batch_train_acc.append(acc)
            batch_train_loss.append(loss.item())


        train_acc = sum(batch_train_acc) / len(batch_train_acc)
        train_loss = sum(batch_train_loss) / len(batch_train_loss)
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        

        val_acc, val_loss = evaluate(model, val_dataloader, device)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(f"[ Train | {epoch + 1:03d}/{EPOCHS:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f} [ Validation | {epoch + 1:03d}/{EPOCHS:03d} ] val_loss = {val_loss:.5f}, val_acc = {val_acc:.5f}")

        if val_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), "./best.ckpt")
            best_acc = val_acc

    # draw model history 
    draw_history(history)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)