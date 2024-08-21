from utils import LoadDataset
import argparse
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from module.ddpm import DDPM, CondDDPM
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import torchvision
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
class Trainer():

    def __init__(self, args):

        train_dataset = LoadDataset(data_path= args.data_path, 
                                        json_path= args.train_json_path,
                                        objects_path= args.objects_file_path)
        self.train_loader = DataLoader(train_dataset,
                                    batch_size= args.batch_size,
                                    num_workers= args.num_workers,
                                    pin_memory= True,
                                    shuffle= True
                                )
        
        val_dataset = LoadDataset(data_path= args.data_path, 
                                json_path= args.test_json_path,
                                objects_path= args.objects_file_path,
                                mode= "new_test")
        self.val_loader = DataLoader(val_dataset,
                                    batch_size= 32,
                                    num_workers= args.num_workers,
                                    pin_memory= True,
                                    shuffle= False
                                )

        # create a scheduler, adding small amount of noise for every time steps
        self.noise_scheduler = DDPMScheduler(num_train_timesteps= 1000)
        self.model = DDPM().to(args.device)
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr= args.learning_rate)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.save_img_per_epoch = args.save_img_per_epoch
        
        self.prev_epochs = 0
        self.history = {"train_loss": list()}
        self.best_loss = np.inf
        if args.pre_train is not None:
            checkpoint = torch.load(args.pre_train)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.noise_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.prev_epochs = checkpoint['epoch']
            self.history = checkpoint['history']
            self.best_loss = checkpoint['best_loss']

    def train(self):
        
        for epoch in range(self.prev_epochs, self.epochs):
            
            self.model.train()
            train_batch_loss = []
            pbar = tqdm(enumerate(self.train_loader))
            
            for i, (img, label) in pbar:
                img = img.to(self.device)
                label = label.to(self.device)
                noise = torch.randn_like(img)
                time_steps = torch.randint(0, 999, (img.shape[0], )).long().to(self.device)
                noisy_x = self.noise_scheduler.add_noise(img, noise, time_steps)
                pred = self.model(noisy_x, time_steps, label)
                # how close is the output to the noise
                loss = self.loss_fn(pred, noise)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                train_batch_loss.append(loss.item())
                pbar.set_description_str(f"[Train] epoch: {epoch + 1} / {self.epochs}, iter: {i + 1} / {len(self.train_loader)}, loss: {np.mean(train_batch_loss)}")
    
            # save model
            train_loss = np.mean(train_batch_loss)
            if self.best_loss > train_loss:
                self.best_loss = train_loss
                torch.save({
                        'model_state_dict': self.model.state_dict(), 
                        'epoch': epoch, 
                        'history': self.history,
                        'best_loss': self.best_loss,
                        # 'scheduler_state_dict': self.noise_scheduler.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict()}, "./result/best.ckpt")

            self.history["train_loss"].append(train_loss)

            if epoch % self.save_img_per_epoch == 0:

                self.model.eval()

                with torch.no_grad():
                    pbar = tqdm(enumerate(self.val_loader))
                    for i, label in pbar:
                        label = label.to(self.device)
                        # sample from normal distribution
                        img = torch.randn(label.shape[0], 3, 64, 64).to(self.device)
                        for t in self.noise_scheduler.timesteps:
                            residual = self.model(img, t, label)
                            img = self.noise_scheduler.step(residual, t, img).prev_sample
                            pbar.set_description_str(f"[Val] epoch: {epoch + 1} / {self.epochs}, T: {t}")
                        
                        self.save_img(img, epoch)


    def save_img(self, img, epoch):
        de_normalize = transforms.Normalize(mean= [-1.0, -1.0, -1.0], std= [2.0, 2.0, 2.0])
        de_img = de_normalize(img)
        
        plt.imshow(torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(de_img, nrow=8)))
        plt.savefig(f'./result/{epoch}.png')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type= str, default= "cuda:0", help= "training device")
    parser.add_argument("--data_path", type= str, default= "./iclevr", help= "training dataset path")
    parser.add_argument("--train_json_path", type= str, default= "./file/train.json", help= "training label json")
    parser.add_argument("--test_json_path", type= str, default= "./file/new_test.json", help= "testing label path")
    parser.add_argument("--objects_file_path", type= str, default= "./file/objects.json", help= "objects json path which has all classification")
    parser.add_argument("--batch_size", type= int, default= 64 , help= "train batch size")
    parser.add_argument("--num_workers", type= int, default= 4, help= "number of worker")
    parser.add_argument("--epochs", type= int, default= 300, help= "training epochs")
    parser.add_argument("--learning-rate", type= float, default= 1e-4, help= "number of training learning rate") 
    parser.add_argument("--save_path", type= str, default= "./best.ckpt", help= "path which save best model")
    parser.add_argument("--save_img_per_epoch", type= int, default= 5, help= "which epoch needed save image")
    parser.add_argument("--save_model_per_epoch", type= int, default= 1)
    parser.add_argument("--pre-train", type= str, default= None, help= "pre train model")

    args = parser.parse_args()

    if not os.path.isdir("./result"):
        os.mkdir("./result")

    trainer = Trainer(args)
    trainer.train()