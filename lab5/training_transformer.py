import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers(args)
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        batch_loss = []
        pbar = tqdm(enumerate(train_loader))

        for i, batch in pbar:
            img = batch.to(self.args.device)
            # output: gt, pred
            logits, z_indices = self.model(img)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
            loss.backward()

            batch_loss.append(loss.item())           
            if i % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            pbar.set_description_str(f"epoch: {epoch} / {self.args.epochs}, iter: {i} / {len(train_loader)}, loss: {np.mean(batch_loss)}")

        return np.mean(batch_loss)
    
    def eval_one_epoch(self, eval_loader, epoch):
        
        batch_loss = []
        pbar = tqdm(enumerate(eval_loader))
        self.model.eval()
        with torch.no_grad():

            for i, batch in pbar:
                img = batch.to(self.args.device)
                # output: gt, pred
                logits, z_indices = self.model(img)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
                batch_loss.append(loss.item())          

                pbar.set_description_str(f"epoch: {epoch} / {self.args.epochs}, iter: {i} / {len(eval_loader)}, loss: {np.mean(batch_loss)}")

        return np.mean(batch_loss)
    

    def configure_optimizers(self, args): 
        optimizer = torch.optim.Adam(self.model.parameters(), lr= args.learning_rate)
        scheduler = None
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default= None, help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)
    
    
    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    

    prev_epochs = 0
    best_loss = np.inf
    history = {"train_loss": list(), "eval_loss": list()}
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path)
        train_transformer.model.load_state_dict(checkpoint['model_state_dict'])
        train_transformer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epochs = checkpoint['epoch']
        history = checkpoint['history']
        best_loss = checkpoint['best_loss']
    
    
    for epoch in range(prev_epochs + args.start_from_epoch+1, args.epochs+1):
        
        train_loss = train_transformer.train_one_epoch(train_loader, epoch)
        eval_loss = train_transformer.eval_one_epoch(val_loader, epoch)

        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_loss)

        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save({'model_state_dict': train_transformer.model.state_dict(), 
                        'epoch': epoch, 
                        'history': history,
                        'best_loss': best_loss,
                        'optimizer_state_dict': train_transformer.optimizer.state_dict()}, "./best.pth")