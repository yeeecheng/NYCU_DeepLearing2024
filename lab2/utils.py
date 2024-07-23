# script for drawing figures, and more if needed
import torch
import matplotlib.pyplot as plt
from Dataloader import MIBCI2aDataset
from torch.utils.data import DataLoader
import numpy as np

# get device, cpu or gpu
def get_device():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    return device

# show total parameters of model
def show_model(model):

    for name, parameters in model.named_parameters():
        print(name, ":", parameters.size())

    print(f"\ntotal parameters {sum(p.numel() for p in model.parameters())}")

# dataloader 
def get_dataloader(method , phase, batch_size):
    
    if method == "SD":
        mode = "SD_train" if phase == "train" else "SD_test"
    elif method == "LOSO":
        mode = "LOSO_train" if phase == "train" else "LOSO_test"
    elif method == "FT":
        mode = "FT" if phase == "train" else "LOSO_test"
        
    dataset = MIBCI2aDataset(mode= mode)

    print(f"In the {phase} set, feature shape: {dataset.features.shape}, label shape: {dataset.labels.shape}")
    
    shuffle = True if phase == "train" else False
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle= shuffle)

    return dataloader


# draw accuracy and lsos during the training and testing 
def draw_history(history):
    
    plt.figure()
    plt.plot(torch.tensor(history["train_acc"]), label = 'train acc')
    plt.plot(torch.tensor(history["test_acc"]), label = 'test acc')
    plt.plot(torch.tensor(history["train_loss"]), label = 'train loss')
    plt.plot(torch.tensor(history["test_loss"]), label = 'test loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig('acc_loss_history.png')
    plt.show()

# draw the signal.
def plot_eeg(eeg_data):
    channels, time = eeg_data.shape
    plt.figure(figsize=(10, 7))

    for i in range(channels):
        plt.plot(np.arange(time), eeg_data[i, :], label=f'Channel {i+1}')

    plt.title('EEG Signals')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()
