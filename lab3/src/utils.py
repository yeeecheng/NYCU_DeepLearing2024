import matplotlib.pyplot as plt
import torch


def dice_score(pred_mask, gt_mask):
    # 2 * (common pixel of pred_mask and gt_mask) / (pred_mask's 1 + gt_mask's 1)
    # pixel is 1 means not exist
    
    # Avoid a denominator of 0
    smooth = 1e-6
    return ((2 * (pred_mask * gt_mask).sum() + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)).float().mean()


# draw accuracy and loss during the training and testing 
def draw_history(history):
    
    plt.figure()
    plt.plot(torch.tensor(history["train_acc"]), label = 'train acc')
    plt.plot(torch.tensor(history["val_acc"]), label = 'val acc')
    plt.plot(torch.tensor(history["train_loss"]), label = 'train loss')
    plt.plot(torch.tensor(history["val_loss"]), label = 'val loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig('acc_loss_history.png')
    plt.show()


def show_img(img):
    plt.imshow(img)
    plt.show()