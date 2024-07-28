import matplotlib.pyplot as plt
import torch


def cal_dice_score(pred_mask, gt_mask):
    # 2 * (common pixel of pred_mask and gt_mask) / (pred_mask's 1 + gt_mask's 1)
    # pixel is 1 means not exist
    
    # Avoid a denominator of 0
    smooth = 1e-6
    pred_mask[pred_mask > 0.5] = torch.tensor(1.0)
    pred_mask[pred_mask <= 0.5] = torch.tensor(0.0)
    intersection = (pred_mask * gt_mask).sum()
    return (2 * intersection + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)


def dice_loss(pred_mask, gt_mask):

    smooth = 1e-6
    intersection = (pred_mask * gt_mask).sum() + smooth
    return 1 - (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + smooth)


# draw accuracy and loss during the training and testing 
def draw_history(history, show= False):
    
    plt.figure()
    plt.plot(torch.tensor(history["train_dice_score"]), label = 'train dice score')
    plt.plot(torch.tensor(history["val_dice_score"]), label = 'val dice score')
    plt.plot(torch.tensor(history["train_loss"]), label = 'train loss')
    plt.plot(torch.tensor(history["val_loss"]), label = 'val loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig('dicescore_loss_history.png')
    if show:
        plt.show()


def show_img(org_img, img):

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(org_img)
    plt.title('org mask')

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title('pred mask')

    plt.show()