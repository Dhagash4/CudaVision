import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from plot_utils import plot_stats


def save_model(model, optimizer, epoch, stats, name):
    """ Saving model checkpoint """

    if name == 'best':
        if(not os.path.exists(f"models/{stats['name']}")):
            os.makedirs(f"models/{stats['name']}")
        savepath = f"models/{stats['name']}/best_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': stats
        }, savepath)

    else:
        if(not os.path.exists(f"models/{stats['name']}")):
            os.makedirs(f"models/{stats['name']}")
        savepath = f"models/{stats['name']}/checkpoint_epoch_{epoch}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': stats
        }, savepath)
    return


def load_model(model, optimizer, savepath, plot):
    """ Loading pretrained checkpoint """

    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]

    if plot:

        train_loss = stats['train_loss']
        val_loss = stats['val_loss']
        loss_iters = stats['loss_iters']
        valid_acc = stats['valid_acc']
        plot_stats(train_loss=train_loss, val_loss=val_loss,
                   loss_iters=loss_iters, valid_acc=valid_acc)

    return model, optimizer, epoch, stats


def test_model(model, test_loader, device):

    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        preds = torch.argmax(outputs, dim=1)
        correct += len(torch.where(preds == labels)[0])
        total += len(labels)

    # Total correct predictions and loss
    accuracy = correct / total * 100

    return accuracy
