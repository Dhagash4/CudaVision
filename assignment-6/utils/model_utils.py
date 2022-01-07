import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from utils.plot_utils import *
from sklearn.metrics import confusion_matrix


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

    return model, optimizer


def test_model(model, test_loader, device, plot_cm, label_dict):

    correct = 0
    total = 0
    labels_out = []
    pred_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        preds = torch.argmax(outputs, dim=1)
        correct += len(torch.where(preds == labels)[0])
        total += len(labels)
        pred_labels.append(preds.detach().cpu().numpy())
        labels_out.append(labels.detach().cpu().numpy())

    # Total correct predictions and loss
    accuracy = correct / total * 100
    if plot_cm:
        labels_out = np.concatenate(labels_out, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)
        cm = confusion_matrix(labels_out, pred_labels)
        plot_confusion_matrix(cm, label_dict)

    return accuracy


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def get_act(act_name):
    """ Gettign activation given name """
    assert act_name in ["ReLU", "Sigmoid", "Tanh"]
    activation = getattr(nn, act_name)
    return activation()


def get_dropout(drop_p):
    """ Getting a dropout layer """
    if(drop_p):
        drop = nn.Dropout(p=drop_p)
    else:
        drop = nn.Identity()
    return drop


class Reshape(nn.Module):
    """ Module for reshaping a tensor"""

    def __init__(self, size):
        """ Module initializer"""
        super().__init__()
        self.size = size

    def forward(self, x):
        """ Rehaping channel spatial dimension"""
        y = x.view(-1, *self.size)
        return y
