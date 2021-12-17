import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
from model_utils import save_model
from plot_utils import plot_stats


class Trainer():

    def __init__(self, name, model, epochs, train_loader, eval_loader, criterion, scheduler, optimizer, device, save_true, plot, save_freq, eval_freq) -> None:

        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criteerion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.device = device
        self.save_true = save_true
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.best_accuracy = 0.0
        self.plot = plot
        self.logger = SummaryWriter(f'runs/{name}')
        self.stats = stats = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "name": self.name
        }

    def train_epoch(self):
        """ Training a model for one epoch """

        loss_list = []
        for i, (images, labels) in enumerate(self.trainloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = self.model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = self.criterion(outputs, labels)
            loss_list.append(loss.item())

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            self.optimizer.step()

        mean_loss = np.mean(loss_list)
        return mean_loss, loss_list

    @torch.no_grad()
    def eval_model(self):
        """ Evaluating the model for either validation or test """
        correct = 0
        total = 0
        loss_list = []

        for images, labels in self.eval_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass only to get logits/output
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)
            loss_list.append(loss.item())

            # Get predictions from the maximum value
            preds = torch.argmax(outputs, dim=1)
            correct += len(torch.where(preds == labels)[0])
            total += len(labels)

        # Total correct predictions and loss
        accuracy = correct / total * 100
        loss = np.mean(loss_list)

        return accuracy, loss

    def train_model(self):
        """ Training a model for a given number of epochs"""

        train_loss = []
        val_loss = []
        loss_iters = []
        valid_acc = []

        for epoch in range(self.epochs):
            self.stats['epochs'].append(epoch)
            # validation epoch
            self.model.eval()  # important for dropout and batch norms
            accuracy, loss = self.eval_model()
            valid_acc.append(accuracy)
            val_loss.append(loss)

            # training epoch
            self.model.train()  # important for dropout and batch norms
            mean_loss, cur_loss_iters = self.train_epoch()
            self.scheduler.step()
            train_loss.append(mean_loss)
            loss_iters = loss_iters + cur_loss_iters

            if(epoch % self.eval_freq == 0 or epoch == self.epochs-1):
                self.logger.add_scalar('train_loss',
                                       round(mean_loss, 5),
                                       (epoch+1)/(self.epochs))
                self.logger.add_scalar('val_loss',
                                       round(loss, 5),
                                       (epoch+1)/(self.epochs))
                self.logger.add_scalar('accuracy',
                                       accuracy,
                                       (epoch+1)/(self.epochs))

            if(epoch % self.save_freq == 0 or epoch == self.epochs-1):
                self.stats['val_loss'] = val_loss
                self.stats['train_loss'] = train_loss
                self.stats['val_accuracy'] = valid_acc
                save_model(model=self.model,
                           optimizer=self.optimizer,
                           epoch=epoch,
                           stats=self.stats,
                           name=self.name
                           )

            if(accuracy > self.best_accuracy):
                # Saving best model
                self.best_accuracy = accuracy
                self.stats['val_loss'] = val_loss
                self.stats['train_loss'] = train_loss
                self.stats['val_accuracy'] = valid_acc
                save_model(model=self.model,
                           optimizer=self.optimizer,
                           epoch=epoch,
                           stats=self.stats,
                           name="best"
                           )
        if self.plot:
            plot_stats(train_loss=train_loss, val_loss=val_loss,
                       loss_iters=loss_iters, valid_acc=valid_acc)

        print(f"Training completed")
        return train_loss, val_loss, loss_iters, valid_acc
