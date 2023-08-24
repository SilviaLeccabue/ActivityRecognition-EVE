import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import argparse
import os
import random
import numpy as np
from utils import AverageMeter, calculate_accuracy
import wandb
from opts import parse_opts


def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    torch.set_num_threads(1)
    # running_accuracy = 0
    losses = AverageMeter()
    accuracies = AverageMeter()
    for batch_idx, (data, targets) in enumerate(data_loader):
        data = np.transpose(data, (0, 2 , 1, 3, 4))
        #print(data.size())
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        #targets = targets.float()
        loss = criterion(outputs, targets)
        
        acc = calculate_accuracy(outputs, targets)
        train_loss += loss.item()
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        wandb.log({"train_loss": loss})
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
        
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader), avg_loss))
            train_loss = 0.0 

    print('Train set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg*100))

    return losses.avg, accuracies.avg


def val_epoch(model, data_loader, criterion, device):
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    torch.set_num_threads(1)
    with torch.no_grad():
        for (data, targets) in data_loader:
            data = np.transpose(data, (0, 2 , 1, 3, 4))
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)  
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))
    wandb.log({"val_acc" : acc, "val_loss": loss })

    # show info
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg*100))
    return losses.avg, accuracies.avg

