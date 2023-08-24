import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
from utils import AverageMeter, calculate_accuracy
import wandb

def test_epoch(model, data_loader, device):
    model.eval()

    accuracies = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            data = np.transpose(data, (0, 2 , 1, 3, 4))
            data, targets = data.to(device), targets.to(device)
            outputs = model(data) 
            targets = targets.float()        
            acc = calculate_accuracy(outputs, targets)
            accuracies.update(acc, data.size(0))



    # show info
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), accuracies.avg * 100))
    return accuracies.avg

    
