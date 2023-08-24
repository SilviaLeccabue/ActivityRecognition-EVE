import torch
import torchvision
import datasets
import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
#from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def train_dataloader():
    train_dataset = datasets.VideoLabelDataset(
        "./datasources/data_train.csv",
        transform=torchvision.transforms.Compose([
            transforms.VideoFolderPathToTensor(max_len=30, padding_mode=None),
            transforms.VideoCenterCrop([224, 224]),
          
        ])
    )
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle = True)
    return data_loader_train

def val_dataloader():
    val_dataset = datasets.VideoLabelDataset(
        "./datasources/data_val.csv",
        transform=torchvision.transforms.Compose([
            transforms.VideoFolderPathToTensor(max_len=30, padding_mode=None),
            transforms.VideoCenterCrop([224, 224]),
            
        ])
    )
    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle = True)
    return data_loader_val

def test_dataloader():
    test_dataset = datasets.VideoLabelDataset(
        "./datasources/data_test.csv",
        transform=torchvision.transforms.Compose([
            transforms.VideoFolderPathToTensor(max_len=30, padding_mode=None),
            transforms.VideoCenterCrop([224, 224]),
           
            
        ])
    )
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = False)
    return data_loader_test
