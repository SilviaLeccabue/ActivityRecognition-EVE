import torch
from torch.utils.data import Dataset 
import cv2
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


__all__ = ['VideoDataset', 'VideoLabelDataset']

class VideoDataset(Dataset):
   
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform 

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video """
        video = self.dataframe.iloc[index].path
        if self.transform:
            video = self.transform(video)
        return video


class VideoLabelDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform 
        
    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video and its label """
        video = self.dataframe.iloc[index].path
        labels=self.one_hot_encoding()
        label = labels[index]
        #label = self.dataframe.iloc[index].label
        if self.transform:
            video = self.transform(video)
        return video, label
    
    def one_hot_encoding(self):
        
        y = LabelEncoder().fit_transform(self.dataframe.label)
        return y
   








