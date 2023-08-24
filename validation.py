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

def test_epoch(opt, model, data_loader, device):
	# checkpoint = torch.load(opt.resume_path)
	# print(checkpoint)
	model=generate_model(opt, device)
	model.load_state_dict(torch.load('tf_logs/cnnlstm-Epoch-10-Loss-2.255442947894335.pth')['state_dict'])
	print("Model Restored from Epoch {}".format(torch.load('tf_logs/cnnlstm-Epoch-10-Loss-2.255442947894335.pth')['epoch']))
	prdY = []
	model.eval()
	with torch.no_grad():
		for (data, targets) in data_loader:
			data = np.transpose(data, (0, 2 , 1, 3, 4))
			data, targets = data.to(device), targets.to(device)
			outputs = model(data) 
			# targets = targets.float()        
			_, predicted = torch.max(outputs.data, 1)
			# save the predictions
			predictions_npy = predicted.data.cpu().detach().numpy()  
			if(len(prdY) >0):
				prdY = np.concatenate((prdY, predictions_npy))
			else:
				prdY = predictions_npy
		
		prdY = prdY.reshape(-1, 1)
		
		print(prdY.shape)
		print(targets.shape)
		correct = (targets == prdY).sum()

		accuracy = (correct/len(data_loader.dataset))*100
		print(' Single Window Prediction Accuracy: {:.1f}%'.format(accuracy))

    
