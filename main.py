import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
import wandb
from train import train_epoch, val_epoch
#from validation import test_epoch
from opts import parse_opts
from model import generate_model
from torch.optim import lr_scheduler
from eve_sequence import *
from utils import AverageMeter, calculate_accuracy


def test(opt, model, data_loader, device):
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
	# 		total += targets.size(0)
	# 		correct += (predicted == labels).sum().item()
	# 	test_accuracy = 100 * correct // total

	# # show info
	# print('Test set ({:d} samples): Acc: {:.4f}%'.format(len(data_loader.dataset), test_accuracy))


def main_worker():
	opt = parse_opts()

	seed = 42
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# CUDA for PyTorch
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	torch.set_num_threads(1)
	wandb.login()
	wandb.init(

        # set the wandb project where this run will be logged
        project="eve",
        # track hyperparameters and run metadata
        config= {"learning_rate": opt.lr_rate,
        "epochs": opt.n_epochs
		}
    )
	# defining model
	model= generate_model(opt, device)
	
	# get data loaders
	train_loader = train_dataloader()
	val_loader = val_dataloader()
	test_loader = test_dataloader()

	# optimizer
	crnn_params = list(model.parameters())
	optimizer = torch.optim.Adam(crnn_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)

	# scheduler = lr_scheduler.ReduceLROnPlateau(
	# 	optimizer, 'min', patience=opt.lr_patience)
	#criterion = nn.CrossEntropyLoss()
	weights = [ 0.49962648,  1.11133267, 10.13333333]
	class_weights = torch.FloatTensor(weights).cuda()
	criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
	# resume model
	# if opt.resume_path:
	# 	start_epoch = resume_model(opt, model, optimizer)
	# else:
	# 	start_epoch = 1
	start_epoch = 1
	
	# start training
	for epoch in range(start_epoch, opt.n_epochs + 1):
		train_loss, train_acc = train_epoch(
			model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
		wandb.log({"train_accuracy_epoch": train_acc*100, "train_loss_epoch": train_loss})
		val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
		wandb.log({"val_accuracy_epoch": val_acc*100, "val_loss_epoch": val_loss})
			# saving weights to checkpoint
		if (epoch) % opt.save_interval == 0:

			state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
			torch.save(state, os.path.join('tf_logs', f'{opt.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
			print("Epoch {} model saved!\n".format(epoch))
	#test(opt, model, test_loader, device )
	
		

if __name__ == "__main__":
	main_worker()
