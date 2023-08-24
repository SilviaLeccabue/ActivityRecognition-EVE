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
	
	
		

if __name__ == "__main__":
	
	main_worker()
	test_epoch()
