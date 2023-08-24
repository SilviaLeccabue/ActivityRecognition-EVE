import torch
from torch import nn

import cnnlstm
# import residual_attention_net

def generate_model(opt, device):
	assert opt.model in [
		'cnnlstm'
	]
	# assert opt.pre_train_model in ['attent_resnet'
	# ]
	if opt.model == 'cnnlstm':
		model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
	# if opt.pre_train_model == 'attent_resnet':
	# 	pre_train_model = residual_attention_net.Model()
	if opt.model == 'reslstm':
	
		model = cnnlstm.Combine(input_dim, hidden_dim, layer_dim, output_dim)
	return model.to(device)
