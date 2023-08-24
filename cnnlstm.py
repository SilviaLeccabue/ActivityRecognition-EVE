import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101, ResNet101_Weights
# from residual_attention_net import Model



class CNNLSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNLSTM, self).__init__()
        
        self.resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3, batch_first= False)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  
                
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         
            #print(out.size())
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x


# ResNet Model
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
        self.model.fc = Identity()
        
      
    # forward function of ResNet model
    def forward(self, x):
        out = self.model(x)
        return out

# LSTM
class Combine(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=100, layer_dim=3, output_dim=3):
        super(Combine, self).__init__()
        # ResNet
        self.resnetmodel = ResNetModel()

        # Building LSTM
        # Hidden dimensions
        self.hidden_dim = hidden_dim
   
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # initialize model
        input_dim = 2048
        hidden_dim = 100
        layer_dim = 3 # changed from 3 to 4
        output_dim = 3
        # call convolutional NN
        for t in range(x.size(1)):
            out = self.resnetmodel(x[:, t, :, :, :])

            # reshape for LSTM
            out = out.view(-1, t, input_dim)


        h0 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim).requires_grad_().to(next(self.parameters()).device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, out.size(0), self.hidden_dim).requires_grad_().to(next(self.parameters()).device)
                
        out, (hn, cn) = self.lstm(out, (h0.detach(), c0.detach()))
        
        out = self.fc(out)
        
        out = self.sigmoid(out)
        out = out.view(out.size(0)*seq_dim)
        return out


