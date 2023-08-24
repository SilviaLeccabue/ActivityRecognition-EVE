import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from basic_layers import ResidualBlock
from attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule_stage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule_stage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 1024)
        self.attention_module3 = AttentionModule_stage3(1024, 1024)
        self.residual_block4 = ResidualBlock(1024, 2048)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.mpool2 = nn.Sequential(
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=7, stride=1)
        # )
        
        self.fc =  nn.Sequential(
            nn.Linear(2048, 2),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.conv1(x)
        #out = self.mpool1(out)
        out = self.residual_block1(out)
        out = self.attention_module1(out)
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.residual_block3(out)
        
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
 
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        #out = torch.cat([x, y], dim=0)
        gaze = self.fc(out)

        return gaze

