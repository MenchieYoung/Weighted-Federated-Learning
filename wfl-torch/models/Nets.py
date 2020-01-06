#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F




class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM,self).__init__()
        self.conv1=nn.Conv1d(6,16,2)#
        self.conv2=nn.Conv1d(16,32,2)
        self.pool=nn.MaxPool1d(2)
        self.conv3=nn.Conv1d(32,16,2)
        self.conv4=nn.Conv1d(16,8,2)
        self.lstm=nn.LSTM(147,8)
        self.fc1=nn.Linear(64,16)
        self.fc2=nn.Linear(16,5)
        
    def forward(self,x):
#        print(x.shape)
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        
        x,_=self.lstm(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x)
class trafficpred(nn.Module):
    def __init__(self):
        super(trafficpred,self).__init__()
        self.conv1=nn.Conv1d(24,16,2)
        self.conv2=nn.Conv1d(16,8,2)
        self.lstm=nn.LSTM(5,4)
        self.fc1=nn.Linear(32,24)
        
    def forward(self,x):
#        print(x.shape)
        x=x.permute(0,2,1)
#        print(x.shape)
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x,_=self.lstm(x)
#        print(x.shape)
        x=x.view(x.size(0),-1)
#        print(x.shape)
        x=self.fc1(x)
        return x
        
        
        
        
        
        
        
        
        
        
        
        
        
        