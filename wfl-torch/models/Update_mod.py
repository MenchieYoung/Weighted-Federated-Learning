# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:27:09 2019

@author: ll
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from torch.autograd import Variable
import sklearn
import torch.nn.functional as F
from collections import deque
from models.Fed import FedAvg
import copy
from math import sqrt
from sklearn.metrics import mean_squared_error
EPOCH=20
is_support=torch.cuda.is_available()
if is_support:
    device=torch.device('cuda:0')
std_boundary=0.01
std_boundary_traffic=1e-5
que = deque([])    
def std(loss):
    que.append(loss)
    if len(que) > 5:
        que.popleft()
    if len(que) == 5:
        return np.std(que)
    else:  # < 5
        return 100
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        #self.labels=labels
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, label = self.dataset[self.idxs[item]]#,self.labels[self.idxs[item]]
        return data, label


class LocalUpdate(object):
    def __init__(self,  train_dataset=None,test_dataset=None, idxs=None):
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(train_dataset, idxs), batch_size=512, shuffle=True)
        self.ldr_test=DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=True)

    def train(self, net):
        net.to(device)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

        epoch_loss = []
        for iter in range(EPOCH):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
#                print('shape:data,labels',data.shape,labels.shape,labels)
#                data=torch.LongTensor(data)
#                labels=torch.Tensor(labels)
                data=data.to(device)
                labels=labels.to(device)
                net.zero_grad()
                log_probs = net(data.float())
                labels=Variable(labels).type(torch.cuda.LongTensor)
#                print('shape:log',log_probs.shape,log_probs)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if  True:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(data), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        net.eval()
        for idx,(data,label) in enumerate(self.ldr_test):
            data=data.to(device)
            label=label.to(device)
            log_probs=net(data.float())
            label=Variable(label).type(torch.cuda.LongTensor)
#            test_loss += F.cross_entropy(log_probs, label, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
#            correct += y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum()
        cm=sklearn.metrics.confusion_matrix(label,y_pred)
        print(cm)
        res=[]
        for i in range(len(cm)):
            res.append(cm[i][i]/sum(cm[i]))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),np.array(res)

class LocalUpdate_traffic_mod(object):
    def __init__(self,  train_dataset=None,test_dataset=None, idxs=None):
        self.loss_func = nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(train_dataset, idxs), batch_size=len(train_dataset), shuffle=False)
        self.ldr_test=DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=False)

    def train(self, net):
        net.to(device)
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        epoch_loss = []
        for iter in range(500):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                data=data.to(device).float()
                labels=labels.to(device).float()
                w_avg=[]
                w_avg.append(copy.deepcopy(net.state_dict()))
#                print('shape:data,labels',data.shape,labels.shape,labels)
#                data=torch.cuda.FloatTensor(data)
#                labels=Variable(labels).type(torch.FloatTensor)
#                labels=torch.Tensor(labels)
                
                log_probs = net(data)
#                labels=Variable(labels).type(torch.LongTensor)
#                print('shape:log',log_probs.shape,log_probs)
                loss = self.loss_func(log_probs, labels)
                net.zero_grad()
                loss.backward()
                optimizer.step()
                std_tmp=std(loss.item())
                if std_tmp<std_boundary_traffic:
                    w_avg.append(copy.deepcopy(net.state_dict()))
                    w_glob=FedAvg(w_avg)
                    net.load_state_dict(w_glob)
                if  True:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(data), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        net.eval()
        for idx,(data,label) in enumerate(self.ldr_test):
            data=data.to(device)
            label=label.to(device)
            log_probs=net(data.float())
            label=Variable(label).type(torch.cuda.FloatTensor)
        error=mean_absolute_error(label.data.cpu().numpy(), log_probs.data.cpu().numpy())
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),error
class LocalUpdate_traffic_DWUP(object):
    def __init__(self,  train_dataset=None,test_dataset=None, idxs=None):
        self.loss_func = nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(train_dataset, idxs), batch_size=len(train_dataset), shuffle=False)
        self.ldr_test=DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=False)

    def train(self, net):
        net.to(device)
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        epoch_loss = []
        for iter in range(500):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                data=data.to(device).float()
                labels=labels.to(device).float()
#                w_avg=[]
#                w_avg.append(copy.deepcopy(net.state_dict()))
#                print('shape:data,labels',data.shape,labels.shape,labels)
#                data=torch.cuda.FloatTensor(data)
#                labels=Variable(labels).type(torch.FloatTensor)
#                labels=torch.Tensor(labels)
                
                log_probs = net(data)
#                labels=Variable(labels).type(torch.LongTensor)
#                print('shape:log',log_probs.shape,log_probs)
                loss = self.loss_func(log_probs, labels)
                net.zero_grad()
                loss.backward()
                optimizer.step()
#                std_tmp=std(loss.item())
#                if std_tmp<std_boundary_traffic:
#                    w_avg.append(copy.deepcopy(net.state_dict()))
#                    w_glob=FedAvg(w_avg)
#                    net.load_state_dict(w_glob)
                if  True:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(data), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        net.eval()
        for idx,(data,label) in enumerate(self.ldr_test):
            data=data.to(device)
            label=label.to(device)
            log_probs=net(data.float())
            label=Variable(label).type(torch.cuda.FloatTensor)
        error=mean_absolute_error(label.data.cpu().numpy(), log_probs.data.cpu().numpy())
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),error
class LocalUpdate_mod(object):
    def __init__(self,  train_dataset=None,test_dataset=None, idxs=None):
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(train_dataset, idxs), batch_size=512, shuffle=True)
        self.ldr_test=DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=True)

    def train(self, net):
        net.to(device)
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

        epoch_loss = []
        for iter in range(EPOCH):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                w_avg=[]
                w_avg.append(copy.deepcopy(net.state_dict()))
#                print('shape:data,labels',data.shape,labels.shape,labels)
#                data=torch.LongTensor(data)
#                labels=torch.Tensor(labels)
                data=data.to(device)
                labels=labels.to(device)
                net.zero_grad()
                log_probs = net(data.float())
                labels=Variable(labels).type(torch.cuda.LongTensor)
#                print('shape:log',log_probs.shape,log_probs)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                std_tmp=std(loss.item())
                if std_tmp<std_boundary:
                    w_avg.append(copy.deepcopy(net.state_dict()))
                    w_glob=FedAvg(w_avg)
                    net.load_state_dict(w_glob)
                if  True:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(data), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        net.eval()
        for idx,(data,label) in enumerate(self.ldr_test):
            data=data.to(device)
            label=label.to(device)
            log_probs=net(data.float())
            label=Variable(label).type(torch.cuda.LongTensor)
#            test_loss += F.cross_entropy(log_probs, label, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
#            correct += y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum()
        cm=sklearn.metrics.confusion_matrix(label,y_pred)
        print(cm)
        res=[]
        for i in range(len(cm)):
            res.append(cm[i][i]/sum(cm[i]))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss),np.array(res)

class LocalUpdate_traffic(object):
    def __init__(self,  dataset=None, idxs=None):
        self.loss_func = nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=len(dataset), shuffle=False)

    def train(self, net):
        net.to(device)
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        epoch_loss = []
        for iter in range(500):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                data=data.to(device).float()
                labels=labels.to(device).float()
#                print('shape:data,labels',data.shape,labels.shape,labels)
#                data=torch.cuda.FloatTensor(data)
#                labels=Variable(labels).type(torch.FloatTensor)
#                labels=torch.Tensor(labels)
                
                log_probs = net(data)
#                labels=Variable(labels).type(torch.LongTensor)
#                print('shape:log',log_probs.shape,log_probs)
                loss = self.loss_func(log_probs, labels)
                net.zero_grad()
                loss.backward()
                optimizer.step()
                if  True:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(data), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdate_traffic_reuse(object):
    def __init__(self,  dataset=None, idxs=None):
        self.loss_func = nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=len(dataset), shuffle=False)

    def train(self, net):
        net.to(device)
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        epoch_loss = []
        for iter in range(500):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                data=data.to(device).float()
                labels=labels.to(device).float()
                w_avg=[]
                w_avg.append(copy.deepcopy(net.state_dict()))
#                print('shape:data,labels',data.shape,labels.shape,labels)
#                data=torch.cuda.FloatTensor(data)
#                labels=Variable(labels).type(torch.FloatTensor)
#                labels=torch.Tensor(labels)
                
                log_probs = net(data)
#                labels=Variable(labels).type(torch.LongTensor)
#                print('shape:log',log_probs.shape,log_probs)
                loss = self.loss_func(log_probs, labels)
                net.zero_grad()
                loss.backward()
                optimizer.step()
                std_tmp=std(loss.item())
                if std_tmp<std_boundary_traffic:
                    w_avg.append(copy.deepcopy(net.state_dict()))
                    w_glob=FedAvg(w_avg)
                    net.load_state_dict(w_glob)
                if  True:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(data), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)