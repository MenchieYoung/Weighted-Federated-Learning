#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import classification_report
is_support=torch.cuda.is_available()
if is_support:
    device=torch.device('cuda:0')
def test_img(net_g, datatest):
    net_g.to(device)
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest,batch_size=len(datatest))
    l = len(data_loader)
    for idx, (data, label) in enumerate(data_loader):
        data=data.to(device).float()
        label=label.to(device)
        log_probs = net_g(data)
#        label=label.long()
#        label=Variable(label).type(torch.LongTensor)
#        print(log_probs.shape,label.shape)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, label).item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum()

#    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if True:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), float(accuracy)))
    print(classification_report(label,y_pred,digits=4))
    return accuracy, test_loss

def test_traffic(net_g, datatest):
    net_g.to(device)
    net_g.eval()
    # testing
#    test_loss = 0
    data_loader = DataLoader(datatest,batch_size=len(datatest))
    l = len(data_loader)
    for idx, (data, label) in enumerate(data_loader):
        data=data.to(device).float()
        label=label.to(device)
        log_probs = net_g(data)
#        label=label.long()
#        label=Variable(label).type(torch.LongTensor)
#        print(log_probs.shape,label.shape)
        # sum up batch loss
#        test_loss += nn.MSELoss(log_probs, label).item()
        # get the index of the max log-probability
#        y_pred = log_probs.data.max(1, keepdim=True)[1]
#        correct += y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum()

#    test_loss /= len(data_loader.dataset)
#    accuracy = 100.00 * correct / len(data_loader.dataset)
#    if True:
#        print('\nTest set: Average loss: {:.4f} \n)\n'.format(
#            test_loss))
    return  log_probs,label

