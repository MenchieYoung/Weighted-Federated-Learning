# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:56:55 2019

@author: ll
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.sampling import Traffic
from models.Nets import trafficpred
import numpy as np
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sklearn
from math import sqrt
from sklearn.metrics import mean_squared_error
EPOCH=600
datasets=['./data/datasetA.npy','./data/datasetB.npy','./data/datasetC.npy']
testsets=['./data/testA.npy','./data/testB.npy','./data/testC.npy']
trainload=DataLoader(dataset=Traffic(datasets[0]),batch_size=len(Traffic(datasets[0])),shuffle=False)
testload=DataLoader(dataset=Traffic(testsets[0]),batch_size=len(Traffic(testsets[0])),shuffle=True)
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
net_glob=trafficpred()
criterion = nn.MSELoss()
optimizer = optim.Adam(net_glob.parameters(), lr=0.01)
list_loss=[]
w_glob=net_glob.state_dict()
for epoch in range(EPOCH):
    batch_loss=[]
    correct=0
    for i,(data,label) in enumerate(trainload):
        output = net_glob(data.float())
        label=Variable(label).type(torch.FloatTensor)
        loss=criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if True:
            print('Train Epoch: {} Loss: {:.6f}'.format(
                epoch,  loss.item()))
        batch_loss.append(loss.item())
print('output',output)
plt.figure()
plt.plot(label.view(-1).data.numpy(),'b',label='real')
plt.plot(output.view(-1).data.numpy(),'r',label='pred')
plt.show()
plt.savefig('./traffic.png')
rmse = sqrt(mean_squared_error(label.view(-1).data.numpy(), output.view(-1).data.numpy()))
print('Train RMSE: %.3f' % rmse)
error=mean_absolute_error(label.view(-1).data.numpy(), output.view(-1).data.numpy())
print("train mean_absolute_error{0:.2f}%".format(error))

test_loss=0
for idx,(data_test,label_test) in enumerate(testload):
    log_probs=net_glob(data_test.float())
    label_test=Variable(label_test).type(torch.FloatTensor)
    test_loss += criterion(log_probs, label_test)
rmse = sqrt(mean_squared_error(label_test.view(-1).data.numpy(), log_probs.view(-1).data.numpy()))
print('Test RMSE: %.3f' % rmse)
error=mean_absolute_error(label_test.view(-1).data.numpy(), log_probs.view(-1).data.numpy())
print("test mean_absolute_error{0:.2f}%".format(error))

#test_loss=0
#correct=0
#for idx,(data,label) in enumerate(testload):
#    log_probs=net_glob(data.float())
#    label=Variable(label).type(torch.LongTensor)
#    test_loss += F.cross_entropy(log_probs, label, reduction='sum').item()
#    y_pred = log_probs.data.max(1, keepdim=True)[1]
#    correct += y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum()
#cm=sklearn.metrics.confusion_matrix(label,y_pred)
#print(cm)
#res=[]
#for i in range(len(cm)):
#    res.append(cm[i][i]/sum(cm[i]))
#print(res)
#test_loss /= len(testload.dataset)
#accuracy = 100.00 * correct / len(testload.dataset)
#print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.4f}%)\n'.format(
#            test_loss, correct, len(testload.dataset), accuracy))