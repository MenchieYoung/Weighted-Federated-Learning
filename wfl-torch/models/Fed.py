#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

def FedAvg(w):
#    w_avg=w[0]
#    print(np.array(w_avg).shape)
#    for i in len(w):
#        if i==0:
#            continue
#        else:
#            w_avg+=w[i]
#    w_avg=torch.div(w_avg, len(w))
    
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
#            print(type(w[i][k]))
        w_avg[k] = torch.div(w_avg[k], len(w))
        
    return w_avg
def FedDyUp(w,res,basis):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
#            print('i',i,res[i])
            w_avg[k] += torch.mul(w[i][k],float(res[i]))
        w_avg[k] = torch.div(w_avg[k], len(w))
#        w_avg[k]=Variable(w_avg[k]).type(torch.FloatTensor)
        w_avg[k] = torch.mul(w_avg[k],float(basis))
    return w_avg
    
    
        