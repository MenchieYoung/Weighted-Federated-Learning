#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
#from torchvision import datasets, transforms
import torch
import GetDataUtil
from utils.sampling import vehicle_iid,Vehicle_train,Vehicle_test,balanced_dataset
#from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import CNNLSTM
from models.Fed import FedAvg
from models.test import test_img
import sklearn
EPOCH=20
USER=3
datasets=['./data/data1.npy','./data/data2.npy','./data/data3.npy']
unbalanced_datasets=['./data/data4.npy','./data/data5.npy','./data/data6.npy']
if __name__ == '__main__':
#    X_train, X_test, y_train, y_test = GetDataUtil.getTrainTestSet(dataPath = "./rawdataset2/RandomCrop_NPAWF_Noise_orgin_all_10000.npy",test_size = 0.1)
    
#    dict_users=vehicle_iid(Vehicle_train(),USER)
    is_support=torch.cuda.is_available()
    if is_support:
        device=torch.device('cuda:0')
    dict_users=balanced_dataset(USER)
#    dict_users=unbalanced_dataset(USER)
    net_glob=CNNLSTM()
    net_glob.train()
    w_glob = net_glob.state_dict()
    loss_train = []
    acc=0
    for iter in range(EPOCH):
        w_locals, loss_locals = [], []
        idxs_users=USER
        for idx in range(idxs_users):
            local = LocalUpdate( dataset=Vehicle_train(datasets[idx]), idxs=dict_users[idx])
#            local = LocalUpdate( dataset=Vehicle_train(unbalanced_datasets[idx]), idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, Vehicle_test())
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
    np.save('loss',loss_train)

    # testing
#    net_glob=torch.load('model_fed.pkl')
#    torch.save(net_glob,'model_fed.pkl')
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, Vehicle_train())
    acc_test, loss_test = test_img(net_glob, Vehicle_test())
    print("Training accuracy: {:.4f}".format(acc_train))
    print("Training loss: {}".format(loss_train))
    print("Testing accuracy: {:.4f}".format(acc_test))
    print("Testing loss:{}".format(loss_test))

