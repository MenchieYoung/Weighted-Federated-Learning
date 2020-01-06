# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:36:43 2019

@author: ll
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:08:11 2019

@author: ll
"""

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
from utils.sampling import vehicle_iid,Vehicle_train,Vehicle_test,balanced_dataset,unbalanced_dataset
#from utils.options import args_parser
from models.Update import LocalUpdate_reuse
from models.Nets import CNNLSTM
from models.Fed import FedAvg
from models.test import test_img
import sklearn
from collections import deque
EPOCH=20
USER=3
start_epoch=1
std_boundary=0.01
que = deque([])
def std(loss):
    que.append(loss)
    if len(que) > 5:
        que.popleft()
    if len(que) == 5:
        return np.std(que, ddof=1)
    else:  # < 5
        return 100

datasets=['./data/data1.npy','./data/data2.npy','./data/data3.npy']
unbalanced_datasets=['./data/data4.npy','./data/data5.npy','./data/data6.npy']
#testsets=['./data/testdata1.npy','./data/testdata2.npy','./data/testdata3.npy']
#def test_std():
#    for i in range(8):
#        print(std(i))


if __name__ == '__main__':
#    test_std()

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
#            local = LocalUpdate_reuse( dataset=Vehicle_train(unbalanced_datasets[idx]), idxs=dict_users[idx])
            local = LocalUpdate_reuse( dataset=Vehicle_train(datasets[idx]), idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob))
            w_locals.append(copy.deepcopy(w))
#            w_avg.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        w_glob = FedAvg(w_locals) #if std_tmp>std_boundary else FedAvg(w_avg)
        net_glob.load_state_dict(w_glob)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, Vehicle_test())
    np.save('loss',loss_train)

    print('test raw method\n')
#    net_glob=torch.load('model_fed_res.pkl')
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, Vehicle_train())
#    acc_test, loss_test = test_img(net_glob, Vehicle_test())
    acc_test, loss_test = test_img(net_glob, Vehicle_test())
    print("Training accuracy: {:.4f}".format(acc_train))
    print("Training loss: {}".format(loss_train))
    print("Testing accuracy: {:.4f}".format(acc_test))
    print("Testing loss:{}".format(loss_test))



