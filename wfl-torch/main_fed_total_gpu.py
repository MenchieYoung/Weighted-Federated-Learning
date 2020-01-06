# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:28:12 2019

@author: ll
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:34:50 2019

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
from utils.sampling import vehicle_iid,Vehicle_train,Vehicle_test,balanced_dataset
#from utils.options import args_parser
from models.Update_mod import LocalUpdate_mod
from models.Nets import CNNLSTM
from models.Fed import FedDyUp,FedAvg
from models.test import test_img
import sklearn
EPOCH=20
USER=3
datasets=['./data/data1.npy','./data/data2.npy','./data/data3.npy']
unbalanced_datasets=['./data/data4.npy','./data/data5.npy','./data/data6.npy']
testsets=['./data/testdata1.npy','./data/testdata2.npy','./data/testdata3.npy']
if __name__ == '__main__':
#    X_train, X_test, y_train, y_test = GetDataUtil.getTrainTestSet(dataPath = "./rawdataset2/RandomCrop_NPAWF_Noise_orgin_all_10000.npy",test_size = 0.1)
    
#    dict_users=vehicle_iid(Vehicle_train(),USER)
    #gpu
    is_support=torch.cuda.is_available()
    if is_support:
        device=torch.device('cuda:0')
        
    dict_users=balanced_dataset(USER)
#    dict_users=unbalanced_dataset(USER)
    net_glob=CNNLSTM()
    net_glob.to(device)
    net_glob.train()
    w_glob = net_glob.state_dict()
    loss_train = []
    acc=0
    lc_res=[0,0,0,0,0]
    lc_res=np.array(lc_res)
    for iter in range(EPOCH):
        w_locals, loss_locals = [], []
        tmp_res=[]
        idxs_users=USER
        for idx in range(idxs_users):
            lctmp_res=lc_res
            lc_res=[] 
            lc_res=np.array(lc_res)
            local = LocalUpdate_mod( train_dataset=Vehicle_train(datasets[idx]),test_dataset=Vehicle_test(testsets[idx]), idxs=dict_users[idx])
#            local = LocalUpdate_mod( train_dataset=Vehicle_train(unbalanced_datasets[idx]),test_dataset=Vehicle_test(testsets[idx]), idxs=dict_users[idx])
            w, loss,res = local.train(net=copy.deepcopy(net_glob))
            lc_res=res
            tmp_res.append(sum(lc_res-lctmp_res)+1)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        tmp=np.array(tmp_res,dtype=np.float16)
        basis=tmp[0]
        tmp=tmp/basis
        w_glob = FedDyUp(w_locals,tmp,basis)

        net_glob.load_state_dict(w_glob)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, Vehicle_test())

        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
    np.save('loss',loss_train)



    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, Vehicle_train())
    acc_test, loss_test = test_img(net_glob, Vehicle_test())
    print("Training accuracy: {:.4f}".format(acc_train))
    print("Training loss: {}".format(loss_train))
    print("Testing accuracy: {:.4f}".format(acc_test))
    print("Testing loss:{}".format(loss_test))

