# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 19:05:19 2019

@author: ll
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:19:45 2019

@author: ll
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
#from torchvision import datasets, transforms
import torch
import GetDataUtil
from utils.sampling import Traffic,traffic_pred
#from utils.options import args_parser
from models.Update_mod import LocalUpdate_traffic_reuse
from models.Nets import trafficpred
from models.Fed import FedAvg
from models.test import test_traffic
import sklearn
from collections import deque
from math import sqrt
from sklearn.metrics import mean_squared_error
EPOCH=10
USER=3
start_epoch=1
std_boundary=0.001
datasets=['./data/datasetA.npy','./data/datasetB.npy','./data/datasetC.npy']
testsets=['./data/testA.npy','./data/testB.npy','./data/testC.npy']
que = deque([])
def std(loss):
    que.append(loss)
    if len(que) > 5:
        que.popleft()
    if len(que) == 5:
        return np.std(que, ddof=1)
    else:  # < 5
        return 100


#def test_std():
#    for i in range(8):
#        print(std(i))
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) 

if __name__ == '__main__':
#    test_std()

#    X_train, X_test, y_train, y_test = GetDataUtil.getTrainTestSet(dataPath = "./rawdataset2/RandomCrop_NPAWF_Noise_orgin_all_10000.npy",test_size = 0.1)

#    dict_users=vehicle_iid(Vehicle_train(),USER)
    is_support=torch.cuda.is_available()
    if is_support:
        device=torch.device('cuda:0')
#    std=100
    dict_users=traffic_pred(USER)
    net_glob=trafficpred()
    net_glob.train()
    #net_glob=torch.load('model_fed.pkl')
    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc=0

    for iter in range(EPOCH):
        w_locals, loss_locals = [], []
        #residual
#        w_avg=[]
#        if iter>=start_epoch:
#        w_avg.append(copy.deepcopy(w_glob))
#        m = max(int(args.frac * args.num_users), 1)
#        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users=USER
        for idx in range(idxs_users):
            local = LocalUpdate_traffic_reuse(dataset=Traffic(datasets[idx]), idxs=dict_users[idx])
#            print(type(Traffic(datasets[idx])))
            w, loss = local.train(net=copy.deepcopy(net_glob))
            w_locals.append(copy.deepcopy(w))
#            w_avg.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
#        std_tmp=std(loss_avg)
#        print(std_tmp)
        # update global weights
        w_glob = FedAvg(w_locals) 
#        if std_tmp<std_boundary:
#            w_avg.append(copy.deepcopy(w_glob))
#            w_glob=FedAvg(w_avg)
        #residual

#        if iter>=start_epoch:
#            w_glob=FedAvg(w_avg)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        net_glob.eval()
#        acc_test, loss_test = test_img(net_glob, Vehicle_test())
#        if acc_test>acc:
#            torch.save(net_glob,'model_fed_res.pkl')
#            acc=acc_test

        # print loss
        

    # plot loss curve
#    plt.figure()
#    plt.plot(range(len(loss_train)), loss_train)
#    plt.ylabel('train_loss')
#    plt.savefig('./fed_vehicle_res_{}.png'.format(EPOCH))

    # testing
    print('test raw method\n')
#    net_glob=torch.load('model_fed_res.pkl')
    
    for i in range(3):
        output,label = test_traffic(net_glob, Traffic(datasets[i]))
        output=output.view(-1)
        label=label.view(-1)
#        print('label',label.shape)
    #    acc_test, loss_test = test_img(net_glob, Vehicle_test())
    #    print("Training accuracy: {:.4f}".format(acc_train))
#        print("Training loss: {}".format(loss_train))
        rmse = sqrt(mean_squared_error(label.data.cpu().numpy(), output.data.cpu().numpy()))
        print('Train RMSE: %.3f' % rmse)
        error=mean_absolute_error(label.data.cpu().numpy(), output.data.cpu().numpy())
        print("Train MAPE: {0:.4f}%".format(error))
    #    print('output',output)
#        plt.figure()
#        plt.plot(label.data.cpu().numpy(),'b',label='real')
#        plt.plot(output.data.cpu().numpy(),'r',label='pred')
#        plt.show()
    for i in range(3):
        output,label = test_traffic(net_glob, Traffic(testsets[i]))
        output=output.view(-1)
        label=label.view(-1)
#        print('label',label.shape)
    #    acc_test, loss_test = test_img(net_glob, Vehicle_test())
    #    print("Training accuracy: {:.4f}".format(acc_train))
#        print("Training loss: {}".format(loss_train))
        rmse = sqrt(mean_squared_error(label.data.cpu().numpy(), output.data.cpu().numpy()))
        print('Test RMSE: %.3f' % rmse)
        error=mean_absolute_error(label.data.cpu().numpy(), output.data.cpu().numpy())
        print("Test MAPE: {0:.4f}%".format(error))
#    print("Testing accuracy: {:.4f}".format(acc_test))
#    print("Testing loss:{}".format(loss_test))



