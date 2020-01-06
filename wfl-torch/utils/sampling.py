#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import GetDataUtil
import torch
#import torchvision
from torch.utils.data import DataLoader, Dataset
from keras.utils import np_utils
def vehicle_iid(dataset,num_users):
    num=int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

class Vehicle_train(Dataset):
    def __init__(self,filename="./rawdataset2/RandomCrop_NPAWF_Noise_orgin_all_10000.npy"):
        super(Vehicle_train,self).__init__()
        self.data,self.label=self.readfile(filename)
    def readfile(self,filename="./rawdataset2/RandomCrop_NPAWF_Noise_orgin_all_10000.npy"):
        data,  label = GetDataUtil.getData(filename)
        label=label-1
#        print(label)
#        label = np_utils.to_categorical(label-1,5)
        return data,label
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        data=self.data[idx]
        label=self.label[idx]
#        data=torch.from_numpy(data)
#        label=torch.from_numpy(label)
        return data,label
class Vehicle_test(Dataset):
    def __init__(self,filename="./DataSet_NPSave/JustifiedData.npy"):
        super(Vehicle_test,self).__init__()
        self.data,self.label=self.readfile(filename)
    def readfile(self,filename="./DataSet_NPSave/JustifiedData.npy"):
#        filename="./DataSet_NPSave/JustifiedData.npy"
        data,  label = GetDataUtil.getData(filename)
        label=label-1
#        label = np_utils.to_categorical(label-1,5)
        return data,label
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        data=self.data[idx]
        label=self.label[idx]
#        data=torch.from_numpy(data)
#        label=torch.from_numpy(label)
        return data,label
def balanced_dataset(num_users=3):
    dict_users={}
    file0='./data/data1.npy'
    file1='./data/data2.npy'
    file2='./data/data3.npy'
    all_idxs=[[i for i in range(len(Vehicle_train(file0)))],[i for i in range(len(Vehicle_train(file1)))],[i for i in range(len(Vehicle_train(file2)))]]
    filelist=[file0,file1,file2]
    for i in range(num_users):
        dict_users[i]=set(np.random.choice(all_idxs[i], len(Vehicle_train(filelist[i])), replace=False))
    return dict_users
def unbalanced_dataset(num_users=3):
    dict_users={}
    file0='./data/data4.npy'
    file1='./data/data5.npy'
    file2='./data/data6.npy'
    all_idxs=[[i for i in range(len(Vehicle_train(file0)))],[i for i in range(len(Vehicle_train(file1)))],[i for i in range(len(Vehicle_train(file2)))]]
    filelist=[file0,file1,file2]
    for i in range(num_users):
        dict_users[i]=set(np.random.choice(all_idxs[i], len(Vehicle_train(filelist[i])), replace=False))
    return dict_users
def balanced_testset(num_users=3):
    dict_users={}
    file0='./data/testdata1.npy'
    file1='./data/testdata2.npy'
    file2='./data/testdata3.npy'
    all_idxs=[[i for i in range(len(Vehicle_test(file0)))],[i for i in range(len(Vehicle_test(file1)))],[i for i in range(len(Vehicle_test(file2)))]]
    filelist=[file0,file1,file2]
    for i in range(num_users):
        dict_users[i]=set(np.random.choice(all_idxs[i], len(Vehicle_train(filelist[i])), replace=False))
    return dict_users
class Vehicle_balanced(Dataset):
    def __init__(self,filename="./DataSet_NPSave/JustifiedData.npy"):
        super(Vehicle_balanced,self).__init__()
        self.data,self.label=self.readfile(filename)
    def readfile(self,filename="./DataSet_NPSave/JustifiedData.npy"):
        data,  label = GetDataUtil.getData(filename)
        label=label-1
#        label = np_utils.to_categorical(label-1,5)
        return data,label
    def __len__(self):
        return len(self.data)
    def __getitem__(self):
        data=self.data
        label=self.label
#        data=torch.from_numpy(data)
#        label=torch.from_numpy(label)
        return data,label
def traffic_pred(num_users=3):
    dict_users={}
    file0='./data/datasetA.npy'
    file1='./data/datasetB.npy'
    file2='./data/datasetC.npy'
    all_idxs=[[i for i in range(len(Traffic(file0)))],[i for i in range(len(Traffic(file1)))],[i for i in range(len(Traffic(file2)))]]
#    print('--------------------')
    filelist=[file0,file1,file2]
    for i in range(num_users):
        dict_users[i]=set(np.random.choice(all_idxs[i], len(Traffic(filelist[i])), replace=False))
#    print('========================')
    return dict_users
class Traffic(Dataset):
    def __init__(self,filename='./data/datasetA.npy'):
        super(Traffic,self).__init__()
        self.data,self.label=self.readfile(filename)
    def readfile(self,filename):
#        filename="./DataSet_NPSave/JustifiedData.npy"
        file=filename
#        print(file)
        data,  label = Get_data(file)
#        label=label-1
#        label = np_utils.to_categorical(label-1,5)
        return data,label
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        data=self.data[idx]
        label=self.label[idx]
#        if self.is_transform:
#            data=self.is_transform(data)
#        print(type(data))
#        data=torch.from_numpy(data)
#        label=torch.from_numpy(label)
        return data,label
def Get_data(dataPath):
    X = []
    Y = []
    dataSet = np.load(dataPath)
    for data in dataSet:
        X.append(data["data"])
        Y.append(data["label"])
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

