# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 07:43:08 2021

@author: Administrator
"""
import numpy as np
import copy
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
from multiprocessing import Process
import time
from PIL import Image

# load data
train_config1 = np.load('./raw data/G00_0-2.npy')
train_config2 = np.load('./raw data/G00_6-8.npy')
train_config = np.concatenate((train_config1, train_config2), axis=0)
train_T1 = np.load('./raw data/U_0-2.npy')
train_T2 = np.load('./raw data/U_6-8.npy')
train_T = np.concatenate((train_T1, train_T2), axis=0)

test_config = np.load('./raw data/G00_GT.npy')
test_T = np.load('./raw data/U_GT.npy')

# visualize the spin configuration (only a part of the whole spin configuration)
# U = 0, step = 1~10

for i in range(1,11):
    plt.figure(i, figsize=(5,5))
    tmp_spin = train_config[0,i-1,:16,:16]
    plt.imshow(tmp_spin)
    plt.title('T=0' + 'step='+str(i), fontsize=10)
    plt.axis('off')


# process the data into series
# trainin set
X_train = []
Y_train = []
for i in range(len(train_config)):
    tmp_x = []
    for j in range(20):
        col_index = np.random.randint(0, 288)
        raw_index = np.random.randint(0, 288)
        tmp_x.append(train_config[i,:,col_index,raw_index])
    X_train.append(tmp_x)
    if train_T[i] <= 0.2:
        Y_train.append(0)
    else:
        Y_train.append(1)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
print('Input shape of training data:', X_train.shape)
# testing set
X_test = []
Y_test = []
for i in range(len(test_config)):
    tmp_x = []
    for j in range(20):
        col_index = np.random.randint(0, 256)
        raw_index = np.random.randint(0, 256)
        tmp_x.append(test_config[i,:,col_index,raw_index])
    X_test.append(tmp_x)
    if test_T[i] <= 0.88:
        Y_test.append(0)
    else:
        Y_test.append(1)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print('Input shape of testing data:', X_test.shape)










