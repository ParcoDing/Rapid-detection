# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 04:40:09 2021

@author: Administrator
"""
import numpy as np
import random
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange,tqdm

# plot the tanh like fitting 


x1 = np.load('fcn_y_tanhFit.npy')
y1 = np.load('fcn_x_tanhFit.npy')
X1 = []
Y1 = []
output1 = []
std1 = []
ste1 = []
bot_err1 = []
up_err1 = []
c = -1
# classify the deep learning output values by the order paras
for i in range(len(y1)):
    if y1[i] not in Y1:
        X1.append([])
        c = c + 1
        X1[c].append(x1[i])
        Y1.append(y1[i])
    else:
        X1[c].append(x1[i])
X1 = np.array(X1)
Y1 = np.array(Y1)
Y1[9:] = [12,14,16,18,20]
# calculate the ave output values and std err of each order paras T
for i in range(len(Y1)):
    std1.append(X1[i].std())
    ste1.append(X1[i].std()/(X1[i].shape[0])**0.5)
    output1.append(X1[i].mean())
    bot_err1.append(0.5*X1[i].std()/((X1[i].shape[0])**0.5))
    up_err1.append(0.5*X1[i].std()/((X1[i].shape[0])**0.5))
plt.errorbar(Y1, output1, yerr=[bot_err1,up_err1],capsize=2,marker='o',label='FCN')



    
x2 = np.load('cnn_y_tanhFit.npy')
y2 = np.load('cnn_x_tanhFit.npy')
X2 = []
Y2 = []
output2 = []
std2 = []
ste2 = []
bot_err2 = []
up_err2 = []
c = -1
for i in range(len(y2)):
    if y2[i] not in Y2:
        X2.append([])
        c = c + 1
        X2[c].append(x2[i])
        Y2.append(y2[i])
    else:
        X2[c].append(x2[i])
X2 = np.array(X2)
Y2 = np.array(Y2)
Y2[9:] = [12,14,16,18,20]
for i in range(len(Y2)):
    std2.append(X2[i].std())
    ste2.append(X2[i].std()/(X2[i].shape[0])**0.5)
    output2.append(X2[i].mean())
    bot_err2.append(0.5*X2[i].std()/((X2[i].shape[0])**0.5))
    up_err2.append(0.5*X2[i].std()/((X2[i].shape[0])**0.5))
plt.errorbar(Y2, output2, yerr=[bot_err2,up_err2],capsize=2,marker='v',label='CNN')





x3 = np.load('lstm_y_tanhFit.npy')
y3 = np.load('lstm_x_tanhFit.npy')
X3 = []
Y3 = []
output3 = []
std3 = []
ste3 = []
bot_err3 = []
up_err3 = []
c = -1
for i in range(len(y3)):
    if y3[i] not in Y3:
        X3.append([])
        c = c + 1
        X3[c].append(x3[i])
        Y3.append(y3[i])
    else:
        X3[c].append(x3[i])
X3 = np.array(X3)
Y3 = np.array(Y3)
Y3[9:] = [12,14,16,18,20]
for i in range(len(Y3)):
    std3.append(X3[i].std())
    ste3.append(X3[i].std()/(X3[i].shape[0])**0.5)
    output3.append(X3[i].mean())
    bot_err3.append(0.5*X3[i].std()/((X3[i].shape[0])**0.5))
    up_err3.append(0.5*X3[i].std()/((X3[i].shape[0])**0.5))
plt.errorbar(Y3, output3, yerr=[bot_err3,up_err3],capsize=2,marker='*',label='Bi-LSTM')





x4 = np.load('trans_y_tanhFit.npy')
y4 = np.load('trans_x_tanhFit.npy')
X4 = []
Y4 = []
output4 = []
std4 = []
ste4 = []
bot_err4 = []
up_err4 = []
c = -1
for i in range(len(y4)):
    if y4[i] not in Y4:
        X4.append([])
        c = c + 1
        X4[c].append(x4[i])
        Y4.append(y4[i])
    else:
        X4[c].append(x4[i])
X4 = np.array(X4)
Y4 = np.array(Y4)
Y4[9:] = [12,14,16,18,20]
for i in range(len(Y4)):
    std4.append(X4[i].std())
    ste4.append(X4[i].std()/(X4[i].shape[0])**0.5)
    output4.append(X4[i].mean())
    bot_err4.append(0.5*X4[i].std()/((X4[i].shape[0])**0.5))
    up_err4.append(0.5*X4[i].std()/((X4[i].shape[0])**0.5))
plt.errorbar(Y4, output4, yerr=[bot_err4,up_err4],capsize=2,marker='p',label='Transformer')








plt.plot([2,20],[2.27,2.27],'-.', color='black',linewidth=2.5)
plt.xlabel('Number of steps',fontsize=17.5)
plt.ylabel('Predicted Tc',fontsize=17.5)
plt.ylim(2,3)
plt.legend(loc='upper left',fontsize=12.5, ncol=2)
plt.xticks([2,3,4,5,6,7,8,9,10,12,14,16,18,20],['2','3','4','5','6','7','8','9','10','20','40','60','80','100'],fontsize=15)
plt.yticks(fontsize=15)
plt.gcf().subplots_adjust(left=0.15,top=0.95,bottom=0.15, right=None)
plt.savefig('Ising on Square tanhFit.png', dpi=150)
plt.show()

























