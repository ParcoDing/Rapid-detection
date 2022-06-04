# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:17:05 2021

@author: Administrator
"""
import copy
import re
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from tqdm import trange, tqdm
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit


def TanhFitting(v,a,b,c,d):
    return (a * np.tanh((b * v) + c) + d)


X = np.load('./LSTM result/lstm_T.npy')
Y = np.load('./LSTM result/lstm_predict.npy')
pars, cov = curve_fit(TanhFitting,X,Y,maxfev=100000)
T = np.arange(0, 5.05, 0.05)
Tanh_fit = TanhFitting(T, pars[0], pars[1], pars[2], pars[3])
plt.plot(T, Tanh_fit, label='Bi-LSTM')



X = np.load('./Transformer result/trans_T.npy')
Y = np.load('./Transformer result/trans_predict.npy')
pars, cov = curve_fit(TanhFitting,X,Y,maxfev=100000)
T = np.arange(0, 5.05, 0.05)
Tanh_fit = TanhFitting(T, pars[0], pars[1], pars[2], pars[3])
plt.plot(T, Tanh_fit, label='Transformer')


plt.plot([0,5],[0.5,0.5],':',label='output = 0.5', color='black',linewidth=3)
plt.plot([2.269,2.269],[-0.5,1.5],'-.',label='Tc = 2.269', color='black',linewidth=2.5)
plt.ylim(-0.1,1.1)
plt.xlim(0,5)
plt.xlabel('T/J')
plt.ylabel('tanh fit')
plt.legend()
plt.savefig('tanh fitting.png', dpi=150)



