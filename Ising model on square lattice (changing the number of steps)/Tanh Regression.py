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
from scipy.optimize import curve_fit


def fitfunction(v,a,b,c,d):
    return (a * np.tanh((b * v) + c) + d)



model1 = ['trans','lstm','fcn','cnn']
model2 = ['Trans','LSTM','FCN','CNN']
model_name = model1[3]
file_name = model2[3]

a_x = []
a_y = []
c = 0
step_range = [2,3,4,5,6,7,8,9,10,20,40,60,80,100]
for step in tqdm(step_range):
    a_y.append([])
    for time in range(10):
        # load the deep learning output data
        raw_x = np.load('./'+str(file_name)+'/'+model_name+'_T_'+str(step)+'_'+str(time)+'.npy')
        raw_y = np.load('./'+str(file_name)+'/'+model_name+'_predict_'+str(step)+'_'+str(time)+'.npy')

        X = raw_x.flatten()
        Y = raw_y.flatten()
        
        # optimise the tanh like fitting
        pars, cov = curve_fit(fitfunction,X,Y,maxfev=10000)
        # calculate the predicted Tc
        Uc = x_05 = (np.arctanh((0.5-pars[3])/pars[0])-pars[2])/pars[1]
        a_y[c].append(Uc)
        
    a_x.append(step)
    c = c + 1


a_x = np.array(a_x)
a_y = np.array(a_y)




np.save(model_name+'_x_tanhFit.npy',a_x)
np.save(model_name+'_y_tanhFit.npy',a_y)


