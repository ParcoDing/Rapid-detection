# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 19:00:02 2021

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:40:49 2021

@author: Parco
"""
import copy
import re
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow
import tensorflow.keras
from tensorflow.keras.layers import *
#from tensorflow.keras.preprocessing import sequence
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from tqdm import trange,tqdm
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.optimize import curve_fit


# ========================== set env ========================

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)
tensorflow.config.experimental_run_functions_eagerly(True)



# ========================== get data ========================

step = 10

# make training set
data_X = np.load('./data/train spin.npy')[:,:,:step] # the shape of raw x is (number of samples, number of series, number of step), we only take the first 10 step
data_Y = np.load('./data/train T.npy')


X = []
Y = []
T = []
for i in trange(len(data_X)):
    X.append(data_X[i])
    T.append(data_Y[i])
    if data_Y[i] >= 4:
        Y.append(1)
    else:
        Y.append(0)


X_train = np.array(X)
X_train = np.reshape(X_train, (len(data_X),20,step,1))
Y_train = np.array(Y)
T_train = np.array(T)


# make testing set
data_X = np.load('./data/test spin.npy')[:,:,:step]
data_Y = np.load('./data/test T.npy')

X = []
Y = []
T = []
for i in trange(len(data_X)):
    X.append(data_X[i])
    T.append(data_Y[i])
    if data_Y[i] >= 2.27:
        Y.append(1)
    else:
        Y.append(0)

X_test = np.array(X)
X_test = np.reshape(X_test, (len(data_X),20,step,1))
Y_test = np.array(Y)
T_test = np.array(T)
data_X = data_Y = data_x = data_y = X = Y = T = 0


print(X_train.shape)
print(X_test.shape)



def Conv2d_BN(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = SpatialDropout2D(0.05)(x)
    return x


# ========================== build model ========================

input_layer = Input(shape=(20,step,1))

x = Conv2d_BN(input_layer, 16, 3)
x = Conv2d_BN(x, 16, 3)


x = Flatten()(x)
x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.1)(x)
x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.1)(x)

output_layer = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)


model = tensorflow.keras.models.Model(input_layer, output_layer)
model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['acc'] )
#model.summary()
checkpoint = ModelCheckpoint(filepath='lstm.h5', monitor='val_acc', mode='auto' ,save_best_only='True')
# ========================== train model ========================

history = model.fit(X_train, Y_train, batch_size=128, epochs=2, verbose=1, validation_data = (X_test,Y_test), shuffle = True, callbacks=[checkpoint])
#model.save('LSTM_HoneyComb_L=3_G00_100step_10_lr=5e-3.h5')

# ========================== record data ========================
#model.load_weights('fcn.h5')

pre = model.predict(X_test)
pre = pre.flatten()

np.save('./CNN result/cnn_predict.npy', pre)
np.save('./CNN result/cnn_T.npy', T_test)


# calculate the average output values of each T
T_ = [round(i,1) for i in T_test]
Y = []
P = []
c = []
for i in range(len(T_)):
    if T_[i] not in Y:
        Y.append(T_[i])
        P.append(pre[i])
        c.append(1)
    else:
        index = Y.index(T_[i])
        P[index] = P[index] + pre[i]
        c[index] = c[index] + 1
P = np.array(P)

c = np.array(c)
P = P/c

# fit the output by a tanh like function
def TanhFitting(v,a,b,c,d):
    return (a * np.tanh((b * v) + c) + d)
pars, cov = curve_fit(TanhFitting,Y,P,maxfev=100000)
T = np.arange(0, 5.05, 0.05)
Tanh_fit = TanhFitting(T, pars[0], pars[1], pars[2], pars[3])
plt.plot(T, Tanh_fit, label='tanh fit')


plt.scatter(Y, P, color='blue',marker='x',s=30,label='avg predicted prob')
plt.plot([2.27,2.27],[0,1],label='Tc = 2.27', color='red')
plt.xlabel('T/J')
plt.ylabel('CNN output')
plt.title('CNN')
plt.legend()      

plt.savefig('./CNN result/figure.png', dpi=150)
plt.show()
plt.close()

tensorflow.keras.backend.clear_session()






