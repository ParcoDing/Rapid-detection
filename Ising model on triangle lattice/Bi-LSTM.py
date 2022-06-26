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
    if data_Y[i] >= 5:
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
    if data_Y[i] >= 3.69:
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




def slice(x,index):
    return x[:,index,:,:]

# ========================== build model ========================

input_layer = Input(shape=(20,step,1))

x0 = Lambda(slice,arguments={'index':0})(input_layer)
x1 = Lambda(slice,arguments={'index':1})(input_layer)
x2 = Lambda(slice,arguments={'index':2})(input_layer)
x3 = Lambda(slice,arguments={'index':3})(input_layer)
x4 = Lambda(slice,arguments={'index':4})(input_layer)
x5 = Lambda(slice,arguments={'index':5})(input_layer)
x6 = Lambda(slice,arguments={'index':6})(input_layer)
x7 = Lambda(slice,arguments={'index':7})(input_layer)
x8 = Lambda(slice,arguments={'index':8})(input_layer)
x9 = Lambda(slice,arguments={'index':9})(input_layer)
x10 = Lambda(slice,arguments={'index':10})(input_layer)
x11 = Lambda(slice,arguments={'index':11})(input_layer)
x12 = Lambda(slice,arguments={'index':12})(input_layer)
x13 = Lambda(slice,arguments={'index':13})(input_layer)
x14 = Lambda(slice,arguments={'index':14})(input_layer)
x15 = Lambda(slice,arguments={'index':15})(input_layer)
x16 = Lambda(slice,arguments={'index':16})(input_layer)
x17 = Lambda(slice,arguments={'index':17})(input_layer)
x18 = Lambda(slice,arguments={'index':18})(input_layer)
x19 = Lambda(slice,arguments={'index':19})(input_layer)

x0 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x0)
x1 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x1)
x2 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x2)
x3 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x3)
x4 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x4)
x5 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x5)
x6 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x6)
x7 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x7)
x8 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x8)
x9 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x9)
x10 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x10)
x11 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x11)
x12 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x12)
x13 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x13)
x14 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x14)
x15 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x15)
x16 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x16)
x17 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x17)
x18 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x18)
x19 = Bidirectional(LSTM(10, return_sequences = False, activation='tanh', kernel_initializer='he_normal'))(x19)

x = Concatenate(axis=-1)([x0,x1])
x = Concatenate(axis=-1)([x,x2])
x = Concatenate(axis=-1)([x,x3])
x = Concatenate(axis=-1)([x,x4])
x = Concatenate(axis=-1)([x,x5])
x = Concatenate(axis=-1)([x,x6])
x = Concatenate(axis=-1)([x,x7])
x = Concatenate(axis=-1)([x,x8])
x = Concatenate(axis=-1)([x,x9])
x = Concatenate(axis=-1)([x,x10])
x = Concatenate(axis=-1)([x,x11])
x = Concatenate(axis=-1)([x,x12])
x = Concatenate(axis=-1)([x,x13])
x = Concatenate(axis=-1)([x,x14])
x = Concatenate(axis=-1)([x,x15])
x = Concatenate(axis=-1)([x,x16])
x = Concatenate(axis=-1)([x,x17])
x = Concatenate(axis=-1)([x,x18])
x = Concatenate(axis=-1)([x,x19])

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

history = model.fit(X_train, Y_train, batch_size=64, epochs=3, verbose=1, validation_data = (X_test,Y_test), shuffle = True, callbacks=[checkpoint])
#model.save('LSTM_HoneyComb_L=3_G00_100step_10_lr=5e-3.h5')

# ========================== record data ========================
model.load_weights('lstm.h5')

pre = model.predict(X_test)
pre = pre.flatten()

np.save('./LSTM result/lstm_predict.npy', pre)
np.save('./LSTM result/lstm_T.npy', T_test)


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
plt.plot([3.69,3.69],[0,1],label='Tc = 3.69', color='red')
plt.xlabel('T/J')
plt.ylabel('Bi-LSTM output')
plt.title('Bi-LSTM')
plt.legend()      

plt.savefig('./LSTM result/figure.png', dpi=150)
plt.show()
plt.close()

tensorflow.keras.backend.clear_session()






