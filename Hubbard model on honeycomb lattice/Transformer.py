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
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
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
# ========================== set env ========================

tensorflow.config.experimental_run_functions_eagerly(True)


# ========================== get data ========================
step = 10
# make training set
data_X = np.load('./data/train_G.npy')[:,:,:step] # the shape of raw x is (number of samples, number of series, number of step), we only take the first 10 step
data_Y = np.load('./data/train_U.npy')

X = []
Y = []
T = []
for i in trange(len(data_Y)):
    X.append(data_X[i])
    T.append(data_Y[i])
    if data_Y[i] >= 6:
        Y.append(1)
    else:
        Y.append(0)


X_train = np.array(X)
X_train = np.reshape(X_train, (X_train.shape[0],20,step,1))
Y_train = np.array(Y)
T_train = np.array(T)


# make testing set
data_X = np.load('./data/test_G.npy')[:,:,:step]
data_Y = np.load('./data/test_U.npy')

X = []
Y = []
T = []
for i in trange(len(data_Y)):
    X.append(data_X[i])
    T.append(data_Y[i])
    if data_Y[i] >= 3.9:
        Y.append(1)
    else:
        Y.append(0)

X_test = np.array(X)
X_test = np.reshape(X_test, (X_test.shape[0],20,step,1))
Y_test = np.array(Y)
T_test = np.array(T)
data_X = data_Y = data_x = data_y = X = Y = T = 0


print(X_train.shape)
print(X_test.shape)
        
        



def slice(x,index):
    return x[:,index,:]

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self): 
        config = {}
        base_config = super(TransformerBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        #self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        #maxlen = tf.shape(x)[-1]
        #positions = tf.range(start=0, limit=maxlen, delta=1)
        #positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x# + positions
    def get_config(self): 
        config = {}
        base_config = super(TokenAndPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
vocab_size = 10  # Only consider the top 20k words
maxlen = step  # Only consider the first 200 words of each movie review
embed_dim = 32  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

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

'''
x0 = embedding_layer(x0)
x1 = embedding_layer(x1)
x2 = embedding_layer(x2)
x3 = embedding_layer(x3)
x4 = embedding_layer(x4)
x5 = embedding_layer(x5)
x6 = embedding_layer(x6)
x7 = embedding_layer(x7)
x8 = embedding_layer(x8)
x9 = embedding_layer(x9)
x10 = embedding_layer(x10)
x11 = embedding_layer(x11)
x12 = embedding_layer(x12)
x13 = embedding_layer(x13)
x14 = embedding_layer(x14)
x15 = embedding_layer(x15)
x16 = embedding_layer(x16)
x17 = embedding_layer(x17)
x18 = embedding_layer(x18)
x19 = embedding_layer(x19)
'''

x0 = Dense(32, activation='relu')(x0)
x1 = Dense(32, activation='relu')(x1)
x2 = Dense(32, activation='relu')(x2)
x3 = Dense(32, activation='relu')(x3)
x4 = Dense(32, activation='relu')(x4)
x5 = Dense(32, activation='relu')(x5)
x6 = Dense(32, activation='relu')(x6)
x7 = Dense(32, activation='relu')(x7)
x8 = Dense(32, activation='relu')(x8)
x9 = Dense(32, activation='relu')(x9)
x10 = Dense(32, activation='relu')(x10)
x11 = Dense(32, activation='relu')(x11)
x12 = Dense(32, activation='relu')(x12)
x13 = Dense(32, activation='relu')(x13)
x14 = Dense(32, activation='relu')(x14)
x15 = Dense(32, activation='relu')(x15)
x16 = Dense(32, activation='relu')(x16)
x17 = Dense(32, activation='relu')(x17)
x18 = Dense(32, activation='relu')(x18)
x19 = Dense(32, activation='relu')(x19)

x0 = transformer_block(x0)
x1 = transformer_block(x1)
x2 = transformer_block(x2)
x3 = transformer_block(x3)
x4 = transformer_block(x4)
x5 = transformer_block(x5)
x6 = transformer_block(x6)
x7 = transformer_block(x7)
x8 = transformer_block(x8)
x9 = transformer_block(x9)
x10 = transformer_block(x10)
x11 = transformer_block(x11)
x12 = transformer_block(x12)
x13 = transformer_block(x13)
x14 = transformer_block(x14)
x15 = transformer_block(x15)
x16 = transformer_block(x16)
x17 = transformer_block(x17)
x18 = transformer_block(x18)
x19 = transformer_block(x19)

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
#x = Dropout(0.1)(x)
x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
#x = Dropout(0.1)(x)

output_layer = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)


model = tensorflow.keras.models.Model(input_layer, output_layer)
model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['acc'] )

checkpoint = ModelCheckpoint(filepath='trans.h5', monitor='val_acc', mode='auto' ,save_best_only='True')
# ========================== train model ========================

history = model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=1, validation_data = (X_test,Y_test), shuffle = True, callbacks=[checkpoint])


# ========================== recard data ========================
model.load_weights('trans.h5')


pre = model.predict(X_test)
pre = pre.flatten()

np.save('./Transformer result/trans_predict.npy', pre)
np.save('./Transformer result/trans_T.npy', T_test)

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



plt.figure(1)
plt.scatter(Y, P, color='blue',marker='x',s=30,label='predicted prob')
plt.plot([3.9,3.9],[0,1],label='Uc = 3.9', color='red')
plt.xlabel('T/J')
plt.ylabel('Trans output')
plt.title('Trans')
plt.legend()      

plt.savefig('./Transformer result/figure.png', dpi=150)
plt.show()
plt.clf()
plt.close()

tensorflow.keras.backend.clear_session()









