# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 18:03:20 2020

@author: huiyeon
"""

# import packages
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# split X, y
X = train.iloc[:,4:]
y = train.iloc[:,0:4]

# modeling --------------------------------
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(units=640, input_dim=226, activation='softplus'))  
model.add(Dense(units=320, activation='softplus'))
model.add(Dense(units=320, activation='softplus'))
model.add(Dense(units=160, activation='softplus'))
model.add(Dense(units=160, activation='softplus'))
model.add(Dense(units=4, activation='linear'))

# compile
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
callback_list = [keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                 monitor='val_loss',
                                                save_best_only=True),
                keras.callbacks.EarlyStopping(patience=100)]

# X, y 전부 넣고 검증만 해보기
# train
hist = model.fit(X, y, epochs=150, batch_size=120, 
                 validation_split = 0.05)  # 9:31

# histroy 
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['mae'], 'b', label='train mae')
acc_ax.plot(hist.history['val_mae'], 'g', label='val mae')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# evaluate
model.evaluate(X, y)  # 2.641

# predict
test_X = test.iloc[:,1:]
pred_test = model.predict(test_X)

# export submission
sample_sub = pd.read_csv('sample_submission.csv', index_col=0)
submission = sample_sub+pred_test
submission.to_csv('submission_keras_4.csv')

# save model
from keras.models import load_model
model.save('keras_4.h5')

# model summary
model = load_model('keras_4.h5')
model.summary()