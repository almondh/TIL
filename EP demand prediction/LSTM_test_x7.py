# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:34:33 2019

@author: huiyeon
"""

# LSTM 'X7'로 돌려보기

# import package
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt


# 1. load data ----------------------------
# 보간법으로 채운 test data
test = pd.read_csv("C:/Users/huiyeon/Desktop/Dacon/보간법/test_interpolation.csv")
test['Time'] = pd.to_datetime(test['Time'])
test = test.set_index('Time')

test['X7'].isnull().sum()  # 결측치 7003개

# X7 한 세대에 대해서 dropna(axis=0)
test_X7 = pd.DataFrame(test['X7'].dropna(axis=0))
test_X7.info()
test_X7.plot()


# 2. Data split ----------------------------
split_n = round(len(test_X7)*0.7)  # 5208번째까지 train

train_1 = test_X7.iloc[:split_n]
test_1 = test_X7.iloc[split_n:]

ax = train_1.plot()
test_1.plot(ax=ax)
plt.legend(['train', 'test'])

# 3. Data fit, transform, scailing --------------------
# 분포 추정, 변환, 스케일링
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

train_sc = scaler.fit_transform(train_1)  # fit + transform
test_sc = scaler.transform(test_1)  # scailing

print("train_sc :\n", train_sc)
print("test_sc :\n", test_sc)

# 사용하기 편리하도록 df로 변환
train_sc_df = pd.DataFrame(train_sc, columns=['X7'], index=train_1.index)
test_sc_df = pd.DataFrame(test_sc, columns=['X7'], index=test_1.index)
train_sc_df.head()


# sliding window 구성하기
for s in range(1, 13):  # 과거 시간 단위 24개가 훈련속성 
    train_sc_df['shift_{}'.format(s)] = train_sc_df['X7'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['X7'].shift(s)

train_sc_df.head()


# 4. Make train test set
train_sc_df.shape  # (1230, 13)

X_train = train_sc_df.dropna().drop('X7', axis=1)
y_train = train_sc_df.dropna()[['X7']]

X_test = test_sc_df.dropna().drop('X7', axis=1)
y_test = test_sc_df.dropna()[['X7']]

X_train.head()
y_train.head()


# df -> ndarray로 변환 
print("X_train type :", type(X_train))
print("y_train type :", type(y_train))
print("X_test type :", type(X_test))
print("y_test type :", type(y_test)) # 전부 df

# ndarray로 변환 
X_train = X_train.values
X_test= X_test.values
y_train = y_train.values
y_test = y_test.values

print(X_train.shape)
print(y_train.shape)


# 1757 데이터와 1218의 slot을가지고 있으므로
# 1218, 1로 reshape
X_train_t = X_train.reshape(X_train.shape[0], 12, 1)
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)

print("최종 DATA")
print(X_train_t.shape)
print(X_train_t)
print(y_train)


# 4. LSTM 모델 생성
# LSTM 모델 생성
from keras.layers import LSTM 
from keras.models import Sequential 
from keras.layers import Dense 
import keras.backend as K 
from keras.callbacks import EarlyStopping

K.clear_session()
    
model = Sequential() # Sequeatial Model 
model.add(LSTM(20, input_shape=(12, 1))) # (timestep, feature) 
model.add(Dense(1)) # output = 1 

# loss 함수:  MAPE
model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
model.summary()

# model train
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train_t, y_train, epochs=100,
          batch_size=30, verbose=1, callbacks=[early_stop])

score = model.evaluate(X_test_t, y_test, batch_size=30)
print(score)  # score값: 90.130


y_pred = model.predict(X_test_t, batch_size=32)
plt.scatter(y_test, y_pred)


############
# LSTM 손실함수 조절해서 다시 돌려보기
K.clear_session()
    
model = Sequential() # Sequeatial Model 
model.add(LSTM(20, input_shape=(12, 1))) # (timestep, feature) 
model.add(Dense(1)) # output = 1 

# loss 함수:  MAPE
model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
model.summary()

# model train
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train_t, y_train, epochs=100,
          batch_size=30, verbose=1, callbacks=[early_stop])

score = model.evaluate(X_test_t, y_test, batch_size=30)
print(score)  # score값: 90.130


y_pred = model.predict(X_test_t, batch_size=32)
plt.scatter(y_test, y_pred)