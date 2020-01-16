# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:11:50 2020

@author: huiyeon
"""
##################
# RNN 실행해보기 #
##################

# import package ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/KOPUBDOTUMMEDIUM.TTF").get_name()
rc('font', family=font_name)


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# load train data ------------------------------
data = pd.read_csv('after_spacing.csv')
data.head()

# text 전처리되서 사라진 id
data[data['after_spacing'].isnull()==True]['id'] 
np.sum(pd.isnull(data))

null_data = data[data['after_spacing'].isnull()==True]

del data['Unnamed: 0']
train = data[['id', 'after_spacing', 'smishing']]
train.head()

# null값 제거
train.dropna(axis=0, inplace=True)
train.reset_index(inplace=True)
del train['index']
np.sum(pd.isnull(train)) 


# load public data ------------------------------
public = pd.read_csv('after_spacing_public.csv')
public.head()

# text 전처리되서 사라진 id
public[public['after_spacing'].isnull()==True]['id'] 
np.sum(pd.isnull(public))

null_data_pub = public[public['after_spacing'].isnull()==True]

del public['Unnamed: 0']
submission = public[['id', 'after_spacing']]
submission.head()

# null값 제거
submission.dropna(axis=0, inplace=True)
submission.reset_index(inplace=True)
del submission['index']
np.sum(pd.isnull(submission)) 

# Tokenize + encoding ---------------------------
X_data = train['after_spacing']
y_data = train['smishing']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data)
sequences = tokenizer.texts_to_sequences(X_data) #단어를 숫자값, 인덱스로 변환하여 저장

# 단어에 부여된 인덱스 리턴
word_to_index = tokenizer.word_index
#print(word_to_index)

vocab_size = len(word_to_index)+1
print('단어 집합의 크기: {}'.format((vocab_size)))  # 135889

X_data = sequences
print('문자의 최대 길이 : %d' % max(len(l) for l in X_data))  # 85
print('문자의 평균 길이 : %f' % (sum(map(len, X_data))/len(X_data)))  #26.061404
plt.hist([len(s) for s in X_data], bins=50)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()

max_len = 85
data = pad_sequences(X_data, maxlen=max_len)
print("data shape: ", data.shape)
# (295941, 85)

# x, y split
n = round(len(train)*0.8)  # train 시킬 데이터 길이
X_train = data[:n]
y_train = np.array(y_data[:n])
X_test = data[n:]
y_test = np.array(y_data[n:])


# modeling --------------------------------------
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(vocab_size, 32)) # 임베딩 벡터의 차원은 32
model.add(SimpleRNN(32)) # RNN 셀의 hidden_size는 32
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
              metrics=['acc'])
history = model.fit(X_train, y_train, epochs=6, 
                    batch_size=64, validation_split=0.2)

# train, test accuracy 비교 시각화
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# loss, acc 시각화
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title("훈련 중 비용함수 그래프")
plt.ylabel("비용함수 값")
plt.subplot(1, 2, 2)
plt.title("훈련 중 성능지표 그래프")
plt.ylabel("성능지표 값")
plt.plot(history.history['accuracy'], 'b-', label="학습 성능")
plt.plot(history.history['val_accuracy'], 'r:', label="검증 성능")
plt.legend()
plt.tight_layout()
plt.show()


# predict prob ---------------------------------------
# endocing submission
submission.head()
pred_data = submission['after_spacing']

sequences = tokenizer.texts_to_sequences(pred_data)

pred_data = sequences

pred_data = pad_sequences(pred_data, maxlen=max_len)

prob = model.predict_proba(pred_data)

prob_df = pd.DataFrame(prob)
submission['smishing'] = prob_df

submission_rnn = submission[['id', 'smishing']]
submission_rnn.loc[1625] = [340486, 0]
submission_rnn.sort_values(['id'], inplace=True)
submission_rnn.to_csv('submission_rnn.csv', index=False)
