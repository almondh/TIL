# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:28:48 2019

NA interpolation Method 1 - test data

@author: huiyeon
"""

# 1. import package ------------------------------
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 

from datetime import datetime, timedelta
from sklearn.cluster import KMeans

%matplotlib inline


# 2. load data -----------------------------------
test = pd.read_csv("C:/Users/huiyeon/Desktop/Dacon/11회 data/test.csv")
test['Time'] = pd.to_datetime(test['Time'])
#train = train.set_index('Time')

test_ex = test.copy()  # 원본 데이터 보존을 위해 복사
test_ex = test_ex.set_index('Time')
test_ex.shape  # (8760, 201)

# 모든 행이 NaN인 세대 수  없음
#NaN_list = set(test_ex.dropna(how='all', axis=1).keys())
#set(test_ex.keys()).difference(NaN_list)
#
#NaN_X = set(test_ex.keys()).difference(NaN_list)
#NaN_X

# 결측치가 아닌 처음 수
first_notnull = []  # 빈 list

for i in test_ex.keys()[1:]:
    _ = test_ex[i]
    first_notnull.append(_[_.isnull() == False].index[0])
        
df_notnull = pd.DataFrame(first_notnull, list(test_ex.keys()[1:]))
df_notnull.columns = ['not_blank_num']


# 3. imputation for test data ----------------------
### Method 1
# 전체 Time 데이터
Raw_Time_df = test[['Time']]

# Time변수만 df로 추출
Time_df = test_ex[:1]

Raw_Time_df = Raw_Time_df.set_index('Time')
result = Raw_Time_df

for k in test_ex.keys()[1:]:
    # randomhouse에 해당하는 k세대만큼 for문 동작
    start_num = df_notnull.loc[k][0]
    tsvar = test_ex[[k]][start_num:]
    
    # Time 인덱스를 column값으로 보내기
    tsvar.reset_index(level=['Time'], inplace=True)
    
    ts_inter_1 = tsvar.interpolate()  # method 1
    ts_inter_1 = ts_inter_1.set_index(tsvar.index)
    result = pd.merge(result, ts_inter_1, how='left',
                      on='Time')
    
# 4. To csv ----------------------------------------
result.to_csv('test_interpolation.csv', index=False)

