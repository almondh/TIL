# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:20:47 2020

@author: huiyeon

"""

# import package -------------------------
# -*- coding: cp949 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plot 한글
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/KOPUBDOTUMMEDIUM.TTF").get_name()
rc('font', family=font_name)


# load data --------------------------------
data = pd.read_csv('after_spacing.csv')
data.head()
data.info()

# text 전처리되서 사라진 id
data[data['after_spacing'].isnull()==True]['id'] 
np.sum(pd.isnull(data))

null_data = data[data['after_spacing'].isnull()==True]


del data['Unnamed: 0']
cls_data = data[['id', 'after_spacing', 'smishing']]
cls_data.head()

# null값 제거
cls_data.dropna(axis=0, inplace=True)
cls_data.reset_index(inplace=True)
del cls_data['index']
np.sum(pd.isnull(cls_data)) 


# 1) test 해보기
test_data = cls_data.loc[:50]

from konlpy.tag import Okt 
okt=Okt()

def tokenize(doc):
    # norm, stem은 optional
    return [''.join(t) for t in okt.pos(doc, join=True)]

test_docs = [tokenize(test_data.loc[i][1]) for i in range(len(test_data))]
test_docs

# 확인
from pprint import pprint
pprint(test_docs[0])


# 2) cls_data 전체 돌리기
cls_docs = [tokenize(cls_data.loc[i][1]) for i in range(len(cls_data))]
pprint(cls_docs[0])


# 3) cls_data의 token 모으기
tokens=[]
for i in range(len(cls_docs)):
    for j in range(len(cls_docs[i])):
            tokens.append(cls_docs[i][j])

print(len(tokens))  # 10825019


# 4) nltk.Text() 써보기
import nltk
text = nltk.Text(tokens, name='NMSC')
print(text)  # <Text: NMSC>
print(len(text.tokens))  # 10825019
print(len(set(text.tokens)))  # 50796
pprint(text.vocab().most_common(10))
#[('고객/Noun', 232310),
# ('은행/Noun', 212920),
# ('님/Suffix', 206462),
# ('을/Josa', 201393),
# ('이/Josa', 158074),
# ('올림/Verb', 141004),
# ('에/Josa', 138664),
# ('의/Josa', 120203),
# ('지점/Noun', 105854),
# ('입니다/Adjective', 96213)]

# 5) plot 그려보기
text.plot(50)
