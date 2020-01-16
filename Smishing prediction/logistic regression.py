# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:23:16 2020

@author: huiyeon

"""
##################################
# 로지스틱 회귀로 문서 분류해보기 #
#################################

# import package ------------------------------
import pandas as pd
import numpy as np


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

# modeling --------------------------------------
# split X, y
n = round(len(train)*0.8)  # train 시킬 데이터 길이
X_train = train.loc[:n, 'after_spacing'].values
y_train = train.loc[:n, 'smishing'].values
X_test = train.loc[n:, 'after_spacing'].values
y_test = train.loc[n:, 'smishing'].values

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(solver='liblinear', random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)
# 3:43 ~ 

print('최적의 매개변수 조합: %s ' % gs_lr_tfidf.best_params_)
# 최적의 매개변수 조합: {'clf__C': 100.0, 'clf__penalty': 'l2', 'vect__ngram_range': (1, 1)} 
print('CV 정확도: %.3f' % gs_lr_tfidf.best_score_)  # CV 정확도: 0.999
# predict
model = gs_lr_tfidf.best_estimator_
pred = model.predict(submission['after_spacing']) 


submission.head()
submission['after_spacing']
pred_df = pd.DataFrame(pred)
submission['smishing'] = pred_df
submission['smishing'].value_counts()

submission.head()

submission_reg = submission[['id', 'smishing']]
submission_reg.loc[1625] = [340486, 0]
submission_reg.sort_values(['id'], inplace=True)
submission_reg.to_csv('submission_reg.csv', index=False)


# 확률값 추출해보기
## submission_1로 다시 
submission_1 = public[['id', 'after_spacing']]
submission_1.head()

# null값 제거
submission_1.dropna(axis=0, inplace=True)
submission_1.reset_index(inplace=True)
del submission_1['index']
np.sum(pd.isnull(submission_1)) 

# 확률
y_prob = gs_lr_tfidf.predict_proba(submission_1['after_spacing'])

prob_df = pd.DataFrame(y_prob)
submission_1['smishing'] = prob_df[1]

submission_reg_pb = submission_1[['id', 'smishing']]
submission_reg_pb.loc[1625] = [340486, 0]
submission_reg_pb.sort_values(['id'], inplace=True)
submission_reg_pb.to_csv('submission_reg_pb.csv', index=False)
# 0.924