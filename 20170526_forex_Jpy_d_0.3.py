# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:48:06 2017

@author: Gorowin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

usdjpy_d = pd.read_csv('usdjpy_d.csv')
jp_b = pd.read_csv('10jpy_b_d.csv') 
us_b = pd.read_csv('10usy_b_d.csv')

usdjpy_d['Date'] = pd.to_datetime(usdjpy_d['Date'])
jp_b['Date'] = pd.to_datetime(jp_b['Date'])
us_b['Date'] = pd.to_datetime(us_b['Date'])


usdjpy_d['day_of_week'] = usdjpy_d['Date'].dt.dayofweek
jp_b['day_of_week'] = jp_b['Date'].dt.dayofweek
us_b['day_of_week'] = us_b['Date'].dt.dayofweek

usdjpy_d.index = usdjpy_d['Date']
del usdjpy_d['Date']
jp_b.index = jp_b['Date']
del jp_b['Date']
us_b.index = us_b['Date']
del us_b['Date']

usdjpy_d['Open_Close_usdjpy'] = (usdjpy_d['Open'] - usdjpy_d['Close']).astype(float)
jp_b['Open_Close_jp_b'] = (jp_b['Open'] - jp_b['Close']).astype(float)
us_b['Open_Close_us_b'] = (us_b['Open'] - us_b['Close']).astype(float)

dset = pd.concat([usdjpy_d, jp_b['Open_Close_jp_b']], axis=1, join='inner')
dset = pd.concat([dset, us_b['Open_Close_us_b']], axis=1, join='inner')


#dset = dset.rename(index=str, columns={"Open_Close": "delta_eur", "Open_Close1": "delta_jpy", "Open_Close2": "delta_jp_b", "Open_Close3": "delta_us_b", "Open_Close4": "delta_jp_b", "Open_Close5": "delta_cn_b"})

X = dset.iloc[:, 6:9]
X = pd.concat([X, dset['day_of_week']], axis=1, join='inner')

y = dset.iloc[:, 5:6]
#Shift y1 data to +1 and drop 

X1=X.shift(periods=1)
X1 = X1.drop(X1.index[0])
y1=y.drop(y.index[0])

#export to csv
#Xy = pd.concat([X1, y1], axis=1, join='inner')
#Xy.to_csv('eur_Xy_train_0.1.csv')


import xgboost as xgb

X_all = X1.values
y_all = y1.values


dtrain = xgb.DMatrix(X_all, y_all)
xgb_params = {
    'booster': 'gbtree',
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0,
    'alpha': 0.0001, 
    'lambda': 1
}
num_round = 100
bst = xgb.train(xgb_params, dtrain, num_round) 

#make xgb matrix for prediction
usdjpy_pred = pd.read_csv('usdjpy_d_pred_0.1.csv')
jp_b_pred = pd.read_csv('10dey_b_d_pred_0.1.csv') 
us_b_pred = pd.read_csv('10usy_b_d_pred_0.1.csv')
usdjpy_pred['Date'] = pd.to_datetime(usdjpy_pred['Date'])
jp_b_pred['Date'] = pd.to_datetime(jp_b_pred['Date'])
us_b_pred['Date'] = pd.to_datetime(us_b_pred['Date'])

usdjpy_pred['day_of_week'] = usdjpy_pred['Date'].dt.dayofweek
jp_b_pred['day_of_week'] = jp_b_pred['Date'].dt.dayofweek
us_b_pred['day_of_week'] = us_b_pred['Date'].dt.dayofweek

usdjpy_pred.index = usdjpy_pred['Date']
del usdjpy_pred['Date']
jp_b_pred.index = jp_b_pred['Date']
del jp_b_pred['Date']
us_b_pred.index = us_b_pred['Date']
del us_b_pred['Date']
usdjpy_pred['Open_Close_usdjpy_d'] = (usdjpy_pred['Open'] - usdjpy_pred['Close']).astype(float)
jp_b_pred['Open_Close_jp_b'] = (jp_b_pred['Open'] - jp_b_pred['Close']).astype(float)
us_b_pred['Open_Close_us_b'] = (us_b_pred['Open'] - us_b_pred['Close']).astype(float)
pred_set = pd.concat([jp_b_pred, usdjpy_pred['Open_Close_usdjpy_d']], axis=1, join='inner')
pred_set = pd.concat([pred_set, us_b_pred['Open_Close_us_b']], axis=1, join='inner')
del pred_set['day_of_week']
pred_set = pd.concat([pred_set, us_b_pred['day_of_week']], axis=1, join='inner')

del pred_set['Open']
del pred_set['High']
del pred_set['Low']
del pred_set['Close']
del pred_set['Open_Close_usdjpy_d']


X2 = pred_set.iloc[-1:, 0:3]


y_pred = xgb.DMatrix(X2.values)

# make prediction


preds = bst.predict(y_pred)
#save data to csv for knime


#X2.to_csv('eur_y_train_0.1.csv') 


















