# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:48:06 2017

@author: Gorowin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

audusd_d = pd.read_csv('audusd_d.csv')
au_b = pd.read_csv('10auy_b_d.csv') 
us_b = pd.read_csv('10usy_b_d.csv')

audusd_d['Date'] = pd.to_datetime(audusd_d['Date'])
au_b['Date'] = pd.to_datetime(au_b['Date'])
us_b['Date'] = pd.to_datetime(us_b['Date'])


audusd_d['day_of_week'] = audusd_d['Date'].dt.dayofweek
au_b['day_of_week'] = au_b['Date'].dt.dayofweek
us_b['day_of_week'] = us_b['Date'].dt.dayofweek

audusd_d.index = audusd_d['Date']
del audusd_d['Date']
au_b.index = au_b['Date']
del au_b['Date']
us_b.index = us_b['Date']
del us_b['Date']
   
audusd_d['Open_Close_eurusd'] = (audusd_d['Open'] - audusd_d['Close']).astype(float)
au_b['Open_Close_au_b'] = (au_b['Open'] - au_b['Close']).astype(float)
us_b['Open_Close_us_b'] = (us_b['Open'] - us_b['Close']).astype(float)

dset = pd.concat([audusd_d, au_b['Open_Close_au_b']], axis=1, join='inner')
dset = pd.concat([dset, us_b['Open_Close_us_b']], axis=1, join='inner')


#dset = dset.rename(index=str, columns={"Open_Close": "delta_eur", "Open_Close1": "delta_jpy", "Open_Close2": "delta_au_b", "Open_Close3": "delta_us_b", "Open_Close4": "delta_jp_b", "Open_Close5": "delta_cn_b"})

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
audusd_pred = pd.read_csv('audusd_d_pred_0.1.csv')
au_b_pred = pd.read_csv('10auy_b_d_pred_0.1.csv') 
us_b_pred = pd.read_csv('10usy_b_d_pred_0.1.csv')
audusd_pred['Date'] = pd.to_datetime(audusd_pred['Date'])
au_b_pred['Date'] = pd.to_datetime(au_b_pred['Date'])
us_b_pred['Date'] = pd.to_datetime(us_b_pred['Date'])

audusd_pred['day_of_week'] = audusd_pred['Date'].dt.dayofweek
au_b_pred['day_of_week'] = au_b_pred['Date'].dt.dayofweek
us_b_pred['day_of_week'] = us_b_pred['Date'].dt.dayofweek

audusd_pred.index = audusd_pred['Date']
del audusd_pred['Date']
au_b_pred.index = au_b_pred['Date']
del au_b_pred['Date']
us_b_pred.index = us_b_pred['Date']
del us_b_pred['Date']
audusd_pred['Open_Close_audusd_d'] = (audusd_pred['Open'] - audusd_pred['Close']).astype(float)
au_b_pred['Open_Close_au_b'] = (au_b_pred['Open'] - au_b_pred['Close']).astype(float)
us_b_pred['Open_Close_us_b'] = (us_b_pred['Open'] - us_b_pred['Close']).astype(float)
pred_set = pd.concat([au_b_pred, audusd_pred['Open_Close_audusd_d']], axis=1, join='inner')
pred_set = pd.concat([pred_set, us_b_pred['Open_Close_us_b']], axis=1, join='inner')
del pred_set['day_of_week']
pred_set = pd.concat([pred_set, us_b_pred['day_of_week']], axis=1, join='inner')

del pred_set['Open']
del pred_set['High']
del pred_set['Low']
del pred_set['Close']
del pred_set['Open_Close_audusd_d']


X2 = pred_set.iloc[-1:, 0:3]


y_pred = xgb.DMatrix(X2.values)

# make prediction


preds = bst.predict(y_pred)
#save data to csv for knime


#X2.to_csv('eur_y_train_0.1.csv') 


















