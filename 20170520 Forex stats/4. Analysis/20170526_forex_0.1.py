# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:48:06 2017

@author: Gorowin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

eurusd_d = pd.read_csv('eurusd_d.csv')
usdjpy_d = pd.read_csv('usdjpy_d.csv')
de_b = pd.read_csv('10dey_b_d.csv') 
us_b = pd.read_csv('10usy_b_d.csv')
jp_b = pd.read_csv('10jpy_b_d.csv')
cn_b = pd.read_csv('10cny_b_d.csv')

eurusd_d['Date'] = pd.to_datetime(eurusd_d['Date'])
usdjpy_d['Date'] = pd.to_datetime(usdjpy_d['Date'])
de_b['Date'] = pd.to_datetime(de_b['Date'])
us_b['Date'] = pd.to_datetime(us_b['Date'])
jp_b['Date'] = pd.to_datetime(jp_b['Date'])
cn_b['Date'] = pd.to_datetime(cn_b['Date'])

eurusd_d.index = eurusd_d['Date']
del eurusd_d['Date']
de_b.index = de_b['Date']
del de_b['Date']
jp_b.index = jp_b['Date']
del jp_b['Date']
cn_b.index = cn_b['Date']
del cn_b['Date']
us_b.index = us_b['Date']
del us_b['Date']
usdjpy_d.index = usdjpy_d['Date']
del usdjpy_d['Date']


eurusd_d['Open_Close_eurusd'] = (eurusd_d['Open'] - eurusd_d['Close']).astype(float)
usdjpy_d['Open_Close_usdjpy'] = (usdjpy_d['Open'] - usdjpy_d['Close']).astype(float)
de_b['Open_Close_de_b'] = (de_b['Open'] - de_b['Close']).astype(float)
us_b['Open_Close_us_b'] = (us_b['Open'] - us_b['Close']).astype(float)
jp_b['Open_Close_jp_b'] = (jp_b['Open'] - jp_b['Close']).astype(float)
cn_b['Open_Close_cn_b'] = (cn_b['Open'] - cn_b['Close']).astype(float)

dset = pd.concat([eurusd_d, usdjpy_d['Open_Close_usdjpy']], axis=1, join='inner')
dset = pd.concat([dset, de_b['Open_Close_de_b']], axis=1, join='inner')
dset = pd.concat([dset, us_b['Open_Close_us_b']], axis=1, join='inner')
dset = pd.concat([dset, jp_b['Open_Close_jp_b']], axis=1, join='inner')
dset = pd.concat([dset, cn_b['Open_Close_cn_b']], axis=1, join='inner')

#dset = dset.rename(index=str, columns={"Open_Close": "delta_eur", "Open_Close1": "delta_jpy", "Open_Close2": "delta_de_b", "Open_Close3": "delta_us_b", "Open_Close4": "delta_jp_b", "Open_Close5": "delta_cn_b"})

X = dset.iloc[:, 5:9]
y = dset.iloc[:, 4:5]


y1= y.shift(periods=1)
y1=y1.drop(y1.index[0])
X1 = X.drop(X.index[0])

import xgboost as xgb

X_all = X1.values
y_all = y1.values


dtrain = xgb.DMatrix(X_all, label=y_all)
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
num_round = 2
bst = xgb.train(xgb_params, dtrain, num_round) 

#make xgb matrix for prediction

de_b_pred = pd.read_csv('10dey_b_d_pred_0.1.csv') 
us_b_pred = pd.read_csv('10usy_b_d_pred_0.1.csv')
jp_b_pred = pd.read_csv('10jpy_b_d_pred_0.1.csv')
usdjpy_d_pred = pd.read_csv('usdjpy_d_pred_0.1.csv')
de_b_pred['Date'] = pd.to_datetime(de_b_pred['Date'])
us_b_pred['Date'] = pd.to_datetime(us_b_pred['Date'])
jp_b_pred['Date'] = pd.to_datetime(jp_b_pred['Date'])
usdjpy_d_pred['Date'] = pd.to_datetime(usdjpy_d_pred ['Date'])
usdjpy_d_pred.index = usdjpy_d_pred['Date']
del usdjpy_d_pred['Date']
de_b_pred.index = de_b_pred['Date']
del de_b_pred['Date']
jp_b_pred.index = jp_b_pred['Date']
del jp_b_pred['Date']
us_b_pred.index = us_b_pred['Date']
del us_b_pred['Date']
usdjpy_d_pred['Open_Close_usdjpy'] = (usdjpy_d_pred['Open'] - usdjpy_d_pred['Close']).astype(float)
de_b_pred['Open_Close_de_b'] = (de_b_pred['Open'] - de_b_pred['Close']).astype(float)
us_b_pred['Open_Close_us_b'] = (us_b_pred['Open'] - us_b_pred['Close']).astype(float)
jp_b_pred['Open_Close_jp_b'] = (jp_b_pred['Open'] - jp_b_pred['Close']).astype(float)
pred_set = pd.concat([jp_b_pred, de_b_pred['Open_Close_de_b']], axis=1, join='inner')
pred_set = pd.concat([pred_set, us_b['Open_Close_us_b']], axis=1, join='inner')
pred_set = pd.concat([pred_set, usdjpy_d_pred['Open_Close_usdjpy']], axis=1, join='inner')


X2 = pred_set.iloc[:, 4:8]


y_pred = xgb.DMatrix(X2.values)

# make prediction


preds = bst.predict(y_pred)
























