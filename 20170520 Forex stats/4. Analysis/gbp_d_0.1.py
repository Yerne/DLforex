# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:48:06 2017

@author: Gorowin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

gbpusd_d = pd.read_csv('gbpusd_d.csv')
uk_b = pd.read_csv('10uky_b_d.csv') 
us_b = pd.read_csv('10usy_b_d.csv')

gbpusd_d['Date'] = pd.to_datetime(gbpusd_d['Date'])
uk_b['Date'] = pd.to_datetime(uk_b['Date'])
us_b['Date'] = pd.to_datetime(us_b['Date'])


gbpusd_d['day_of_week'] = gbpusd_d['Date'].dt.dayofweek
uk_b['day_of_week'] = uk_b['Date'].dt.dayofweek
us_b['day_of_week'] = us_b['Date'].dt.dayofweek

gbpusd_d.index = gbpusd_d['Date']
del gbpusd_d['Date']
uk_b.index = uk_b['Date']
del uk_b['Date']
us_b.index = us_b['Date']
del us_b['Date']
   
gbpusd_d['Open_Close_gbpusd'] = (gbpusd_d['Open'] - gbpusd_d['Close']).astype(float)
uk_b['Open_Close_uk_b'] = (uk_b['Open'] - uk_b['Close']).astype(float)
us_b['Open_Close_us_b'] = (us_b['Open'] - us_b['Close']).astype(float)

dset = pd.concat([gbpusd_d, uk_b['Open_Close_uk_b']], axis=1, join='inner')
dset = pd.concat([dset, us_b['Open_Close_us_b']], axis=1, join='inner')


#dset = dset.rename(index=str, columns={"Open_Close": "delta_eur", "Open_Close1": "delta_jpy", "Open_Close2": "delta_uk_b", "Open_Close3": "delta_us_b", "Open_Close4": "delta_jp_b", "Open_Close5": "delta_cn_b"})

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
gbpusd_pred = pd.read_csv('gbpusd_d_pred.csv')
uk_b_pred = pd.read_csv('10uky_b_d_pred_0.1.csv') 
us_b_pred = pd.read_csv('10usy_b_d_pred_0.1.csv')
gbpusd_pred['Date'] = pd.to_datetime(gbpusd_pred['Date'])
uk_b_pred['Date'] = pd.to_datetime(uk_b_pred['Date'])
us_b_pred['Date'] = pd.to_datetime(us_b_pred['Date'])

gbpusd_pred['day_of_week'] = gbpusd_pred['Date'].dt.dayofweek
uk_b_pred['day_of_week'] = uk_b_pred['Date'].dt.dayofweek
us_b_pred['day_of_week'] = us_b_pred['Date'].dt.dayofweek

gbpusd_pred.index = gbpusd_pred['Date']
del gbpusd_pred['Date']
uk_b_pred.index = uk_b_pred['Date']
del uk_b_pred['Date']
us_b_pred.index = us_b_pred['Date']
del us_b_pred['Date']
gbpusd_pred['Open_Close_gbpusd_d'] = (gbpusd_pred['Open'] - gbpusd_pred['Close']).astype(float)
uk_b_pred['Open_Close_uk_b'] = (uk_b_pred['Open'] - uk_b_pred['Close']).astype(float)
us_b_pred['Open_Close_us_b'] = (us_b_pred['Open'] - us_b_pred['Close']).astype(float)
pred_set = pd.concat([uk_b_pred, gbpusd_pred['Open_Close_gbpusd_d']], axis=1, join='inner')
pred_set = pd.concat([pred_set, us_b_pred['Open_Close_us_b']], axis=1, join='inner')
del pred_set['day_of_week']
pred_set = pd.concat([pred_set, us_b_pred['day_of_week']], axis=1, join='inner')

del pred_set['Open']
del pred_set['High']
del pred_set['Low']
del pred_set['Close']
del pred_set['Open_Close_gbpusd_d']


X2 = pred_set.iloc[-1:, 0:3]


y_pred = xgb.DMatrix(X2.values)

# make prediction


preds = bst.predict(y_pred)
#save data to csv for knime


#X2.to_csv('eur_y_train_0.1.csv') 


















