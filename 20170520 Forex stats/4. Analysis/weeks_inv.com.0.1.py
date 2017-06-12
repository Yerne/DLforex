# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:48:06 2017

@author: Gorowin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

train = pd.read_csv('20170603_forex_deltas_0.8_Investing.com_train.csv')

#export to csv
#Xy = pd.concat([X1, y1], axis=1, join='inner')
#Xy.to_csv('eur_Xy_train_0.1.csv')

X = train.iloc[:, 1:9]
y = train.iloc[:, 9:15]
import xgboost as xgb

X_all = X.values
y_eurusd = y.eurusd_w_delta
y_gbpusd = y.gbpusd_w_delta
y_audusd= y.audusd_w_delta
y_usdcad= y.usdcad_w_delta
y_usdjpy= y.usdjpy_w_delta
y_nzdusd= y.nzdusd_w_delta


dtrain_eur = xgb.DMatrix(X_all, y_eurusd)
dtrain_gbp = xgb.DMatrix(X_all, y_gbpusd)
dtrain_jpy = xgb.DMatrix(X_all, y_usdjpy)
dtrain_cad = xgb.DMatrix(X_all, y_usdcad)
dtrain_nzdusd = xgb.DMatrix(X_all, y_nzdusd)
dtrain_aud = xgb.DMatrix(X_all, y_audusd)

xgb_params = {
    'booster': 'gbtree',
    'eta': 0.1,
    'max_depth': 5,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0,
    'alpha': 0.01, 
    'lambda': 1
}
num_round = 100
# make model
bst_eur = xgb.train(xgb_params, dtrain_eur, num_round)
bst_gbp = xgb.train(xgb_params, dtrain_gbp, num_round)
bst_jpy = xgb.train(xgb_params, dtrain_jpy, num_round)
bst_cad = xgb.train(xgb_params, dtrain_cad, num_round)
bst_nzdusd = xgb.train(xgb_params, dtrain_nzdusd, num_round)
bst_aud = xgb.train(xgb_params, dtrain_aud, num_round) 
# compose test matrix
X2 = train.iloc[-1:, 1:9]
y_pred = xgb.DMatrix(X2.values)
#feed model with test matrix to make prediction
pred_eur = bst_eur.predict(y_pred)
pred_gbp = bst_gbp.predict(y_pred)
pred_jpy = bst_jpy.predict(y_pred)
pred_cad = bst_cad.predict(y_pred)
pred_nzdusd = bst_nzdusd.predict(y_pred)
pred_aud = bst_aud.predict(y_pred)
#save data to csv for knime
#X2.to_csv('eur_y_train_0.1.csv') 
 
 
X1.to_csv('X1.csv') 
y1.to_csv('y1.csv') 





