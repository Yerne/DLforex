# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:48:06 2017

@author: Gorowin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

train = pd.read_csv('investing.com_train_0.9.csv')

#export to csv
#Xy = pd.concat([X1, y1], axis=1, join='inner')
#Xy.to_csv('eur_Xy_train_0.1.csv')

X = train.iloc[:, 1:9]
y = train.iloc[:, 9:19]
import xgboost as xgb

X_all = X.values
y_eurusd = y.eurusd_w_delta
y_gbpusd = y.gbpusd_w_delta
y_audusd= y.audusd_w_delta
y_usdcad= y.usdcad_w_delta
y_usdjpy= y.usdjpy_w_delta
y_nzdusd= y.nzdusd_w_delta
y_gbpjpy= y.gbpjpy_w_delta
y_audnzd= y.audnzd_w_delta
y_eurgbp= y.eurgbp_w_delta
y_eurjpy= y.eurjpy_w_delta


dtrain_eur = xgb.DMatrix(X_all, y_eurusd)
dtrain_gbp = xgb.DMatrix(X_all, y_gbpusd)
dtrain_jpy = xgb.DMatrix(X_all, y_usdjpy)
dtrain_cad = xgb.DMatrix(X_all, y_usdcad)
dtrain_nzdusd = xgb.DMatrix(X_all, y_nzdusd)
dtrain_aud = xgb.DMatrix(X_all, y_audusd)
dtrain_gbpjpy = xgb.DMatrix(X_all, y_gbpjpy)
dtrain_audnzd = xgb.DMatrix(X_all, y_audnzd)
dtrain_eurgbp = xgb.DMatrix(X_all, y_eurgbp)
dtrain_eurjpy = xgb.DMatrix(X_all, y_eurjpy)

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
bst_gbpjpy = xgb.train(xgb_params, dtrain_gbpjpy, num_round)
bst_audnzd = xgb.train(xgb_params, dtrain_audnzd, num_round)
bst_eurgbp = xgb.train(xgb_params, dtrain_eurgbp, num_round)
bst_eurjpy = xgb.train(xgb_params, dtrain_eurjpy, num_round)


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
pred_gbpjpy = bst_gbpjpy.predict(y_pred)
pred_audnzd = bst_audnzd.predict(y_pred)
pred_eurgbp = bst_eurgbp.predict(y_pred)
pred_eurjpy = bst_eurjpy.predict(y_pred)
#save data to csv for knime
#X2.to_csv('eur_y_train_0.1.csv') 
df_inv=pd.DataFrame() 
df_inv['eur'] = pd.DataFrame(pred_eur)
df_inv['gbp'] = pd.DataFrame(pred_gbp)
df_inv['aud'] = pd.DataFrame(pred_aud)
df_inv['cad'] = pd.DataFrame(pred_cad)
df_inv['jpy'] = pd.DataFrame(pred_jpy)
df_inv['nzdusd'] = pd.DataFrame(pred_nzdusd)
df_inv['gbpjpy'] = pd.DataFrame(pred_gbpjpy)
df_inv['audnzd'] = pd.DataFrame(pred_audnzd)
df_inv['eurgbp'] = pd.DataFrame(pred_eurgbp)
df_inv['eurjpy'] = pd.DataFrame(pred_eurjpy)

df_inv.to_csv('pred_fx_inv.csv')

# feature importance 
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(8, 16))
xgb.plot_importance(bst_jpy, max_num_features=50, height=0.5, ax=ax)

import gplearn 

