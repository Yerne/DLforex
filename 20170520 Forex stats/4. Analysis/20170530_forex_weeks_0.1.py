# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:48:06 2017

@author: Gorowin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime



eurusd_w = pd.read_csv('eurusd_w.csv')
gbpusd_w = pd.read_csv('gbpusd_w.csv')
usdjpy_w = pd.read_csv('usdjpy_w.csv')
usdcad_w = pd.read_csv('usdcad_w.csv')
audusd_w = pd.read_csv('audusd_w.csv')
gbpjpy_w = pd.read_csv('gbpjpy_w.csv')
audnzd_w = pd.read_csv('audnzd_w.csv')

de_b_w = pd.read_csv('10dey_b_w.csv') 
us_b_w = pd.read_csv('10usy_b_w.csv')
jp_b_w = pd.read_csv('10jpy_b_w.csv')
ca_b_w = pd.read_csv('10cay_b_w.csv')
uk_b_w = pd.read_csv('10uky_b_w.csv')
au_b_w = pd.read_csv('10auy_b_w.csv')
cn_b_w = pd.read_csv('10cny_b_w.csv')

#make "dates" data type if some of data types in files mismatch other data type
eurusd_w['Date'] = pd.to_datetime(eurusd_w['Date'])
gbpusd_w['Date'] = pd.to_datetime(gbpusd_w['Date'])
usdjpy_w['Date'] = pd.to_datetime(usdjpy_w['Date'])
usdcad_w['Date'] = pd.to_datetime(usdcad_w['Date'])
audusd_w['Date'] = pd.to_datetime(audusd_w['Date'])
gbpjpy_w['Date'] = pd.to_datetime(gbpjpy_w['Date'])
audnzd_w['Date'] = pd.to_datetime(audnzd_w['Date'])

de_b_w['Date'] = pd.to_datetime(de_b_w['Date'])
us_b_w['Date'] = pd.to_datetime(us_b_w['Date'])
au_b_w['Date'] = pd.to_datetime(au_b_w['Date'])
jp_b_w['Date'] = pd.to_datetime(jp_b_w['Date'])
ca_b_w['Date'] = pd.to_datetime(ca_b_w['Date'])
uk_b_w['Date'] = pd.to_datetime(uk_b_w['Date'])
cn_b_w['Date'] = pd.to_datetime(cn_b_w['Date'])


eurusd_w.index = eurusd_w['Date']
del eurusd_w['Date'] 
gbpusd_w.index = gbpusd_w['Date']
del gbpusd_w['Date'] 
usdjpy_w.index = usdjpy_w['Date']
del usdjpy_w['Date'] 
audusd_w.index = audusd_w['Date']
del audusd_w['Date'] 
gbpjpy_w.index = gbpjpy_w['Date']
del gbpjpy_w['Date'] 
audnzd_w.index = audnzd_w['Date']
del audnzd_w['Date'] 
usdcad_w.index = usdcad_w['Date']
del usdcad_w['Date'] 

de_b_w.index = de_b_w['Date']
del de_b_w['Date'] 
us_b_w.index = us_b_w['Date']
del us_b_w['Date'] 
au_b_w.index = au_b_w['Date']
del au_b_w['Date'] 
jp_b_w.index = jp_b_w['Date']
del jp_b_w['Date'] 
ca_b_w.index = ca_b_w['Date']
del ca_b_w['Date'] 
uk_b_w.index = uk_b_w['Date']
del uk_b_w['Date'] 
cn_b_w.index = cn_b_w['Date']
del cn_b_w['Date'] 




eurusd_w['O_C_eurusd'] = (eurusd_w['Open'] - eurusd_w['Close']).astype(float)
gbpusd_w['O_C_gbpusd'] = (gbpusd_w['Open'] - gbpusd_w['Close']).astype(float)
audusd_w['O_C_audusd'] = (audusd_w['Open'] - audusd_w['Close']).astype(float)
usdjpy_w['O_C_usdjpy'] = (usdjpy_w['Open'] - usdjpy_w['Close']).astype(float)
gbpjpy_w['O_C_gbpjpy'] = (gbpjpy_w['Open'] - gbpjpy_w['Close']).astype(float)
usdcad_w['O_C_usdcad'] = (usdcad_w['Open'] - usdcad_w['Close']).astype(float)
audnzd_w['O_C_audnzd'] = (audnzd_w['Open'] - audnzd_w['Close']).astype(float)
# 
de_b_w['Open_Close_de_b_w'] = (de_b_w['Open'] - de_b_w['Close']).astype(float)
us_b_w['Open_Close_us_b_w'] = (us_b_w['Open'] - us_b_w['Close']).astype(float)
jp_b_w['Open_Close_jp_b_w'] = (jp_b_w['Open'] - jp_b_w['Close']).astype(float)
ca_b_w['Open_Close_ca_b_w'] = (ca_b_w['Open'] - ca_b_w['Close']).astype(float)
uk_b_w['Open_Close_uk_b_w'] = (uk_b_w['Open'] - uk_b_w['Close']).astype(float)
au_b_w['Open_Close_au_b_w'] = (au_b_w['Open'] - au_b_w['Close']).astype(float)
cn_b_w['Open_Close_cn_b_w'] = (cn_b_w['Open'] - cn_b_w['Close']).astype(float)



dset = pd.concat([audusd_w['O_C_audusd'], eurusd_w['O_C_eurusd']], axis=1, join='inner')
dset = pd.concat([dset,  gbpusd_w['O_C_gbpusd']], axis=1, join='inner')
dset = pd.concat([dset, usdjpy_w['O_C_usdjpy']], axis=1, join='inner')
dset = pd.concat([dset, usdcad_w['O_C_usdcad']], axis=1, join='inner')
dset = pd.concat([dset, gbpjpy_w['O_C_gbpjpy']], axis=1, join='inner')
dset = pd.concat([dset, audnzd_w['O_C_audnzd']], axis=1, join='inner')

dset = pd.concat([dset, uk_b_w['Open_Close_uk_b_w']], axis=1, join='inner')
dset = pd.concat([dset, au_b_w['Open_Close_au_b_w']], axis=1, join='inner')
dset = pd.concat([dset, jp_b_w['Open_Close_jp_b_w']], axis=1, join='inner')
dset = pd.concat([dset, us_b_w['Open_Close_us_b_w']], axis=1, join='inner')
dset = pd.concat([dset, cn_b_w['Open_Close_cn_b_w']], axis=1, join='inner')
dset = pd.concat([dset, de_b_w['Open_Close_de_b_w']], axis=1, join='inner')
dset = pd.concat([dset, ca_b_w['Open_Close_ca_b_w']], axis=1, join='inner')
 


#dset = dset.rename(index=str, columns={"Open_Close": "delta_eur", "Open_Close1": "delta_jpy", "Open_Close2": "delta_de_b", "Open_Close3": "delta_us_b", "Open_Close4": "delta_jp_b", "Open_Close5": "delta_cn_b"})

X = dset.iloc[:, 7:14]
#X = pd.concat([X, dset['day_of_week']], axis=1, join='inner')

y = dset.iloc[:, 0:7]
#Shift y1 data to +1 and drop 

X1=X.shift(periods=1)
X1 = X1.drop(X1.index[0])
y1=y.drop(y.index[0])

#export to csv
#Xy = pd.concat([X1, y1], axis=1, join='inner')
#Xy.to_csv('eur_Xy_train_0.1.csv')


import xgboost as xgb

X_all = X1.values
y_eurusd = y1.O_C_eurusd
y_gbpusd = y1.O_C_gbpusd
y_audusd= y1.O_C_audusd
y_usdcad= y1.O_C_usdcad
y_usdjpy= y1.O_C_usdjpy
y_gbpjpy= y1.O_C_gbpjpy
y_audnzd= y1.O_C_audnzd



dtrain_eur = xgb.DMatrix(X_all, y_eurusd)
dtrain_gbp = xgb.DMatrix(X_all, y_gbpusd)
dtrain_jpy = xgb.DMatrix(X_all, y_usdjpy)
dtrain_cad = xgb.DMatrix(X_all, y_usdcad)
dtrain_gbpjpy = xgb.DMatrix(X_all, y_gbpjpy)
dtrain_audnzd = xgb.DMatrix(X_all, y_audnzd)
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
bst_gbpjpy = xgb.train(xgb_params, dtrain_gbpjpy, num_round)
bst_audnzd = xgb.train(xgb_params, dtrain_audnzd, num_round)
bst_aud = xgb.train(xgb_params, dtrain_aud, num_round) 
# compose test matrix
X2 = dset.iloc[-1:, 7:14]
y_pred = xgb.DMatrix(X2.values)
#feed model with test matrix to make prediction
pred_eur = bst_eur.predict(y_pred).astype(float)
pred_gbp = bst_gbp.predict(y_pred).astype(float)
pred_jpy = bst_jpy.predict(y_pred).astype(float)
pred_cad = bst_cad.predict(y_pred).astype(float)
pred_gbpjpy = bst_gbpjpy.predict(y_pred).astype(float)
pred_audnzd = bst_audnzd.predict(y_pred).astype(float)
pred_aud = bst_aud.predict(y_pred).astype(float)
#save data to csv for knime
#X2.to_csv('eur_y_train_0.1.csv')
df=pd.DataFrame() 
df['eur'] = pd.DataFrame(pred_eur)
df['gbp'] = pd.DataFrame(pred_gbp)
df['aud'] = pd.DataFrame(pred_aud)
df['cad'] = pd.DataFrame(pred_cad)
df['jpy'] = pd.DataFrame(pred_jpy)
df['gbpjpy'] = pd.DataFrame(pred_gbpjpy)
df['audnzd'] = pd.DataFrame(pred_audnzd)



df.to_csv('pred_fx.csv')




--------------------------------------
def xgbt(yz):
    dtrain = xgb.DMatrix(X_all, yz),
    bst =0
    xgb_params = {
    'booster': 'gbtree',
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0,
    'alpha': 0.0001, 
    'lambda': 1},
    num_round = 100,
    bst = xgb.train(xgb_params, dtrain, num_round)     
     

xgbt(y_gbpusd)       

-----------------------------        


