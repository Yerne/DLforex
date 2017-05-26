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



jp_b.head()
de_b.head()
us_b.head()
cn_b.head()
eurusd_d.head()
usdjpy_d.head()

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


















