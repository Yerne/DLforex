# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:48:06 2017

@author: Gorowin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


eurusd_d = pd.read_csv('eurusd_d.csv')
usdjpy_d = pd.read_csv('usdjpy_d.csv')
de_b = pd.read_csv('10dey_b_d.csv') 
us_b = pd.read_csv('10jpy_b_d.csv')
jp_b = pd.read_csv('10cny_b_d.csv')
cn_b = pd.read_csv('10usy_b_d.csv')

jp_b.head()
de_b.head()
us_b.head()
cn_b.head()
eurusd.head()
usdjpy.head()

eurusd_d['open_close'] = eurusd_d.open.fillna(eurusd_d.open)
eurusd_d[['open_close', 'Open']].sub()

eurusd_d[['Open_Close', 'Open']].sub(eurusd_d['Close'], axis=0)

eurusd_d['Open_Close'] = eurusd_d[['Open', 'Close']].sum(axis=1)