'''
2026-09-29

https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

x = (n, 144, 13)
y = (n, 144, 1)

2016-12-31 00:00~ 2017-01-01 00:00의 wd(풍향)을 맞춰보자
'''
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datapath = './_data/kaggle_jena/'

data = pd.read_csv(datapath + 'jena_climate_2009_2016.csv')

# print('data')
# print(data)         # [420551 rows x 15 columns]
# print('data.shape')
# print(data.shape)   # (420551, 15)
# print('data.columns')
# print(data.columns)
# Index(['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
#        'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')

data['Date Time'] = pd.to_datetime(
    data['Date Time'],
    format='%d.%m.%Y %H:%M:%S'
)

print(data.head()) # [5 rows x 15 columns]
print(data.tail()) # [5 rows x 15 columns]
