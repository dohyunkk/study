'''
2026-09-29

https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

x = (n, 144, 13)
y = (n, 144, 1)

2016-12-31 00:00~ 2017-01-01 00:00의 T (degC)을 맞춰보자
'''
# import os
# os.environ["TF_GRU_ALLOCATOR"] = "cuda_malloc_async"    # 메모리 모으기

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

# 1. 데이터
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
# print(datasets.shape) # (420551, 14)

y_cor = datasets[-144:]['T (degC)'] # 예측치 정답 데이터
# print(y_cor.shape)      # (144,)


#### 훈련할 데이터 자르기 ####
start_time = time.time()

x_data = datasets[:-288].drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)
y_data = datasets[144:-144]['T (degC)'].to_numpy(dtype=np.float32)

print(x_data.shape)      # (420263, 13)
print(y_data.shape)      # (420263),)

size_x = 144
size_y = 144

def split_x(dataset, size):                  
    aaa = []                                 
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]       
        aaa.append(subset)                   
    return np.array(aaa)

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

end_time = time.time()
#### 자르기 끝 ####

print(x.shape, y.shape)
print("자르는시간:", round(end_time - start_time, 2),'sec')


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    shuffle = False,
)

scaler = StandardScaler()