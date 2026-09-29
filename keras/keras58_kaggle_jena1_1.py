'''
2026-09-29

https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

x = (n, 144, 13)
y = (n, 144, 1)

2016-12-31 00:00~ 2017-01-01 00:00의 wd(풍향)을 맞춰보자
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

y_cor = datasets[-144:]['wd (deg)'] # 예측치 정답 데이터
# print(y_cor.shape)      # (144,)


#### 훈련할 데이터 자르기 ####
start_time = time.time()

x_data = datasets[:-288].drop(['wd (deg)'], axis=1)
y_data = datasets[144:-144]['wd (deg)']

print(x_data.shape)      # (420263, 13)
print(y_data.shape)      # (420263,)

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
# 자르는시간: 24.31 sec

# --------------------------------------------------------------





# ============================================================
# train / test 분리
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    shuffle=False,          # 시계열이므로 순서를 섞지 않음 # 앞에서 이미 잘라놨기 때문에 섞어도 됨.
)

print('x_train.shape :', x_train.shape) # (336096, 144, 13)
print('x_test.shape  :', x_test.shape)  # (84024, 144, 13)
print('y_train.shape :', y_train.shape) # (336096, 144)
print('y_test.shape  :', y_test.shape)  # (84024, 144)


# ============================================================
# 스케일링
# ============================================================

scaler_x = StandardScaler()
scaler_y = StandardScaler()

# -----------------------------
# x : (n, 144, 13)
# sklearn scaler는 2차원만 받을 수 있으므로
# (n*144, 13)으로 잠깐 변경
# -----------------------------

x_train_shape = x_train.shape
x_test_shape = x_test.shape

x_train = x_train.reshape(-1, x_train.shape[2])
x_test = x_test.reshape(-1, x_test.shape[2])

print('reshape x_train :', x_train.shape) # (48397824, 13)
print('reshape x_test  :', x_test.shape)  # (12099456, 13)


# train으로 fit + transform
x_train = scaler_x.fit_transform(x_train)

# test는 transform만
x_test = scaler_x.transform(x_test)


# 다시 LSTM용 3차원으로 복구
x_train = x_train.reshape(x_train_shape)
x_test = x_test.reshape(x_test_shape)

print('scale 후 x_train.shape :', x_train.shape)# (336096, 144, 13)
print('scale 후 x_test.shape  :', x_test.shape) # (84024, 144, 13)


# ============================================================
# y 스케일링
# ============================================================

# 현재 y.shape = (n, 144)
# 풍향 하나의 feature이므로 (-1, 1)로 변경해서 scaling

y_train_shape = y_train.shape
y_test_shape = y_test.shape

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


# 다시 원래 shape으로 복구
y_train = y_train.reshape(y_train_shape)
y_test = y_test.reshape(y_test_shape)

print('scale 후 y_train.shape :', y_train.shape) # (336096, 144)
print('scale 후 y_test.shape  :', y_test.shape)  # (84024, 144)


# ============================================================
# LSTM 출력용으로 y에 feature 차원 추가
# ============================================================

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

print('최종 x_train.shape :', x_train.shape) # (336096, 144, 13)
print('최종 x_test.shape  :', x_test.shape)  # (84024, 144, 13)
print('최종 y_train.shape :', y_train.shape) # (336096, 144, 1)
print('최종 y_test.shape  :', y_test.shape)  # (84024, 144, 1)


# np_path = './_data/save_kaggle_jena_npy/'
# np.save(np_path + 'keras58_01_x_train.npy', arr = x_train) # x_train
# np.save(np_path + 'keras58_01_y_train.npy', arr = y_train) # y_train
# np.save(np_path + 'keras58_01_x_test.npy', arr = x_test)   # x_test
# np.save(np_path + 'keras58_01_y_test.npy', arr = y_test)   # y_test