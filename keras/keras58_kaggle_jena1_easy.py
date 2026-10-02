"""수업에서 배운 방식으로 Jena 기온 144개 예측하기."""

# import os
# os.environ["TF_GRU_ALLOCATOR"] = "cuda_malloc_async"    # 메모리 모으기

import numpy as np
import pandas as pd
import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# 1. 데이터

path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# print(datasets.shape) # (420551, 14)

# 수정 : T (degC) < 이놈을 y로 잡는다.

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

print('x, y:', x.shape, y.shape)
print("자르는시간:", round(end_time - start_time, 2),'sec')
# 자르는시간: 24.31 sec

# 2. 선생님 코드와 같은 train/test 분리.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False
)


#스케일링. fit은 train에만 한다.
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_shape = x_train.shape
x_test_shape = x_test.shape
x_train = scaler_x.fit_transform(x_train.reshape(-1, 13)).reshape(x_train_shape)
x_test = scaler_x.transform(x_test.reshape(-1, 13)).reshape(x_test_shape)

y_train_shape = y_train.shape
y_test_shape = y_test.shape
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train_shape)
y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test_shape)

print('x_train, y_train:', x_train.shape, y_train.shape)


# 4. 모델
model = Sequential()
model.add(LSTM(units=32, input_shape=(144, 13)))
model.add(Dense(64, activation='relu'))
model.add(Dense(144))

model.summary()

# exit()




# 컴파일, 훈련
model.compile(
    loss='mse', optimizer=Adam(learning_rate=0.001))


es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=3,
    verbose=1,
    restore_best_weights=True
)


# rlr = ReduceLROnPlateau(
#     monitor='val_loss',
#     mode='min',
#     patience=5,
#     factor=0.5,
#     verbose=1
# )

#★★★★★★★ mcp 세이브 파일명 만들기 시작 ★★★★★★★##
import datetime

date = datetime.datetime.now()
print(date)         # 
print(type(date))   # <class 'datetime.datetime'>        # ★ calss
date = date.strftime("%m%d_%H%M")
print(date)         # 
print(type(date))   # <class 'datetime.datetime'>

path = './_save/keras58/01/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'jena_easy_', date, "-",filename])

#★★★★★★★ mcp 세이브 파일명 만들기 끝 ★★★★★★★##

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath,
    verbose = 1,
)

start_fit = time.time()

model.fit(
    x_train,y_train,
    epochs=1000,
    batch_size=256,
    shuffle=False,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, mcp]
)

end_fit = time.time()

print('test loss:', model.evaluate(x_test, y_test))


# 5. 마지막 144개 기온 예측. 입력에는 바로 앞의 144개 날씨를 쓴다.
x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)
x_predict = scaler_x.transform(x_predict).reshape(1, 144, 13)

y_predict = model.predict(x_predict).reshape(-1, 1)
y_predict = scaler_y.inverse_transform(y_predict).reshape(-1)

rmse = np.sqrt(mean_squared_error(y_cor.to_numpy(), y_predict))
print('실제 기온:', y_cor.to_numpy())
print('예측 기온:', y_predict)
print('RMSE (°C):', rmse)
print('훈련 시간 : ', round(end_fit - start_fit, 2),'sec')