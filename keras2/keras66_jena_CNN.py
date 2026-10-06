# 3차원(LSTM)에서 4차원(CNN)으로
# 58_1_jena 카피

"""수업에서 배운 방식으로 Jena 기온 144개 예측하기."""

# import os
# os.environ["TF_GRU_ALLOCATOR"] = "cuda_malloc_async"    # 메모리 모으기

import numpy as np
import pandas as pd
import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, LSTM, Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten
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
# x_train, y_train: (336096, 144, 13) (336096, 144)
# exit()
# 4. 모델
model = Sequential()
model.add(LSTM(units=16, return_sequences=True, input_shape=(144, 13)))
model.add(Reshape((144, 16, 1)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(144))

model.summary()

# exit()


# 컴파일, 훈련
model.compile(
    loss='mse', optimizer=Adam(learning_rate=0.001))


es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    verbose=1,
    restore_best_weights=True
)


rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=5,
    factor=0.5,
    verbose=1
)

#★★★★★★★ mcp 세이브 파일명 만들기 시작 ★★★★★★★##
import datetime

date = datetime.datetime.now()
print(date)         # 
print(type(date))   # <class 'datetime.datetime'>        # ★ calss
date = date.strftime("%m%d_%H%M")
print(date)         # 
print(type(date))   # <class 'datetime.datetime'>

path = './_save/keras66/01/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'jena_CNN_', date, "-",filename])

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
    callbacks=[es, mcp, rlr]
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



'''
Epoch 11: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
1051/1051 [==============================] - 59s 57ms/step - loss: 0.1217 - val_loss: 0.1670 - lr: 0.0010
Epoch 11: early stopping
2026-10-06 10:43:42.037653: W tensorflow/core/common_runtime/bfc_allocator.cc:290] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.04GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
2026-10-06 10:43:42.041713: W tensorflow/core/common_runtime/bfc_allocator.cc:290] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.04GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
2624/2626 [============================>.] - ETA: 0s - loss: 0.18452026-10-06 10:43:55.190611: W tensorflow/core/common_runtime/bfc_allocator.cc:290] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.79GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
2026-10-06 10:43:55.196568: W tensorflow/core/common_runtime/bfc_allocator.cc:290] Allocator (GPU_0_bfc) ran out of memory trying to allocate 2.79GiB with freed_by_count=0. The caller indicates that this is not a failure, but this may mean that there could be performance gains if more memory were available.
2626/2626 [==============================] - 13s 5ms/step - loss: 0.1845
test loss: 0.18447351455688477
1/1 [==============================] - 0s 363ms/step
실제 기온: [-4.17 -4.17 -4.09 -4.04 -4.14 -4.3  -4.26 -4.58 -4.68 -4.82 -4.96 -5.08
 -5.05 -5.25 -5.34 -5.15 -5.07 -4.96 -4.67 -4.17 -4.07 -3.97 -4.14 -4.15
 -4.47 -4.61 -4.86 -4.92 -5.12 -5.22 -4.9  -4.86 -4.99 -4.92 -5.07 -5.27
 -5.17 -5.37 -5.43 -5.62 -5.77 -6.09 -6.2  -6.04 -6.26 -6.35 -6.46 -6.8
 -6.84 -6.64 -6.87 -6.31 -6.42 -6.36 -6.31 -6.37 -6.09 -5.96 -5.6  -5.24
 -5.02 -4.48 -4.6  -4.41 -4.15 -3.76 -3.31 -2.96 -2.52 -1.92 -1.89 -1.19
 -0.71 -0.2   0.58  1.26  1.57  1.64  1.97  2.17  2.55  3.08  3.92  3.97
  4.15  4.41  4.78  5.19  5.21  5.09  4.97  4.6   4.42  3.98  3.56  2.87
  2.44  2.13  1.98  1.67  1.61  1.41  1.29  0.79  0.44  0.24 -0.06 -0.08
 -0.42 -0.95 -1.28 -1.67 -1.25 -1.03 -0.98 -1.4  -1.47 -1.61 -1.65 -1.52
 -1.4  -2.15 -3.19 -3.3  -3.46 -3.09 -2.75 -2.61 -2.51 -2.48 -2.48 -2.59
 -2.89 -3.22 -4.08 -4.45 -4.09 -3.76 -3.93 -4.05 -3.35 -3.16 -4.23 -4.82]
예측 기온: [-1.7049415  -1.7169654  -1.7480018  -1.7087934  -1.6575878  -1.7612569
 -1.6788967  -1.6381719  -1.6349742  -1.6682794  -1.6894443  -1.7286937
 -1.6945779  -1.5725582  -1.6143997  -1.5742223  -1.553726   -1.5776365
 -1.5727613  -1.4784706  -1.5074909  -1.5380657  -1.4494731  -1.5431306
 -1.472224   -1.4952056  -1.4350889  -1.4152095  -1.4417341  -1.3671672
 -1.4426868  -1.3759935  -1.2767885  -1.3756244  -1.3637598  -1.2693241
 -1.3142555  -1.3127935  -1.2511384  -1.2384384  -1.1934907  -1.1476772
 -1.1667516  -1.2323387  -1.1568935  -1.1299684  -1.1472137  -1.036166
 -1.1400173  -1.078078   -1.0645311  -1.001972   -1.0532396  -1.0146911
 -1.0009849  -0.92805463 -0.89976865 -0.9222229  -0.9370745  -0.918268
 -0.92738324 -0.8457468  -0.9216526  -0.89348394 -0.90052396 -0.8482531
 -0.821225   -0.8553751  -0.8705738  -0.7382725  -0.7960833  -0.83261853
 -0.8402403  -0.8020285  -0.77250844 -0.7513054  -0.78859407 -0.759137
 -0.6990202  -0.83995706 -0.76521474 -0.7793701  -0.74579984 -0.8223017
 -0.7797392  -0.81183606 -0.7426289  -0.6825207  -0.7912491  -0.7902344
 -0.7258766  -0.73118764 -0.8280104  -0.7421463  -0.8218201  -0.696451
 -0.7567871  -0.7365883  -0.73150903 -0.83515817 -0.7281187  -0.8034094
 -0.748431   -0.7735947  -0.8600758  -0.8688181  -0.8901413  -0.8464306
 -0.8281048  -0.9232357  -0.8335617  -0.8594263  -0.8589762  -0.8721159
 -0.89513284 -0.82769376 -0.8954056  -0.85126954 -0.9776228  -0.8948315
 -0.9618967  -0.9019945  -1.0340955  -0.9362944  -0.9223278  -0.9495495
 -0.9937523  -0.9361275  -0.97134095 -1.0244644  -1.0102098  -1.022207
 -1.0142982  -1.0657833  -0.97980577 -0.9681795  -1.050204   -1.0120008
 -1.0056617  -0.9722517  -1.0177076  -1.0341251  -1.0222518  -1.0672309 ]
RMSE (°C): 3.4261584478199776
훈련 시간 :  637.54 sec

'''
