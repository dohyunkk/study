'''
2026-09-17 (목)

DNN모델을 CNN모델로 변환
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.datasets import boston_housing

# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
# print(y_train.shape, y_test.shape) # (404,) (102,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)

# print(x_train)

# print(x_test)

# print(np.min(x_train), np.max(x_train))
# 0.0 1.0000000000000002
# print(np.min(x_test), np.max(x_test))
# -0.0019120458891013214 1.1478180091225068


x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)

print(x_train.shape) # 

# exit()

# 2. 모델구성

model = Sequential()
model.add(Conv2D(64, (2, 1), padding = 'same', activation = 'relu', input_shape = (13, 1, 1)))
model.add(Conv2D(64, (2, 1), padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))

model.add(Conv2D(32, (2, 1), padding = 'same', activation = 'relu'))
model.add(Conv2D(32, (2, 1), padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(              
    monitor = 'val_loss',        
    mode = 'min',                 
    patience = 30,                
    restore_best_weights = True,
    verbose = 1,
    )

start_time = time.time()    

hist = model.fit(x_train, y_train, 
                 epochs=1000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es]
                 )

end_time = time.time()      

print("===================keras42_cnn_boston==========================")

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, )
print('loss(mse) : ', loss)

# 평가지표 (r2)
y_predict = model.predict(x_test)      # results = model.predict(x)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)
#RMSE함수의 정의
def RMSE(y_test, y_predict):    
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


# loss(mse) :  23.699596405029297
# r2 =: 0.7152991240762334


# - loss: 37.3669 - val_loss: 69.8842
# loss(mse) :  54.49088668823242
# r2 : 0.34540641003351025

print("훈련시간 :", round(end_time - start_time, 2),"sec")

# print("============================ history =================================")
# print(hist)
# print("========================= hist.histoy ==============================")
# print(hist.history)
# print("============================= loss =================================")
# print(hist.history['loss'])
# print("=========================== val_loss =================================")
# print(hist.history['val_loss'])
# print("============================ history =================================")

# import matplotlib.pyplot as plt

# print(plt.rcParams['font.family'])         # 그래프 출력시 ['sans-serif']폰트 가 디폴트이기 때문에 한글이 출력되지 않는다.

# plt.rcParams['font.family'] = 'Malgun Gothic'   # 그래서 맷플롯립 폰트를 한글을 지원하는 폰트로 삽입하여 그래프에 한글이 표시되도록 한다.

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][5:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][5:], c='blue', label='val_loss')
# plt.legend(loc='upper right')    # 우측 상단에 라벨표시
# plt.title('보스턴 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')

# plt.grid()    #격자 표시를 추가
# plt.show()    #


"""
--cpu-------------------------------------------
Epoch 30/30
11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 67.5976 - val_loss: 44.7122
=====================keras35_boston==============================
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 39.6326 
loss(mse) :  39.63264465332031
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
r2 : 0.5238969978824023
mse :  39.632646188099365
RMSE :  6.295446464556693
훈련시간 : 2.37 sec

cnn-----------------------------------------------
Epoch 267: early stopping
===================keras42_cnn_boston==========================
4/4 [==============================] - 0s 37ms/step - loss: 24.4603
loss(mse) :  24.460315704345703
4/4 [==============================] - 0s 2ms/step
r2 : 0.7061607109919852
mse :  24.460313263348937
RMSE :  4.945736877690617
훈련시간 : 15.24 sec
"""