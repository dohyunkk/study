'''
53 카피
'''

import numpy as np
import pandas as pd
import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Reshape
from tensorflow.keras.layers import GlobalAveragePooling1D, Dropout, MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)  #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    random_state=95,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

print(x_test)

print(x_train.shape, y_train.shape)  # (331, 10) (331,)
print(x_test.shape, y_test.shape)    # (111, 10) (111,)

# print(np.min(x_train), np.max(x_train))
# 0.0 1.0
# print(np.min(x_test), np.max(x_test))
# -0.10144927536231879 1.0144927536231885

# exit()

#2. 모델 구성
model = Sequential()
model.add(Reshape(target_shape=(10, 1), input_shape=(10, )))

model.add(Conv1D(filters=32, kernel_size=2, padding='same',input_shape=(10, 1)))

model.add(Conv1D(64, 2, padding='same'))
# model.add(MaxPool1D())

model.add(Conv1D(128, 2, padding='same'))
model.add(MaxPool1D())

# model.add(GlobalAveragePooling1D())
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()

# exit()




# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009


model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

es = EarlyStopping(         
    monitor = 'val_loss',        
    mode = 'auto',                 
    patience = 40,                
    verbose = 1,
    restore_best_weights = True,  
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=10,
    verbose=1, 
    factor=0.5,
)

start_time = time.time()     # 현재 시간을 반환. 시작시간

hist = model.fit(x_train, y_train, 
                 epochs=10000, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es, rlr],
                 )

end_time = time.time()       # 현재 시간을 반환. 종료시간

print("=============== keras68_Conv1D_diabetes =====================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# 평가(보조)지표-R2
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("r2 = : ", r2 )

mse = mean_squared_error(y_test, y_predict)
print('mse : ', mse)

#RMSE함수의 정의
def RMSE(y_tset, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

print("훈련시간 :", round(end_time - start_time, 2),"sec")


'''
Epoch 98: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
9/9 [==============================] - 0s 6ms/step - loss: 2126.4380 - val_loss: 3080.1663 - lr: 1.2500e-04
Epoch 98: early stopping
=============== keras68_Conv1D_diabetes =====================
4/4 [==============================] - 0s 14ms/step - loss: 3615.0356
loss :  3615.03564453125
4/4 [==============================] - 0s 0s/step
r2 = :  0.4204945757327093
mse :  3615.035640923073
RMSE :  60.125166452352325
훈련시간 : 7.12 sec
'''