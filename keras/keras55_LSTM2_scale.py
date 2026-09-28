'''
2026-09-28 (월)

실습 : x_predict = np.array([50, 60, 70])     #목표: 최대한 80 맞춰보아요.
'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1. 데이터
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              [8,9,10],
              [9,10,11],
              [10,11,12],
              [20,30,40], 
              [30,40,50],
              [40,50,60]
              ])  # (13, 3)

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0], x.shape[1], 1)
print('x.shape :', x.shape)
# x.shape : (13, 3, 1)
print('y.shape : ', y.shape)
# (13,)

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
model.add(LSTM(16, input_shape=(3, 1)))
# 3차원으로 들어가서 2차원 또는 1차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(         
    monitor = 'loss',        
    mode = 'auto',                 
    patience = 30,                
    verbose = 1,
    restore_best_weights = True,  
)

model.fit(x, y, 
          epochs = 10000,
          batch_size = 32,
          verbose = 1,
          callbacks=[es],
)


#4. 평가 예측
results = model.evaluate(x, y)
print('loss : ', results)


x_predict = np.array([50, 60, 70]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[80]의 예측값 : ', y_predict)

'''
# x_predict = np.array([50, 60, 70])     #목표: 최대한 80 맞춰보아요.

Epoch 1330: early stopping
1/1 [==============================] - 0s 215ms/step - loss: 0.0018
loss :  0.0018399360124021769
1/1 [==============================] - 0s 185ms/step
[80]의 예측값 :  [[78.81624]]

'''