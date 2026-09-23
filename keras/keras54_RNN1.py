'''
2026-09-23 (수)

from tensorflow.keras.models import Dense, SimpleRNN

DNN은 2차원 데이터
RNN은 3차원 데이터
CNN은 4차원 데이터
'''

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
datatests = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
])
y= np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape)     # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)              # (7, 3, 1)


#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
model.add(SimpleRNN(32, input_shape=(3, 1)))
# 3차원으로 들어가서 2차원 또는 1차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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


x_predict = np.array([8,9,10]).reshape(1,3,1)
y_predict = model.predict(x_predict)

print('[8,9,10]의 예측값 : ', y_predict)

'''
목표는 11

Epoch 519: early stopping
1/1 [==============================] - 0s 105ms/step - loss: 4.1762e-06
loss :  4.176234597252915e-06
1/1 [==============================] - 0s 86ms/step
[8,9,10]의 예측값 :  [[10.723633]]
'''