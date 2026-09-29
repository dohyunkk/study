'''
2029-09-29
55_2 카피

# 2. 모델구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(3, 1), return_sequences=True,)) ★
model.add(LSTM(5, return_sequences=True))                             ★
model.add(LSTM(5))                                                    ★
model.add(Dense(8))
model.add(Dense(1))


'''

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
x_predict = np.array([50, 60, 70])

x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(1,3,1)

print('x.shape :', x.shape)
# x.shape : (13, 3, 1)
print('y.shape : ', y.shape)
# (13,)



# 2. 모델구성
model = Sequential()
model.add(LSTM(units=10, input_shape=(3, 1), return_sequences=True,))
model.add(LSTM(5, return_sequences=True))
model.add(LSTM(5))
model.add(Dense(8))
model.add(Dense(1))

model.summary()
# exit()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  lstm (LSTM)                 (None, 3, 10)             480       
                                                                 
#  lstm_1 (LSTM)               (None, 3, 5)              320       
                                                                 
#  lstm_2 (LSTM)               (None, 5)                 220       
                                                                 
#  dense (Dense)               (None, 8)                 48        
                                                                 
#  dense_1 (Dense)             (None, 1)                 9         
                                                                 
# =================================================================
# Total params: 1,077
# Trainable params: 1,077
# Non-trainable params: 0
# _________________________________________________________________
