'''

54-1 카피

x.shape = (batch_size, time_steps, features)

model.add(SimpleRNN(5, input_shape=(3, 1)))
                
            batch_size = 5, time_steps = 3, features = 1
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
print('x.shape :', x.shape)              # (7, 3, 1)
#     x.shape : (batch_size, time_steps, features)

#2. 모델 구성
# Simple RNN 파라미터 참고 
# => output_dim, ( time-length or setence-length, input_dim) 
#         => Dh, (t, d) 

model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(3, 1)))
model.add(SimpleRNN(5, input_shape=(3, 1)))
# model.add(SimpleRNN(5, input_length=3, input_dim=1))와 동일함.
# 3차원으로 들어가서 2차원 또는 1차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(7, activation='relu'))
model.add(Dense(1))


model.summary()

# SimpleRNN 파라미터 수 = (features × units) + (units × units) + units
#                             1        5        5        5        5 
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 5)                 35          # ★
                                                                 
#  dense (Dense)               (None, 7)                 42        
                                                                 
#  dense_1 (Dense)             (None, 1)                 8         
                                                                 
# =================================================================
# Total params: 85
# Trainable params: 85
# Non-trainable params: 0
# _________________________________________________________________

