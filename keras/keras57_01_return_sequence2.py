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
x_predict = np.array([50, 60, 70])

x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(1,3,1)

print('x.shape :', x.shape)
# x.shape : (13, 3, 1)
print('y.shape : ', y.shape)
# (13,)



# 2. 모델구성
model = Sequential()
model.add(LSTM(units=32, input_shape=(3, 1), return_sequences=True,))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()
# exit()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  lstm (LSTM)                 (None, 3, 32)             4352      
                                                                 
#  lstm_1 (LSTM)               (None, 3, 32)             8320      
                                                                 
#  lstm_2 (LSTM)               (None, 3, 64)             24832     
                                                                 
#  lstm_3 (LSTM)               (None, 3, 32)             12416     
                                                                 
#  dense (Dense)               (None, 3, 256)            8448      
                                                                 
#  dense_1 (Dense)             (None, 3, 16)             4112      
                                                                 
#  dense_2 (Dense)             (None, 3, 1)              17        
                                                                 
# =================================================================
# Total params: 62,497
# Trainable params: 62,497
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.002
learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

es = EarlyStopping(         
    monitor = 'loss',        
    mode = 'auto',                 
    patience = 40,                
    verbose = 1,
    restore_best_weights = True,  
)

rlr = ReduceLROnPlateau(
    monitor='loss',
    mode='auto',
    patience=20,
    verbose=1, 
    factor=0.5,
)

model.fit(x, y, 
          epochs = 10000,
          batch_size = 32,
          verbose = 1,
          callbacks=[es, rlr],
)


#4. 평가 예측
results = model.evaluate(x, y)
print('loss : ', results)


x_predict = x_predict
y_predict = model.predict(x_predict)

print('[80]의 예측값 : ', y_predict)

'''
Epoch 1764: early stopping
1/1 [==============================] - 1s 543ms/step - loss: 0.0061
loss :  0.0061494638212025166
1/1 [==============================] - 0s 451ms/step
[80]의 예측값 :  [[78.732346]]

'''