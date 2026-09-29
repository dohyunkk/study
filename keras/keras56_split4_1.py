'''
2026-09-29

'''

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 1. 데이터
a = np.array(range(1, 101)).reshape(-1, 2)
x_predict = np.array(range(96, 106)).reshape(-1, 2)

size = 6

# 데이터를 reshape한 후 , split_x 함수로 시계열데이터로 변환.
# (n, 10, 1)의 데이터를 (n, 5, 2)의 데이터로 변환해보기.

print('a.shape : ')
print(a.shape)
# (50, 2)

print(x_predict.shape)
# (5, 2)
# exit()
def split_x(dataset, size):                  
    aaa = []                                 
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]       
        aaa.append(subset)                   
    return np.array(aaa)                     

bbb = split_x(a, size)
print('bbb : ')
print(bbb)
print('bbb.shape : ')
print(bbb.shape)
# (45, 6, 2)

x = bbb[:, :-1, :]

y = bbb[:, -1, -1]

print('x:')
print(x)
print('y:')
print(y)

print('x.shape :', x.shape)
# (45, 5, 2)
print('y.shape :', y.shape)
# (45,)

# 예측데이터

# x_predict = x_predict.reshape(1, 5, 2)
x_predict = split_x(x_predict, size-1)

print('x_preict : ')
print(x_predict)

print('x_predict.shape :', x_predict.shape)
# (1, 5, 2)

# exit()
#2. 모델구성
model = Sequential()
model.add(LSTM(32, input_shape=(5, 2)))
# 3차원으로 들어가서 2차원 또는 1차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
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
print('------------keras56_split3_1------------------')


results = model.evaluate(x, y)
print('loss : ', results)


# x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)

y_predict = model.predict(x_predict)

print('목표는 107 : ')
print(y_predict)