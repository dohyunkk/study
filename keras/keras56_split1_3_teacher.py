'''
2026-09-28 (월)

def split_x(dataset, size):                  # 함수 정의      : dataset을 size만큼 잘라주는 함수split_x 만들기
    aaa = []                                 # 빈 리스트 생성 : 잘라낸 데이터를 담을 빈 리스트
    for i in range(len(dataset) - size + 1): # 반복           : 자를 수 있는 횟수만큼 반복
        subset = dataset[i : (i+size)]       # 데이터 자르기  : i번째부터 size개만큼 자르기
        aaa.append(subset)                   # 리스트에 추가  : 잘라낸 데이터를 aaa에 추가
    return np.array(aaa)                     # 결과 반환      : aaa를 넘파이 배열로 바꿔서 반환

'''

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#1.데이터

a = np.array(range(1,11))  # [1,2,3,4,5,6,7,8,9,10]
size = 5                   # timestep 사이즈

# print(a.shape)           # (10,)

def split_x(dataset, size):                  # 함수 정의      : dataset을 size만큼 잘라주는 함수split_x 만들기
    aaa = []                                 # 빈 리스트 생성 : 잘라낸 데이터를 담을 빈 리스트
    for i in range(len(dataset) - size + 1): # 반복           : 자를 수 있는 횟수만큼 반복
        subset = dataset[i : (i+size)]       # 데이터 자르기  : i번째부터 size개만큼 자르기
        aaa.append(subset)                   # 리스트에 추가  : 잘라낸 데이터를 aaa에 추가
    return np.array(aaa)                     # 결과 반환      : aaa를 넘파이 배열로 바꿔서 반환

bbb = split_x(a, size)
print(bbb)
print(bbb.shape) # (6, 5)

x = bbb[:, :-1]     # batch_size

y = bbb[:, -1]      # batch_size

print('x : ')
print(x)
print('y : ')
print(y)

print(x.shape, y.shape)
# # (6, 4) (6,)


# LSTM 입력을 위해 3차원으로 변환
x = x.reshape(x.shape[0], x.shape[1], 1)
print('x.shape : ', x.shape)
# (6, 4, 1)

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(4, 1)))
model.add(LSTM(16, input_shape=(4, 1)))
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
print('------------keras56_split1_3_teacher----------------')

results = model.evaluate(x, y)
print('loss : ', results)


x_predict = np.array([7, 8, 9, 10]).reshape(1, 4, 1)
y_predict = model.predict(x_predict)

print('[7, 8, 9, 10] 다음 예측값 : ', y_predict)

'''
Epoch 708: early stopping
1/1 [==============================] - 0s 199ms/step - loss: 2.2766e-05
loss :  2.2765956600778736e-05
1/1 [==============================] - 0s 177ms/step
[7, 8, 9, 10] 다음 예측값 :  [[10.470667]]

'''