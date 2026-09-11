# 2026-09-01 화

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])
x = x.T                      #x = x.transpose() 와 같다
# x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]])
y = np.array([1,2,3,4,5])

print(x.shape)  # (5,2)
print(y.shape)  # (5,)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2)) #행 무시, 열 우선
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))  

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
results = model.predict(np.array([[6,11]]))
print("[6, 11]의 예측값 : ", results)