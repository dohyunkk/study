# 2026-09-02 수

# 09 카피
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_tset = np.array([8,9,10])

#[찾아보기] 넘파이 리스트의 슬라이싱 =>7:3으로 나누자

x_train = x[:7] #x_train = x[0:7] 와 같다
y_train = y[:7]
print(x_train)


x_test = x[7:]  #x_test = x[7:10] 와 같다
y_test = y[7:]
print(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=120, batch_size=6)

#4. 평가, 예측
loss = model.evaluate(x_test, y_tset)
print('loss : ' , loss)