# 2026-09-01 화

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x = np.array(range(10))
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]]).transpose()
print(x.shape, y.shape) # (10,) (10, 3)

# [실습]
# 11, 0, -1 이 나오면 좋다
#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(32))
model.add(Dense(3))


#컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=300,batch_size=32)

#평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)
results = model.predict(np.array([10]))
print("[11, 0, -1]의 예측값 : ", results)