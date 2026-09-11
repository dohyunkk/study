# 2026-09-02 수

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_tset = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=120, batch_size=6)

print("=========================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_tset)
print('loss : ' , loss)

results = model.predict(x)
print(results)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)     # 데이터 점 찍기
plt.plot(x, results)  # 데이터 선 긋기
plt.show()            # 점 찍고 선 그은 거 보여줘


# 결과. loss :
