# 2026-09-02 수

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)  #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=96, 
)

#2. 모델 구성
model = Sequential()
model.add(Dense(8, input_dim=8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

print("=============================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# loss :  0.6193994879722595


# results= model.predict(x)
# print(results)

# 그래프 그리기
# import matplotlib.pyplot as plt
# plt.scatter(x, y)
# plt.plot(x, results)
# plt.show()






