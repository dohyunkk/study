# 2026-09-02 수

from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)  #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state= 3000,
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(6))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

print("=============================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# results= model.predict(x)
# print(results)



