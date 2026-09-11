# keras12_3 카피
from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)  #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    random_state=99,
)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32,
           validation_split=0.25
          )

print("=============================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# 평가지표(r2)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 = : ", r2)

# loss :  3158.30517578125
# r2 = :  0.43387182657837486

# - loss: 2756.9297 - val_loss: 3143.7991
# loss :  3208.415771484375
# r2 = :  0.42488947072663297