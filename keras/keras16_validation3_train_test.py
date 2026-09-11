# 16_2 카피
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
"""
# 실습 8개, 4개, 4개 잘라봅시다

x_train = x[:8]
y_train = y[:8]

x_val = x[8:12]
y_val = y[8:12]

x_test = x[12:]
y_test = y[12:]
"""

# [실습] train_test_split으로 짤라보세요.
# x_train, x_val, x_test, y_train, y_val, y_test = train_test_split(
#     x, y, val_size=
# )

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    #   train_size=0.75,
    #   test_size=0.25,
    #  shuffle=True,          #디폴트 설정값 = 섞는다.
      random_state=99,   #랜덤 난수표 중 고정값을 넣어 고정된 리턴을 뽑아낸다.
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    #   train_size=0.75,
    #   test_size=0.25,
    #  shuffle=True,          #디폴트 설정값 = 섞는다.
      random_state=99,   #랜덤 난수표 중 고정값을 넣어 고정된 리턴을 뽑아낸다.
)

print(x_train.shape, x_val.shape, x_test.shape) #(9,) (3,) (4,)
print(y_train.shape, y_val.shape, y_test.shape) #(9,) (3,) (4,)

"""
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=111)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y, test, test_size=0.5, random_state=111)


"""

# 모델구성
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=250, batch_size=6,
          verbose = 1,
          validation_data = (x_val, y_val),
          )


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

"""
- loss: 0.0026 - val_loss: 0.0049
loss :  0.003999218810349703

"""


