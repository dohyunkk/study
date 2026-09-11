# keras16_3 카피

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
       train_size=0.75,
       random_state=99,   #랜덤 난수표 중 고정값을 넣어 고정된 리턴을 뽑아낸다.
)


# print(x_train.shape, x_val.shape, x_test.shape) #(9,) (3,) (4,)
# print(y_train.shape, y_val.shape, y_test.shape) #(9,) (3,) (4,)

"""
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=111)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y, test, test_size=0.5, random_state=111)


"""

# 2.모델구성
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))


#3. 컴파일, 훈련, 훈련검증
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=6,
          verbose = 1,
        #   validation_data = (x_val, y_val),
        validation_split=0.33,  # 1. 데이터에서 검증을 분리해도 되고, 훈련에서 분리해도 된다.
          )


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

"""
- loss: 0.0026 - val_loss: 0.0049
loss :  0.003999218810349703

"""


