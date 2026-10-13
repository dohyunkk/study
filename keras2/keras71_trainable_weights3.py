import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

print('tf.version : ', tf.__version__)

random.seed(333)   # 파이썬 랜덤값 고정
np.random.seed(333)  #넘파이도 랜덤값이 있다. 난수표라는게 있는데, 이 코드에 있는 모든 넘파이는 333의 랜덤값으로 고정하겠다.
tf.random.set_seed(333) # tensorflow 도 랜덤값을 난수표 상 333에 고정하겠다.

#1. 데이터
x = np.array([1,])
y = np.array([1,])


#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

################################# 동결 ######################################
# model.trainable = False         #* 동결 ★★★★★★
model.trainable = True
################################# 동결 ######################################
# model.summary()



# exit()
#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=500, verbose=1)

#4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)

print("=========================================================")
print(model.weights)
# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, value=[[ 0.8516406  -0.52920127 -0.9112464 ]]>,
#  <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, value=[0. 0. 0.]>, 
#  <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, value=[[ 0.6380346  -1.0862008 ]
#  [-0.38601977 -0.21482044]
#  [ 0.20393836 -0.87937653]]>,
#  <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, value=[0. 0.]>,
#  <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, value=[[-0.5044266 ] [ 0.90831745]]>,
#  <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, value=[0.]>]
print("=========================================================")


#! 동결 x
# [[1.0037434]
#  [2.0023544]
#  [3.0009656]
#  [3.9995768]
#  [4.9981885]]

#! 동결 o
# [[-0.07029988]
#  [-0.14059976]
#  [-0.2108996 ]
#  [-0.2811995 ]
#  [-0.35149962]]

