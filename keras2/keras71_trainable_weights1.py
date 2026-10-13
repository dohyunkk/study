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
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])


#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print("===================model.weights=============================")
print(model.weights)

print("==================model.trainable weights==============================")
print(len(model.weights))                  # 6
print(len(model.trainable_weights))        # 6

################################# 동결 ######################################
model.trainable = False         # 동결 중요 ★★★★★★
print(len(model.weights))                  # 6
print(len(model.trainable_weights))        # 0

################################# 동결 ######################################

print(model.weights)
# [<Variable path=sequential/dense/kernel, shape=(1, 3), dtype=float32, value=[[ 0.8516406  -0.52920127 -0.9112464 ]]>, <Variable path=sequential/dense/bias, shape=(3,), dtype=float32, value=[0. 0. 0.]>, <Variable path=sequential/dense_1/kernel, shape=(3, 2), dtype=float32, value=[[ 0.6380346  -1.0862008 ]
#  [-0.38601977 -0.21482044]
#  [ 0.20393836 -0.87937653]]>, <Variable path=sequential/dense_1/bias, shape=(2,), dtype=float32, value=[0. 0.]>, <Variable path=sequential/dense_2/kernel, shape=(2, 1), dtype=float32, value=[[-0.5044266 ]
#  [ 0.90831745]]>, <Variable path=sequential/dense_2/bias, shape=(1,), dtype=float32, value=[0.]>]