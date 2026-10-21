import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

print("tf.version : ", tf.__version__)

random.seed(333)  # 파이썬 랜덤값 고정
np.random.seed(
    333
)  # 넘파이도 랜덤값이 있다. 난수표라는게 있는데, 이 코드에 있는 모든 넘파이는 333의 랜덤값으로 고정하겠다.
tf.random.set_seed(333)  # tensorflow 도 랜덤값을 난수표 상 333에 고정하겠다.

# 1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])


# 2. 모델
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(64))
model.add(Dense(1))

# model.summary()

################################# 동결 ######################################
model.trainable = False  # 동결 ★★★★★★
# model.trainable = True
################################# 동결 ######################################
model.summary()

# 3. 컴파일 훈련
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, batch_size=1, epochs=100)

# 4. 평가, 예측
y_predict = model.predict(x)
print(y_predict)


#! 동결 x
# [[1.0037434]
#  [2.0023544]
#  [3.0009656]
#  [3.9995768]
#  [4.9981885]]

#! 동결 x
# [[-0.07029988]
#  [-0.14059976]
#  [-0.2108996 ]
#  [-0.2811995 ]
#  [-0.35149962]]
