'''
2026-09-10 (목)
파라미터의 갯수를 확인해보자.

#2. 모델
model.summary()

y = w x + ★b
사실은
y = x w + b 였다 .. 이유는 [행렬]연산이고, [[메트릭스]] 연산이고, [[[텐][서]]] 연산이기 때문
하지만 수업의 편의상 wx + b 라고 해도 xw + b로 알아들을 것.
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np




#2. 모델
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.summary()
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ dense (Dense)                        │ (None, 3)                   │               6 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 4)                   │              16 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ (None, 3)                   │              15 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ (None, 1)                   │               4 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 41 (164.00 B)
#  Trainable params: 41 (164.00 B)
#  Non-trainable params: 0 (0.00 B)






