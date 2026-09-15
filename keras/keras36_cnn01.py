'''
2026-09-15 (화)

from tensorflow.keras.layers import Dense, Conv2D

#2. 모델
커널사이즈, 알고 봤더니 가중치였더라.
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D  # 이미지를 짜르면(조각조각) conv2D를 쓴다.

model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape= (10, 10, 1)))
model.add(Conv2D(5, (2, 2)))

# model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 8, 8, 10)          100                                                                       
#  conv2d_1 (Conv2D)           (None, 7, 7, 5)           205                                                                       
# =================================================================
# Total params: 305
# Trainable params: 305
# Non-trainable params: 0
# _________________________________________________________________

