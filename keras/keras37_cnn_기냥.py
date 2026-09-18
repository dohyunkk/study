'''
2026-09-16 (수)

36_1 카피
각 부분에 들어가는 용어들의 의미를 
model.add(Conv2D(10, (3, 3), input_shape= ( a, b, c)))
                            # (a = height, b = width, c = channel)
10 = filter
(3, 3) = kernel_size
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D  # 이미지를 짜르면(조각조각) conv2D를 쓴다.

model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape= (10, 10, 1)))
                                # (height, width, channel) 요기가 기냥
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

