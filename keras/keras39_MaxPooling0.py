'''
2026-09-16 (수)

from tensorflow.keras.layers import Dense, Conv2D, Flatten, 
                                 ★ MaxPooling2D
#2. 모델구성

model.add(Conv2D(10, (2, 2), input_shape = (10, 10, 1),  # ( 9, 9, 10)
                 strides = 1,       # default
                 padding = 'same',
                 ))
model.add(MaxPooling2D())

MaxPool2D는 Conv2D 다음에 쓰인다
'''

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape = (10, 10, 1),  # ( 9, 9, 10)
                 strides = 1,       # default
                 padding = 'same',
                 ))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 9, kernel_size = (3, 3),      # ( 7, 7, 9)
                 strides = 2,
                 padding = 'valid', # default
                 ))
                 
model.summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                 Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)              (None, 10, 10, 10)        50        
                                                                 
#  max_pooling2d (MaxPooling2D)   (None, 5, 5, 10)         0           # Maxpooling을 거치니 shape가 반땡
#                                                                 
                                                                 
#  conv2d_1 (Conv2D)            (None, 2, 2, 9)           819       
                                                                 
# =================================================================
# Total params: 869
# Trainable params: 869
# Non-trainable params: 0
# _________________________________________________________________