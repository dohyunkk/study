'''
2026-09-16 (수)

from tensorflow.keras.layers import Dense, Conv2D, Flatten
#2. 모델구성
strides = 1,         # default
padding = 'valid',   # default
'''

import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape = (10, 10, 1),  # ( 9, 9, 10)
                 strides = 1,       # default
                 padding = 'same',
                 ))

model.add(Conv2D(filters = 9, kernel_size = (3, 3),      # ( 7, 7, 9)
                 strides = 2,
                 padding = 'valid', # default
                 ))
                 
model.summary()