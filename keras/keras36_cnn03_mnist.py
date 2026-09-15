'''
2026-09-15 (화)

#1.데이터
이미지 데이터의 x값은 거의 255이기 때문에, 스케일러를 부르지 않아도 입력할 수 있다.
'''

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

# print(np.max(x_train), np.min(x_train)) # 255 0
# print(np.max(x_test), np.min(x_test))   # 255 0
# print(np.max(y_train), np.min(y_train)) # 9 0
# print(np.max(y_test), np.min(y_test))   # 9 0


#이미지 데이터의 x값은 거의 255이기 때문에, 스케일러를 부르지 않아도 입력할 수 있다.

############### 스케일링 1
# x_train = x_train/255.                    # ★ .을 붙이면 float 형태로 출력하게 된다.
# x_test = x_test/255.                      # ★
# print(np.min(x_train), np.max(x_train))   # 0.0 1.0
# print(np.min(x_test), np.max(x_test))     # 0.0 1.0

############### 스케일링 2
x_train = (x_train - 127.5)/127.5             
x_test = (x_test - 127.5)/127.5                
print(np.min(x_train), np.max(x_train))   # -1.0 1.0
print(np.min(x_test), np.max(x_test))     # -1.0 1.0



