'''
2026-09-23 (수)

52 카피

한도에 닿으면 러닝레이트를 절반으로 낮춰서 이어가겠다.
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1, 
    factor=0.5,
)
'''


import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import tensorflow as tf
import random

#! 랜덤값 고정 
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from keras.applications import VGG16
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# exit()
# plt.imshow(x_train[10], )
# plt.show()

# print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)

# print(np.min(x_train), np.max(x_train)) # 0 255
# print(np.min(x_test), np.max(x_test))   # 0 255
# print(np.min(y_train), np.max(y_train)) # 0 99
# print(np.min(y_test), np.max(y_test))   # 0 99

# exit()

############### 스케일링 1
x_train = x_train/255.
x_test = x_test/255.

# print(np.min(x_train), np.max(x_train))   # 0.0 1.0
# print(np.min(x_test), np.max(x_test))     # 0.0 1.0

################ reshape
# x_train = x_train.reshape(-1, 32 , 32 * 3)
# x_test = x_test.reshape(-1, 32 , 32 * 3)
# print(x_train.shape, x_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3)

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)
# (50000, 100) (10000, 100)

# exit()

# 2. 모델
vgg16 = VGG16(
weights='imagenet',
# include_top=True,
# input_shape=(224,224,3),
include_top=False,
input_shape=(32,32,3),
)

vgg16.trainable=False #! 동결o
# vgg16.trainable=True    #! 동결x

# vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(1024))
model.add(Dense(100, activation='softmax'))

# model.summary()


# exit()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', 
              optimizer = Adam(learning_rate=learning_rate), 
              metrics=['acc'],
              )

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(         
    monitor = 'val_loss',        
    mode = 'auto',                 
    patience = 40,                
    verbose = 1,
    restore_best_weights = True,  
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1, 
    factor=0.5,
)

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 5000,
          batch_size = 128,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es, rlr],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras73_vgg16_cifar100 --------------------")
loss = model.evaluate(x_test, y_test, verbose = 1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis = 1)#.reshape(-1, 1)
y_test = np.argmax(y_test, axis = 1)#.reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')

'''

cnn
Epoch 81: early stopping
------------------ 40_cifar100_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 1.8781 - acc: 0.5041
loss :  1.8780783414840698
acc :  0.5041000247001648
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.5041
걸린 시간 :  411.95 sec

#* vgg16.trainable=True    #! 동결x
Epoch 89: early stopping
------------------ keras73_vgg16_cifar100 --------------------
313/313 [==============================] - 2s 4ms/step - loss: 4.6059 - acc: 0.0100  
loss :  4.605860710144043
acc :  0.009999999776482582
313/313 [==============================] - 1s 4ms/step
accuracy_score :  0.01
걸린 시간 :  984.76 sec

#* vgg16.trainable=False    #! 동결o
Epoch 54: early stopping
------------------ keras73_vgg16_cifar100 --------------------
313/313 [==============================] - 4s 12ms/step - loss: 2.7771 - acc: 0.3293
loss :  2.777141809463501
acc :  0.3292999863624573
313/313 [==============================] - 3s 10ms/step
accuracy_score :  0.3293
걸린 시간 :  304.64 sec
'''