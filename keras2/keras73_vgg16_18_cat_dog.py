'''
45 카피

# 동결 후 성능 안 좋으면 동결 풀고 학습
'''


from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import time

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import random

#! 랜덤값 고정 
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from tensorflow.keras.applications import VGG16

#1. 데이터
np_path = './_data/save_kaggle_cat_dog_npy/'

x_train = np.load(np_path + 'keras45_03_x_train.npy') 
y_train = np.load(np_path + 'keras45_03_y_train.npy') 
x_test = np.load(np_path + 'keras45_03_x_test.npy')
y_test = np.load(np_path + 'keras45_03_y_test.npy')

# print(x_train.shape, y_train.shape) # (8005, 150, 150, 3) (8005,)
# print(x_test.shape, y_test.shape)   # (2023, 150, 150, 3) (2023,)

# exit()

#2. 모델
vgg16 = VGG16(
weights='imagenet',
# include_top=True,
# input_shape=(224,224,3),
include_top=False,
input_shape=(150,150,3),
)

vgg16.trainable=False  #! 동결o
# vgg16.trainable=True    #! 동결x

# vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', 
              metrics=['acc'],
)

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

# #★★★★★★★ mcp 세이브 파일명 만들기 시작 ★★★★★★★##
# import datetime

# date = datetime.datetime.now()
# print(date)         # 2026-09-14 11:42:10.635267
# print(type(date))   # <class 'datetime.datetime'>        # ★ calss
# date = date.strftime("%m%d_%H%M")
# print(date)         # 2026-09-14 11:48:27.764409
# print(type(date))   # <class 'datetime.datetime'>

# path = './_save/keras45/'
# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, 'IDG3_CatDog_', date, "-",filename])

# #★★★★★★★ mcp 세이브 파일명 만들기 끝 ★★★★★★★##

# mcp = ModelCheckpoint(
#     monitor = 'val_loss',
#     mode = 'auto',
#     save_best_only = True,
#     filepath = filepath,
#     verbose = 1,
# )

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 300,
          batch_size = 16,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es],
          )

end_time = time.time()

# path = './_save/keras45/'
# # model.save(path + 'keras29_1_save_model.keras')
# model.save_weights(path + 'keras45_save_1.weights.h5')

#4. 평가, 예측
print("------------------ keras73_vgg16_cat_dog --------------------")
loss = model.evaluate(x_test, y_test, 
                      verbose = 1,
                      batch_size = 32,
)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_predict = np.round(y_predict) 

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')

'''

------------------ keras45_CatDog_load --------------------
253/253 [==============================] - 1s 5ms/step - loss: 0.3711 - acc: 0.8626
loss :  0.37105846405029297
acc :  0.8625802993774414



'''