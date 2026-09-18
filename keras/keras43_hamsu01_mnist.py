'''
2026-09-18 (금)

Dnn -> 함수형

!! 목표 : dnn 모델을 함수형 모델로 리모델링
'''

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

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
x_train = x_train/255.                    # ★ .을 붙이면 float 형태로 출력하게 된다.
x_test = x_test/255.                      # ★
# print(np.min(x_train), np.max(x_train))   # 0.0 1.0
# print(np.min(x_test), np.max(x_test))     # 0.0 1.0

############### 스케일링 2
# x_train = (x_train - 127.5)/127.5             
# x_test = (x_test - 127.5)/127.5                
# print(np.min(x_train), np.max(x_train))   # -1.0 1.0
# print(np.min(x_test), np.max(x_test))     # -1.0 1.0

x_train = x_train.reshape(-1, 28 * 28 * 1)    # ★
x_test = x_test.reshape(-1 , 28 * 28 * 1)      # ★
print(x_train.shape, x_test.shape)
# (60000, 784) (10000, 784)

# exit()

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)
# (60000, 10) (10000, 10)



# 2.모델구성
# DNN 모델
# model = Sequential()
# model.add(Dense(1024, activation='relu', input_shape = (784,)))
# model.add(Dropout(0.2))

# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))

# model.add(Dense(10, activation='softmax'))


# 함수형 모델
input1 = Input(shape=(784,))
dense1 = Dense(1024, activation = 'relu')(input1)
drop1 = Dropout(0.2)(dense1)

dense2 = Dense(512, activation = 'relu')(drop1)
drop2 = Dropout(0.2)(dense2)

dense3 = Dense(256, activation = 'relu')(drop2)
drop4 = Dropout(0.2)(dense3)

dense4 = Dense(128, activation = 'relu')(drop4)
drop5 = Dropout(0.3)(dense4)

output1 = Dense(10, activation = 'softmax')(drop5)

model = Model(inputs=input1, outputs=output1)



# exit()

# CNN 모델
# # 블록 1: 
# model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
# model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D()) # 크기 반감 -> (14, 14, 32)
# model.add(Dropout(0.25))

# # 블록 2: 첫 번째 레이어에서만 strides=(2, 2) 적용하여 반으로 줄임
# model.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu')) # 크기 반감 -> (7, 7, 64)
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                 # 크기 유지 -> (7, 7, 64)
# # model.add(MaxPooling2D()) <- 차원 붕괴를 막기 위해 삭제 완료!
# model.add(Dropout(0.25))

# model.add(GlobalAveragePooling2D()) # (64, )

# # Fully Connected 분류층
# model.add(Dense(32, activation='relu'))

# model.add(Dense(10, activation='softmax'))

model.summary()

# exit()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', 
              metrics=['acc'],
              )

start_time = time.time()

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

model.fit(x_train, y_train,
          epochs = 500,
          batch_size = 128,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es ],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ 43_hamsu_mnist_model.evaluate --------------------")
loss = model.evaluate(x_test, y_test, verbose = 1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis = 1)#.reshape(-1, 1)
y_test_labels = np.argmax(y_test, axis = 1)#.reshape(-1, 1)

acc_score = accuracy_score(y_test_labels, y_predict)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')


'''
cnn
Epoch 35: early stopping
------------------ 39_mnist_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.0204 - acc: 0.9940
loss :  0.020386828109622
acc :  0.9940000176429749
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.994
걸린 시간 :  60.83 sec


Epoch 24: early stopping
------------------ 43_hamsu_mnist_model.evaluate --------------------
313/313 [==============================] - 0s 1ms/step - loss: 0.0757 - acc: 0.9818
loss :  0.07566951215267181
acc :  0.9818000197410583
313/313 [==============================] - 0s 751us/step
accuracy_score :  0.9818
걸린 시간 :  19.95 sec
'''