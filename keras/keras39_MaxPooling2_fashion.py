'''
2026-09-17 (목)


'''

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.datasets import fashion_mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

print(np.min(x_train), np.max(x_train)) # 0 255
print(np.min(x_test), np.max(x_test))   # 0 255
print(np.min(y_train), np.max(y_train)) # 0 9
print(np.min(y_test), np.max(y_test))   # 0 9


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

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1 , 28, 28, 1)
print(x_train.shape, x_test.shape)
# (60000, 28, 28, 1) (10000, 28, 28, 1)

# exit()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)
# (60000, 10) (10000, 10)


# exit()

# 2.모델구성
model = Sequential()

# 블록 1: padding='same'으로 입력 크기 (28, 28)를 유지하여 외곽 정보 보존
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) # 크기 반감 -> (14, 14, 32)
model.add(Dropout(0.25))

# 블록 2: 첫 번째 레이어에서만 strides=(2, 2) 적용하여 반으로 줄임
model.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu')) # 크기 반감 -> (7, 7, 64)
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                 # 크기 유지 -> (7, 7, 64)
# model.add(MaxPooling2D()) <- 차원 붕괴를 막기 위해 삭제 완료!
model.add(Dropout(0.25))

# Fully Connected 분류층
model.add(Flatten()) # (7 * 7 * 64) = 3136 차원으로 정확히 펼쳐짐!
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

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
          epochs = 50,
          batch_size = 128,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es ],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ 39_fashion_model.evaluate --------------------")
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
!!!!!!!!!!! 목표 acc : 0.92

Epoch 35: early stopping
------------------ model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.2132 - acc: 0.9265
loss :  0.21319623291492462
acc :  0.9265000224113464
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9265
걸린 시간 :  56.71 sec

'''