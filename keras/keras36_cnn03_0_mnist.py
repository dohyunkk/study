'''
2026-09-15 (화)
2026-09-16 (수) 이어서.

#1.데이터
이미지 데이터의 x값은 거의 255이기 때문에, 스케일러를 부르지 않아도 입력할 수 있다.
'''

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score

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
print(np.min(x_train), np.max(x_train))   # 0.0 1.0
print(np.min(x_test), np.max(x_test))     # 0.0 1.0

############### 스케일링 2
# x_train = (x_train - 127.5)/127.5             
# x_test = (x_test - 127.5)/127.5                
# print(np.min(x_train), np.max(x_train))   # -1.0 1.0
# print(np.min(x_test), np.max(x_test))     # -1.0 1.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1 , 28, 28, 1)
# print(x_train.shape, x_test.shape)
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



# 2.모델구성
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape =(28, 28, 1), activation = 'relu'))  # (24, 26, 64)
model.add(Conv2D(filters = 32, kernel_size=(3, 3), activation = 'relu'))      # (24, 24, 32)
model.add(Conv2D(32, (2, 2), activation='relu'))  # (23, 23, 32)
model.add(Conv2D(16, (2, 2), activation='relu'))  # (22, 22, 16)
model.add(Dropout(0.2))
model.add(Conv2D(16, (2, 2), activation='relu'))  # (21, 21, 16)
model.add(Dropout(0.2))
model.add(Conv2D(16, (2, 2), activation='relu'))  # (20, 20, 16)
model.add(Flatten())                              # (6400, ) # 대문자(F)latten에 괄호() 넣은거 보니까 클래스 같다.

model.add(Dense(units=32, activation='relu'))     # (32, )
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape=(32,), activation = 'relu')) # (16, )
model.add(Dense(10, activation='softmax'))                         # (10, )

model.summary()

# exit()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', 
              metrics=['acc'],
              )

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 50,
          batch_size = 128,
          verbose = 1,
          validation_split = 0.2,
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ model.evaluate --------------------")
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
!!!!!!!!! 목표 acc : 0.995

CPU
Epoch 50/50
375/375 ━━━━━━━━━━━━━━━━━━━━ 11s 30ms/step - acc: 0.9931 - loss: 0.0214 - val_acc: 0.9888 - val_loss: 0.0549
------------------ model.evaluate --------------------
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - acc: 0.9893 - loss: 0.0470 
loss :  0.04704029858112335
acc :  0.989300012588501
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step  
accuracy_score :  0.9893
걸린 시간 :  556.1 sec

GPU
Epoch 50/50
375/375 [==============================] - 2s 6ms/step - loss: 0.0183 - acc: 0.9935 - val_loss: 0.0592 - val_acc: 0.9881
------------------ model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.0484 - acc: 0.9900
loss :  0.04839156195521355
acc :  0.9900000095367432
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.99
걸린 시간 :  140.82 sec

'''