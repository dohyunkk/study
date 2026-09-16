'''
2026-09-16 (수)

!! 실습 만들어보기.
!! 목표 acc : 0.4
'''

import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

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
x_train = x_train.reshape(-1, 32, 32, 3)
x_test = x_test.reshape(-1, 32, 32, 3)
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

# 2. 모델구성 
# 2. 모델구성 (다이어트 버전)
# 2. 모델구성 (목표 0.4 달성용 초다이어트 & 깊은 모델)
model = Sequential()

# [앞단] 이미지 크기: 32 -> 30 -> 28
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) 
model.add(Conv2D(32, (3, 3), activation='relu'))                          
model.add(Dropout(0.25)) 

# [중간 1] 이미지 크기: 28 -> 26 -> 24
model.add(Conv2D(64, (3, 3), activation='relu'))                          
model.add(Conv2D(64, (3, 3), activation='relu'))                          
model.add(Dropout(0.25)) 

# [중간 2] 이미지 크기: 24 -> 22 -> 20
model.add(Conv2D(128, (3, 3), activation='relu'))                         
model.add(Conv2D(128, (3, 3), activation='relu'))                         
model.add(Dropout(0.3))

# [중간 3] 이미지 크기: 20 -> 18 -> 16
model.add(Conv2D(128, (3, 3), activation='relu'))                         
model.add(Conv2D(128, (3, 3), activation='relu'))                         
model.add(Dropout(0.3))

# [뒷단] 층을 한 세트 더 추가! 이미지 크기: 16 -> 14 -> 12
model.add(Conv2D(256, (3, 3), activation='relu'))                         
model.add(Conv2D(256, (3, 3), activation='relu'))                         
model.add(Dropout(0.4)) # 뒤로 갈수록 드롭아웃을 강하게 주어 과적합 방지

# [분류기] 
model.add(Flatten()) # 12 * 12 * 256 = 36,864 노드로 대폭 축소! (이전의 절반 수준)
model.add(Dense(256, activation='relu')) # 100개 클래스 분류를 위해 뇌 용량 증가
model.add(Dropout(0.5))  
model.add(Dense(100, activation='softmax'))



model.summary()

# exit()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', 
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 1000,
          batch_size = 256,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es],
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
Epoch 38/1000
156/157 [============================>.] - ETA: 0s - loss: 1.9361 - acc: 0.4449Restoring model weights from the end of the best epoch: 23.
157/157 [==============================] - 8s 49ms/step - loss: 1.9368 - acc: 0.4448 - val_loss: 2.9610 - val_acc: 0.3146
Epoch 38: early stopping
------------------ model.evaluate --------------------
313/313 [==============================] - 1s 3ms/step - loss: 2.7172 - acc: 0.3232
loss :  2.71718168258667
acc :  0.323199987411499
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.3232
걸린 시간 :  298.77 sec

'''