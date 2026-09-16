'''
2026-09-16 (수)

!! 실습 만들어보기.
!! 목표 acc : 0.67
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

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# plt.imshow(x_train[10], )
# plt.show()

# print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)

# print(np.min(x_train), np.max(x_train)) # 0 255
# print(np.min(x_test), np.max(x_test))   # 0 255
# print(np.min(y_train), np.max(y_train)) # 0 9
# print(np.min(y_test), np.max(y_test))   # 0 9

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

# print(y_train.shape, y_test.shape)
# (50000, 10) (10000, 10)

# exit()

# 2. 모델구성 
model = Sequential()

# [앞단] 이미지의 기본 특징을 촘촘하게 추출 (드롭아웃 없이 정보 온전히 보존)
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3))) # (26, 26, 32)
model.add(Conv2D(64, (3, 3), activation='relu'))                          # (24, 24, 32)

# [중간] 필터 수를 64개로 늘려 더 복잡한 숫자 패턴 추출
model.add(Conv2D(128, (3, 3), activation='relu'))                          # (22, 22, 64)
model.add(Conv2D(128, (3, 3), activation='relu'))                          # (20, 20, 64)
model.add(Dropout(0.25)) # 특징을 어느 정도 추출한 뒤 첫 드롭아웃 적용

# [뒷단] 필터 수를 128개로 크게 늘려 핵심 특징 강조
model.add(Conv2D(256, (3, 3), activation='relu'))                         # (18, 18, 128)
model.add(Conv2D(256, (3, 3), activation='relu'))                         # (16, 16, 128)
model.add(Dropout(0.25))

# [분류기] 1차원으로 펼친 후 정답 분류
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # 과적합을 막기 위해 Dense 층 뒤에 강력한 드롭아웃 배치
model.add(Dense(10, activation='softmax'))

# model.summary()

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
          batch_size = 32,
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
Epoch 10/10
313/313 [==============================] - 18s 57ms/step - loss: 1.1333 - acc: 0.5824 - val_loss: 0.9394 - val_acc: 0.6641
------------------ model.evaluate --------------------
313/313 [==============================] - 2s 4ms/step - loss: 0.9549 - acc: 0.6605
loss :  0.9549481272697449
acc :  0.6604999899864197
313/313 [==============================] - 1s 4ms/step
accuracy_score :  0.6605
걸린 시간 :  182.48 sec

Epoch 34: early stopping
------------------ model.evaluate --------------------
313/313 [==============================] - 2s 4ms/step - loss: 0.9051 - acc: 0.6932
loss :  0.9051247239112854
acc :  0.6931999921798706
313/313 [==============================] - 1s 4ms/step
accuracy_score :  0.6932
걸린 시간 :  605.58 sec
PS C:\study> 
'''