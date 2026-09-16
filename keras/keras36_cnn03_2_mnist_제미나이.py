'''
2026-09-16 (수)

#2 모델구성
Conv2D, Dropout, Flatten, Dense
'''

import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping


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

# 2. 모델구성 (배운 레이어만 활용하여 최적화)
# 기존 맨 위 import 부분에 MaxPooling2D와 BatchNormalization을 추가해야 합니다.
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization

# ==========================================
# 2. 모델구성 (0.995% 달성을 위한 최적화 구조)
# ==========================================
model = Sequential()

# [첫 번째 특징 추출 블록] 
# 이미지의 기본적인 선과 면을 학습합니다.
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2))) # 이미지 크기를 14x14로 압축
model.add(Dropout(0.25))

# [두 번째 특징 추출 블록] 
# 필터 수를 64개로 늘려 더 복잡하고 구체적인 숫자 패턴을 학습합니다.
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2))) # 이미지 크기를 7x7로 압축
model.add(Dropout(0.25))

# [최종 분류기 블록]
# 압축된 특징들을 하나로 펼쳐 최종 숫자가 무엇인지 판별합니다.
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))                     # 완전연결층에 강력한 드롭아웃 적용
model.add(Dense(10, activation='softmax'))  # 최종 10개 클래스 분류


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
          epochs = 100,
          batch_size = 128,
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
Epoch 100/100
375/375 [==============================] - 2s 6ms/step - loss: 0.0031 - acc: 0.9989 - val_loss: 0.0325 - val_acc: 0.9947
------------------ model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.0213 - acc: 0.9955 
loss :  0.021264325827360153
acc :  0.9955000281333923
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9955
걸린 시간 :  220.96 sec

'''