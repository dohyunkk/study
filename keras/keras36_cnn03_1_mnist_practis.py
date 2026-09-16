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
y_test = ohe.fittransform(y_test)

print(y_train.shape, y_test.shape)
# (60000, 10) (10000, 10)

# 2. 모델구성 (배운 레이어만 활용하여 최적화)
model = Sequential()

# [앞단] 이미지의 기본 특징을 촘촘하게 추출 (드롭아웃 없이 정보 온전히 보존)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # (26, 26, 32)
model.add(Conv2D(32, (3, 3), activation='relu'))                          # (24, 24, 32)

# [중간] 필터 수를 64개로 늘려 더 복잡한 숫자 패턴 추출
model.add(Conv2D(64, (3, 3), activation='relu'))                          # (22, 22, 64)
model.add(Conv2D(64, (3, 3), activation='relu'))                          # (20, 20, 64)
model.add(Dropout(0.25)) # 특징을 어느 정도 추출한 뒤 첫 드롭아웃 적용

# [뒷단] 필터 수를 128개로 크게 늘려 핵심 특징 강조
model.add(Conv2D(128, (3, 3), activation='relu'))                         # (18, 18, 128)
model.add(Conv2D(128, (3, 3), activation='relu'))                         # (16, 16, 128)
model.add(Dropout(0.25))

# [분류기] 1차원으로 펼친 후 정답 분류
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # 과적합을 막기 위해 Dense 층 뒤에 강력한 드롭아웃 배치
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', 
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=5, 
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

'''