'''
2026-09-16 (수)

!! 실습 만들어보기.
!! acc : 0.92
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

from tensorflow.keras.datasets import fashion_mnist


# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# plt.imshow(x_train[1], 'gray')
# plt.show()

# print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

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
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# print(x_train.shape, x_test.shape)
# (60000, 28, 28, 1) (10000, 28, 28, 1)

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)
# (60000, 10) (10000, 10)

# 2. 모델구성 (배운 레이어만 활용하여 최적화)
model = Sequential()

# [앞단] 이미지의 기본 특징을 촘촘하게 추출 (드롭아웃 없이 정보 온전히 보존)
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))) # (26, 26, 32)
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
    patience=25, 
    restore_best_weights=True, 
    verbose=1
)

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 100,
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
Epoch 12/100
375/375 [==============================] - ETA: 0s - loss: 0.1559 - acc: 0.9410Restoring model weights from the end of the best epoch: 7.
375/375 [==============================] - 6s 15ms/step - loss: 0.1559 - acc: 0.9410 - val_loss: 0.2660 - val_acc: 0.9194
Epoch 12: early stopping
------------------ model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.2563 - acc: 0.9084
loss :  0.25627705454826355
acc :  0.9083999991416931
313/313 [==============================] - 1s 1ms/step
accuracy_score :  0.9084
걸린 시간 :  71.98 sec

Epoch 23/100
1499/1500 [============================>.] - ETA: 0s - loss: 0.1176 - acc: 0.9562Restoring model weights from the end of the best epoch: 8.
1500/1500 [==============================] - 8s 5ms/step - loss: 0.1176 - acc: 0.9561 - val_loss: 0.3227 - val_acc: 0.9158
Epoch 23: early stopping
------------------ model.evaluate --------------------
313/313 [==============================] - 1s 3ms/step - loss: 0.2584 - acc: 0.9111
loss :  0.2583582401275635
acc :  0.9110999703407288
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.9111
걸린 시간 :  191.49 sec

'''
