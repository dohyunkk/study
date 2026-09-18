'''
2026-09-17 (목)
GlobalAveragePooling2D을 사용해보자.
Flatten() -> GlobalAveragePooling2D

!! 실습 만들어보기.
!! 목표 acc : 성능개선
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
model = Sequential()

# [블록 1] 입력 이미지: (32, 32, 3) -> (16, 16, 64)로 압축
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))

# [블록 2] 중수준 특성 추출 -> (8, 8, 128)로 압축
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.35))

# [블록 3] 고수준 특성 추출 -> (4, 4, 128)로 최종 압축
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.4))

# [분류기 진입]
# 4 * 4 * 128 = 2,048 노드 (과적합을 방지하는 가장 이상적인 크기)
model.add(GlobalAveragePooling2D()) 

model.add(Dense(128, activation='relu'))

# 최종 출력층 (100개 카테고리 다중 분류)
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
          epochs = 100,
          batch_size = 256,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ 40_cifar100_model.evaluate --------------------")
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
!! 목표 acc : 0.4

Epoch 87: early stopping
------------------ 39_cifar100_model.evaluate --------------------
313/313 [==============================] - 1s 3ms/step - loss: 2.0449 - acc: 0.4793
loss :  2.044857978820801
acc :  0.47929999232292175
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.4793
걸린 시간 :  652.64 sec

!! 목표 acc : 성능개선

Epoch 81: early stopping
------------------ 40_cifar100_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 1.8781 - acc: 0.5041
loss :  1.8780783414840698
acc :  0.5041000247001648
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.5041
걸린 시간 :  411.95 sec
'''