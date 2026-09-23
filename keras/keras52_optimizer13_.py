'''
2026-09-23 (수)

41 카피

지금까지 러닝레이트를 기본값으로 하고있었기 때문에
에포를 늘려 해결했다.
이제부터는 러닝레이트를 조절해서 모델 가동시간의 단축과, 성능 향상을 노려보자.

from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

optimizer=Adam(learning_rate=learning_rate)
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
x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)
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
# DNN 모델
model = Sequential()
model.add(Dense(3072, activation='relu', input_shape = (3072,)))
model.add(Dropout(0.2))

model.add(Dense(1512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(756, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))


model.add(Dense(10, activation='softmax'))

#CNN 모델
# # [앞단] 이미지의 기본 특징을 촘촘하게 추출 (드롭아웃 없이 정보 온전히 보존)
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3))) # (26, 26, 32)
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                          # (24, 24, 32)

# # [중간] 필터 수를 64개로 늘려 더 복잡한 숫자 패턴 추출
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))                          # (22, 22, 64)
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))                          # (20, 20, 64)
# model.add(MaxPooling2D())
# model.add(Dropout(0.25)) # 특징을 어느 정도 추출한 뒤 첫 드롭아웃 적용

# # [뒷단] 필터 수를 128개로 크게 늘려 핵심 특징 강조
# model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))                         # (18, 18, 128)
# model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))                         # (16, 16, 128)
# model.add(Dropout(0.25))

# # [분류기] 
# model.add(GlobalAveragePooling2D())

# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax')) 

model.summary()

# exit()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='categorical_crossentropy', 
              optimizer = Adam(learning_rate=learning_rate), 
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
          epochs = 500,
          batch_size = 128,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras52_learning_rate_cifar10 --------------------")
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

cnn
Epoch 31: early stopping
------------------ 40_cifar10_model.evaluate --------------------
313/313 [==============================] - 1s 4ms/step - loss: 0.6181 - acc: 0.7989
loss :  0.6181313395500183
acc :  0.7989000082015991
313/313 [==============================] - 1s 3ms/step
accuracy_score :  0.7989
걸린 시간 :  469.75 sec

dnn
Epoch 38: early stopping
------------------ 41_dnn_cifar10_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 1.4887 - acc: 0.4846
loss :  1.4887492656707764
acc :  0.4846000075340271
313/313 [==============================] - 1s 1ms/step
accuracy_score :  0.4846
걸린 시간 :  141.43 sec

poch 24: early stopping
------------------ keras52_learning_rate_cifar10 --------------------
313/313 [==============================] - 0s 1ms/step - loss: 1.9270 - acc: 0.2591
loss :  1.927004098892212
acc :  0.2590999901294708
313/313 [==============================] - 0s 838us/step
accuracy_score :  0.2591
걸린 시간 :  40.96 sec
'''