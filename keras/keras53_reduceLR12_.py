'''
2026-09-23 (수)

52 카피

한도에 닿으면 러닝레이트를 절반으로 낮춰서 이어가겠다.
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1, 
    factor=0.5,
)
'''

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
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

x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1 , 28 * 28 * 1)
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
# DNN 모델
model = Sequential()
model.add(Dense(512, activation='relu', input_shape = (784,)))
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

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

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(         
    monitor = 'val_loss',        
    mode = 'auto',                 
    patience = 40,                
    verbose = 1,
    restore_best_weights = True,  
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1, 
    factor=0.5,
)

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 500,
          batch_size = 128,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es, rlr],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras53_reducel_fashion --------------------")
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
Epoch 75: early stopping
------------------ 40_fashion_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.2131 - acc: 0.9280
loss :  0.2131367176771164
acc :  0.9279999732971191
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.928
걸린 시간 :  114.16 sec

dnn
Epoch 30: early stopping
------------------ 41_dnn_fashion_model.evaluate --------------------
313/313 [==============================] - 0s 1ms/step - loss: 0.3474 - acc: 0.8843
loss :  0.3474321663379669
acc :  0.8842999935150146
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.8843
걸린 시간 :  46.28 sec

Epoch 44: early stopping
------------------ keras52_learning_rate__fashion --------------------
313/313 [==============================] - 0s 1ms/step - loss: 0.4343 - acc: 0.8475
loss :  0.4343332052230835
acc :  0.8475000262260437
313/313 [==============================] - 0s 842us/step
accuracy_score :  0.8475
걸린 시간 :  39.04 sec

Epoch 118: early stopping
------------------ keras53_reducel_fashion --------------------
313/313 [==============================] - 0s 1ms/step - loss: 0.4235 - acc: 0.8646
loss :  0.42349159717559814
acc :  0.8646000027656555
313/313 [==============================] - 0s 830us/step
accuracy_score :  0.8646
걸린 시간 :  170.31 sec
'''