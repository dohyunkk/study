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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

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
# print(np.min(x_train), np.max(x_train))   # 0.0 1.0
# print(np.min(x_test), np.max(x_test))     # 0.0 1.0

############### 스케일링 2
# x_train = (x_train - 127.5)/127.5             
# x_test = (x_test - 127.5)/127.5                
# print(np.min(x_train), np.max(x_train))   # -1.0 1.0
# print(np.min(x_test), np.max(x_test))     # -1.0 1.0

x_train = x_train.reshape(-1, 28 * 28 * 1)    # ★
x_test = x_test.reshape(-1 , 28 * 28 * 1)      # ★
print(x_train.shape, x_test.shape)
# (60000, 784) (10000, 784)

# exit()

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)
# (60000, 10) (10000, 10)



# 2.모델구성
# DNN 모델
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape = (784,)))
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

# exit()


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

start_time = time.time()

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

model.fit(x_train, y_train,
          epochs = 500,
          batch_size = 128,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es ],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras52_learning_rate_mnist --------------------")
loss = model.evaluate(x_test, y_test, verbose = 1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis = 1)#.reshape(-1, 1)
y_test_labels = np.argmax(y_test, axis = 1)#.reshape(-1, 1)

acc_score = accuracy_score(y_test_labels, y_predict)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')


'''
cnn
Epoch 35: early stopping
------------------ 39_mnist_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.0204 - acc: 0.9940
loss :  0.020386828109622
acc :  0.9940000176429749
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.994
걸린 시간 :  60.83 sec

dnn
Epoch 27: early stopping
------------------ 41_dnn_mnist_model.evaluate --------------------
313/313 [==============================] - 1s 3ms/step - loss: 0.1006 - acc: 0.9804
loss :  0.10059752315282822
acc :  0.980400025844574
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.9804
걸린 시간 :  42.13 sec


Epoch 26: early stopping
------------------ keras52_learning_rate_mnist --------------------
313/313 [==============================] - 0s 1ms/step - loss: 0.1444 - acc: 0.9710
loss :  0.14438049495220184
acc :  0.9710000157356262
313/313 [==============================] - 0s 812us/step
accuracy_score :  0.971
걸린 시간 :  22.62 sec
'''