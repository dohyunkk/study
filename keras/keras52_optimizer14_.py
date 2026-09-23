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
x_train = x_train.reshape(-1, 32 * 32 * 3)
x_test = x_test.reshape(-1, 32 * 32 * 3)
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
# DNN 모델
model = Sequential()
model.add(Dense(3072, activation='relu', input_shape = (3072, )))
model.add(Dropout(0.2))

model.add(Dense(1534, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(712, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(356, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(100, activation='softmax'))


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
          epochs = 100,
          batch_size = 256,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras52_learning_rate_cifar100 --------------------")
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
Epoch 81: early stopping
------------------ 40_cifar100_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 1.8781 - acc: 0.5041
loss :  1.8780783414840698
acc :  0.5041000247001648
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.5041
걸린 시간 :  411.95 sec

dnn
Epoch 63: early stopping
------------------ 41_dnn_cifar100_model.evaluate --------------------
313/313 [==============================] - 0s 1ms/step - loss: 3.3134 - acc: 0.2119
loss :  3.313413619995117
acc :  0.211899995803833
313/313 [==============================] - 0s 779us/step
accuracy_score :  0.2119
걸린 시간 :  69.41 sec

'''