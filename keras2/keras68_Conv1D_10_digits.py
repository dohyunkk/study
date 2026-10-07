'''
53 카피
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_digits

#1. 데이터
datasets = load_digits()

# print(datasets)
# shape=(1797, 64)

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets['target']

# print(x.shape, y.shape)
# (1797, 64) (1797,)

# print(y)
# [0 1 2 ... 8 9 8]

# print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
#  array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

y = pd.get_dummies(y, dtype=int)
# print(y)
#       0  1  2  3  4  5  6  7  8  9
# 0     1  0  0  0  0  0  0  0  0  0
# 1     0  1  0  0  0  0  0  0  0  0
# ...  .. .. .. .. .. .. .. .. .. ..
# 1795  0  0  0  0  0  0  0  0  0  1
# 1796  0  0  0  0  0  0  0  0  1  0

# [1797 rows x 10 columns]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.8,
    random_state = 52525,
    stratify = y,
    shuffle = True,
)


# print(x_train.shape, x_test.shape)
# (1257, 64) (540, 64)
# print(y_train.shape, y_test.shape)
# (1257, 10) (540, 10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

print(x_test)

print(np.min(x_train), np.max(x_train))
# 
print(np.min(x_test), np.max(x_test))
# 

# exit()

#2. 모델
from tensorflow.keras.layers import Reshape, Conv1D, MaxPool1D, Flatten

model = Sequential()
model.add(Reshape(target_shape=(64, 1), input_shape=(64, )))

model.add(Conv1D(filters=32, kernel_size=2, padding='same',input_shape=(64, 1)))

model.add(Conv1D(64, 2, padding='same'))
model.add(MaxPool1D())

model.add(Conv1D(128, 2, padding='same'))
model.add(MaxPool1D())

model.add(Conv1D(256, 2, padding='same'))
model.add(MaxPool1D())

# model.add(GlobalAveragePooling1D())
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()
# exit()

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
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
    patience=10,
    verbose=1, 
    factor=0.5,
)

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 10000,
          batch_size = 32,
          verbose = 1,
          validation_split = 0.3,
          callbacks = [es, rlr],
          )

end_time = time.time()

#.4 평가, 예측

print("================ keras68_Conv1D_digits =======================")

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', round(result[1],2))

y_predict = model.predict(x_test)

# print(y_predict)
# [[6.2782107e-08 1.0113516e-03 3.1124256e-11 ... 3.6221562e-14
#   4.0224128e-05 1.3369295e-09]
# ...
#  [3.5025188e-04 4.1906126e-03 3.0809257e-08 ... 5.5959754e-07
#   9.5695919e-05 9.8536956e-01]]

# print(y_predict.shape)
# (540, 10)


y_predict = np.argmax(y_predict, axis=1)

y_test = np.argmax(y_test, axis=1)


accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score : ', accuracy_score)
print('걸린시간 : ', round(end_time - start_time, 2),'sec')

'''
minmaxscaler
Epoch 59/10000
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - acc: 1.0000 - loss: 3.5371e-07 - val_acc: 0.9676 - val_loss: 0.1815
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - acc: 0.9861 - loss: 0.0650
loss :  0.06498773396015167
acc :  0.99
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
acc_score :  0.9861111111111112
걸린시간 :  11.43 sec

---------------------------------------------------------------------

Epoch 51: early stopping
================ keras53_reducel_digits =======================
12/12 [==============================] - 0s 1ms/step - loss: 0.1199 - acc: 0.9722
loss :  0.11986266076564789
acc :  0.97
12/12 [==============================] - 0s 987us/step
acc_score :  0.9722222222222222
걸린시간 :  5.49 sec
------------------------------------------------------------------
Epoch 53: early stopping
================ keras68_Conv1D_digits =======================
12/12 [==============================] - 0s 7ms/step - loss: 0.0398 - acc: 0.9806
loss :  0.039842523634433746
acc :  0.98
12/12 [==============================] - 0s 1ms/step
acc_score :  0.9805555555555555
걸린시간 :  10.3 sec

'''

'''
[실험 기록]

스케일러: RobustScaler
모델: 64(입력) → 128 → 256 → 512 → 64 → 10(출력)
처음 학습률: 0.001
최대 에포크: 10000
batch_size: 32
validation_split: 0.3
EarlyStopping patience: 40
ReduceLROnPlateau patience: 20, factor: 0.5

가장 좋았던 에포크: 11
학습 종료: 51에포크
테스트 loss: 0.1199
테스트 정확도: 0.9722 (97.22%)
걸린 시간: 5.49초

'''