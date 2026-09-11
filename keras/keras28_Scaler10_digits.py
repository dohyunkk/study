'''
2026-09-10 (목) 

'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
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

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
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

model = Sequential()
model.add(Dense(640, input_shape = (64,), activation = 'relu'))
model.add(Dense(320, activation = 'relu'))
model.add(Dense(504, activation = 'relu'))
model.add(Dense(700, activation = 'relu'))
model.add(Dense(350, activation = 'relu'))
model.add(Dense(130, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 50,
    restore_best_weights = True,
)

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 10000,
          batch_size = 32,
          verbose = 1,
          validation_split = 0.3,
          callbacks = [es],
          )

end_time = time.time()

#.4 평가, 예측

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
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step - acc: 0.9495 - loss: 0.1226 - val_acc: 0.9200 - val_loss: 0.2319
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - acc: 0.9630 - loss: 0.1651
loss :  0.1650628000497818
acc :  0.96
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step
acc_score :  0.9629629629629629

======================================================================

Epoch 59/10000
32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - acc: 1.0000 - loss: 3.5371e-07 - val_acc: 0.9676 - val_loss: 0.1815
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - acc: 0.9861 - loss: 0.0650
loss :  0.06498773396015167
acc :  0.99
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step 
acc_score :  0.9861111111111112
걸린시간 :  11.43 sec


'''