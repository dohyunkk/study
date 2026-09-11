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

from sklearn.datasets import load_wine


#1. 데이터
datasets = load_wine()

# print(datasets)
# shape=(178, 13)

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)
# (178, 13) (178,)

# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

# print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48]))

y = pd.get_dummies(y, dtype=int)
# print(y)
#      0  1  2
# 0    1  0  0
# 1    1  0  0
# 2    1  0  0
# 3    1  0  0
# 4    1  0  0
# ..  .. .. ..
# 173  0  0  1
# 174  0  0  1
# 175  0  0  1
# 176  0  0  1
# 177  0  0  1

# [178 rows x 3 columns]


# exit()
# train_test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7,
    random_state = 5252,
    stratify = y,
    shuffle = True,
)

# print(x_train.shape, x_test.shape)
# (124, 13) (54, 13)
# print(y_train.shape, y_test.shape)
# (124, 3) (54, 3)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

print(x_test)

print(np.min(x_train), np.max(x_train))
# 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))
# -0.08032267285709349 1.0867433267723332

# exit()

#2. 모델

model = Sequential()
model.add(Dense(39, input_shape = (13,), activation = 'relu'))
model.add(Dense(78, activation = 'relu'))
model.add(Dense(156, activation = 'relu'))
model.add(Dense(78, activation = 'relu'))
model.add(Dense(39, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 30,
    restore_best_weights = True,
)

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 1000,
          batch_size = 64,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es],
          )

end_time = time.time()

#.4 평가, 예측

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', round(result[1],2))

y_predict = model.predict(x_test)

# print(y_predict)
# [[0.25198266 0.5241955  0.22382186]
#  [0.10012859 0.72903174 0.17083973]
#  ...
#   [0.03851363 0.81407875 0.1474076 ]
#  [0.05151741 0.77465177 0.17383084]]

# print(y_predict.shape)
# (54, 3)

y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
# [2 1 1 1 1 0 1 1 0 0 1 0 2 0 0 1 1 1 2 0 0 2 1 0 1 1 1 2 0 0 1 0 1 1 0 1 1
#  0 0 1 1 1 2 1 2 0 0 0 2 1 1 1 1 1]
y_test = np.argmax(y_test, axis=1)
# print(y_test)
# [2 0 2 2 1 0 2 1 0 0 1 0 2 0 0 1 1 1 1 0 0 1 1 0 2 1 2 1 0 0 1 2 1 1 0 1 2
#  0 0 1 1 2 2 2 2 0 0 0 1 1 1 2 2 1]

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

=================================================================

Epoch 67/1000
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - acc: 1.0000 - loss: 7.6167e-04 - val_acc: 1.0000 - val_loss: 0.0661
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step - acc: 0.9815 - loss: 0.1035
loss :  0.10349512100219727
acc :  0.98
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step
acc_score :  0.9814814814814815
걸린시간 :  4.82 sec


'''
