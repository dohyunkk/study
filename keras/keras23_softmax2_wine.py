'''
2026-09-09 (수) 다중분류

1. 데이터
OneHotEncoding

2. 모델구성
마지막 노드의 activation = 'softmax' :

4. 평가, 예측
np.argmax() :

# acc = 0.95 이상 뽑아보자.
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

#2. 모델

model = Sequential()
model.add(Dense(39, input_dim = 13, activation = 'relu'))
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
'''
