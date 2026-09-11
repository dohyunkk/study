'''
2026-09-09 (수) 다중분류

1. 데이터
OneHotEncoding

2. 모델구성
마지막 노드의 activation = 'softmax' :

4. 평가, 예측
np.argmax() :

### acc : 0.93 이상 뽑아보자.
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from sklearn.datasets import fetch_covtype

#1. 데이터
datasets = fetch_covtype()


# print(datasets)
# shape=(581012, 54)

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data      
y = datasets['target']

# print(x.shape, y.shape)
# (581012, 54) (581012,)

# print(y)
# [5 5 2 ... 3 3 3]

# print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), 
#  array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

####### to_categorical 활용 ######

'''
from tensorflow.keras.utils import to_categorical       # ★ OneHotEncoding ★
y_oh = to_categorical(y)                                # ★ OneHotEncoding ★
print(y_oh[:10])
# [[0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]]
print(y[:10])
# [5 5 2 2 5 2 5 5 5 5]
print(y.shape)
# (581012,)
print(y_oh.shape)
# (581012, 8)
'''
# from tensorflow.keras.utils import to_categorical       # ★ OneHotEncoding ★
# y = to_categorical(y)  
 
# print(y)
# [[0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]]
# print(y.shape)
# (581012, 8)


# exit()


####### pd.get_dummies 활용 ######
y = pd.get_dummies(y, dtype=int)

# print(y)
#         1  2  3  4  5  6  7
# 0       0  0  0  0  1  0  0
# 1       0  0  0  0  1  0  0
# 2       0  1  0  0  0  0  0
# 3       0  1  0  0  0  0  0
# 4       0  0  0  0  1  0  0
# ...    .. .. .. .. .. .. ..
# 581007  0  0  1  0  0  0  0
# 581008  0  0  1  0  0  0  0
# 581009  0  0  1  0  0  0  0
# 581010  0  0  1  0  0  0  0
# 581011  0  0  1  0  0  0  0

# [581012 rows x 7 columns]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7,
    random_state = 5252,
    stratify = y,
    shuffle = True,
)

# print(x_train.shape, x_test.shape)
# (406708, 54) (174304, 54)
# print(y_train.shape, y_test.shape)
# (406708, 7) (174304, 7)

#2. 모델

model = Sequential()
model.add(Dense(78, input_dim = 54, activation = 'relu'))
model.add(Dense(126, activation = 'relu'))
model.add(Dense(156, activation = 'relu'))
model.add(Dense(78, activation = 'relu'))
model.add(Dense(39, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

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
          batch_size = 5000,
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

# print(y_predict.shape)
# (174304, 7)

y_predict = np.argmax(y_predict, axis=1)

y_test = np.argmax(y_test, axis=1)


accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score : ', accuracy_score)
print('걸린시간 : ', round(end_time - start_time, 2),'sec')


'''
66/66 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - acc: 0.9218 - loss: 0.1994 - val_acc: 0.9004 - val_loss: 0.2616
5447/5447 ━━━━━━━━━━━━━━━━━━━━ 4s 710us/step - acc: 0.9048 - loss: 0.2437 
loss :  0.24370771646499634
acc :  0.9
5447/5447 ━━━━━━━━━━━━━━━━━━━━ 3s 476us/step 
acc_score :  0.9047813016339269
걸린시간 :  266.17 sec
'''