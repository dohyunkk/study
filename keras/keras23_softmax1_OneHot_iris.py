'''
2026-09-09
1. 데이터
OneHotEncoding : 범주형 데이터를 컴퓨터가 이해할 수 있도록 0과 1의 조합인 이진 형태의 숫자 벡터로 변환하는 전처리 기법

reshape : 데이터 전처리
reshape의 조건 
-1. 내용물이 바뀌지 않을 것
-2. 순서가 바뀌지 않을 것

2. 모델구성
마지막 노드의 activation = 'softmax' : 여러 개의 입력값을 0과 1 사이의 값으로 정규화하여, 
전체 출력값의 합이 항상 1(확률 분포)이 되도록 만들어주는 활성화 함수

4. 평가, 예측
np.argmax() : 배열에서 가장 큰 값이 있는 위치의 인덱스(index)를 반환
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris       # 아이리스 데이터 셋 불러오기


#1. 데이터
datasets = load_iris()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


x = datasets.data
y = datasets['target']

# print(x.shape, y.shape)
# (150, 4) (150,)
# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
# print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50]))


'''
다중분류 데이터를
★ OneHotEncoding ★을 거쳐 위치값을 변경한다.

[0,0,1,1,2]    # (5,)

-OneHotEncoding->

[[1,0,0]
 [1,0,0]
 [0,1,0]
 [0,1,0]
 [0,0,1]]      # (5,3)

 왜 onehot을 train_test 보다 먼저 했을까 ?
 train_test 후에 onehot을 하면 뒤에 train_test를 또 해줘야하기 때문( 번거롭기 때문 )
'''

######################## onehot1. tensorflow_to_categorical ########################
# from tensorflow.keras.utils import to_categorical       # ★ OneHotEncoding ★
# y = to_categorical(y)                                   # ★ OneHotEncoding ★

# print(y)
# [[1. 0. 0.]
#  ...
#  [0. 0. 1.]]
# print(y.shape)
# (150, 3)

######################## onehot2. pandas_get_dummies ########################
# y = pd.get_dummies(y, dtype=int)
# print(y)
#      0  1  2
# 0    1  0  0
# 1    1  0  0
# 2    1  0  0
# 3    1  0  0
# 4    1  0  0
# ..  .. .. ..
# 145  0  0  1
# 146  0  0  1
# 147  0  0  1
# 148  0  0  1
# 149  0  0  1

# [150 rows x 3 columns]

# exit()
######################## onehot3. sklearn ########################
'''
reshape의 조건 
1. 내용물(값)이 바뀌지 않을 것
2. 순서가 바뀌지 않을 것

[1, 2, 3]
# (3,)
-reshape->
[[1],[2],[3]]
# (3,1)
'''
from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(150, 1) #  (150, 1)
y = y.reshape(-1, 1)    # (150, 1)

# print(y, y.shape)

# exit()

# ohe = OneHotEncoder()   # sparse 형태로 나온다.
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y)  # 
# print(y)

# exit()

# train_test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.8,
    random_state=9978,
    stratify=y,                                # y_train과 y_predict의 값이 치우치지 않게 해줌.
    shuffle=True,                              # y_train과 y_predict의 값이 치우치지 않게 해줌.
)

# print(x_train.shape, x_test.shape)
# (120, 4) (30, 4)
# print(y_train.shape, y_test.shape)
# (120, 3) (30, 3)


# exit()

#2. 모델구성

model = Sequential()
model.add(Dense(16, input_dim = 4, activation='relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))          #★ 다중분류는 activation = 'softmax'

#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy',       #★ 다중분류는 loss='categorical_crossentropy'를 사용
              optimizer='adam',
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 20,
    restore_best_weights = True,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs = 10000,
                 batch_size = 8,
                 verbose = 1,
                 validation_split = 0.2,
                 callbacks = [es],
                 )

end_time = time.time()


# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', round(result[1],2))

y_predict = model.predict(x_test)

# print(y_predict)
# [[3.4872625e-07 4.6985470e-02 9.5301419e-01]
#  [1.5822366e-04 9.8547149e-01 1.4370291e-02]
#  [1.9652845e-04 9.9685776e-01 2.9457449e-03]
#  ...
#  [9.9984348e-01 1.5593825e-04 6.4264015e-07]
#  [2.0610716e-07 7.5146407e-02 9.2485338e-01]
#  [1.8125295e-06 1.7040132e-01 8.2959688e-01]]
# print(y_predict.shape)
# (30, 3)

y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
# [2 1 1 2 1 1 1 1 2 0 0 0 0 1 1 1 2 1 1 0 1 0 0 0 2 2 0 0 2 2]

y_test = np.argmax(y_test, axis=1)
# print(y_test)
# [2 1 1 2 1 1 1 1 2 0 0 0 0 2 1 1 2 1 1 0 2 0 0 0 2 2 0 0 2 2]
# exit()

accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score : ', accuracy_score)
print('걸린시간 : ', round(end_time - start_time, 2),'sec')

'''
12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - acc: 0.9479 - loss: 0.0919 - val_acc: 1.0000 - val_loss: 0.0336
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 111ms/step - acc: 1.0000 - loss: 0.0299
걸린시간 :  7.54 sec
'''

