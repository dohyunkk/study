'''
2026-09-17 (목)

DNN모델을 CNN모델로 변환
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

print(x_test)

print(np.min(x_train), np.max(x_train))
# 
print(np.min(x_test), np.max(x_test))
# 

x_train = x_train.reshape(-1, 54, 1, 1)
x_test = x_test.reshape(-1, 54, 1, 1)

# print(x_train.shape) # (331, 10, 1, 1)

# exit()

#2. 모델
model = Sequential()
model.add(Conv2D(128, (2, 1), padding = 'same', activation = 'relu', input_shape = (54, 1, 1)))
model.add(Conv2D(128, (2, 1), padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (2, 1), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (2, 1), padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu'))
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
          batch_size = 2500,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es, ]#mcp],
          )

end_time = time.time()

#.4 평가, 예측

print("==================== keras42_cnn_fetch_covtype ========================")

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
==================== keras31_fetch_covtype ========================
5447/5447 ━━━━━━━━━━━━━━━━━━━━ 4s 796us/step - acc: 0.9295 - loss: 0.1853 
loss :  0.1852918565273285
acc :  0.93
5447/5447 ━━━━━━━━━━━━━━━━━━━━ 3s 569us/step 
acc_score :  0.9295483752524325
걸린시간 :  143.15 sec

==================== keras42_cnn_fetch_covtype ========================
5447/5447 [==============================] - 8s 2ms/step - loss: 0.4557 - acc: 0.8104
loss :  0.4556868374347687
acc :  0.81
5447/5447 [==============================] - 5s 907us/step

'''