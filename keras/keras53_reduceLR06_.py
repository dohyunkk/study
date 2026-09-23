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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.datasets import load_breast_cancer    # 유방암 관련 데이터 셋 불러오기


#1. 데이터

datasets = load_breast_cancer()     #
# print(datasets.DESCR)             #
# print(datasets.feature_names)     #

x = datasets.data        # x = datasets['data']     #딕셔너리 형태의 데이터였구나
y = datasets.target

print(x.shape, y.shape)
# (569, 30) (569,)
print(type(x))           
# <class 'numpy.ndarray'> 넘파이로 구성되어있구나.

# print(y)      

### 0과 1의 개수가 몇개인지 찾아보기. -numpy            
print(np.unique(y))
# [0 1]
print(np.unique(y, return_counts=True))
# (array([0, 1]), array([212, 357])

### 0과 1의 개수가 몇개인지 찾아보기. -pandas
print(pd.DataFrame(y).value_counts())  
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

# train_test_나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.7, random_state = 7,
    stratify = y,         # 새로 추가된 부분 < 무조건 넣는건 아니고 분류모델에 넣으면 좋아질 수 있다.
)

print(np.unique(y_train, return_counts=True))  # y_train의 카테고리 확인
# (array([0, 1]), array([148, 250])
print(np.unique(y_test, return_counts=True))
# (array([0, 1]), array([ 64, 107])

print(x_train.shape, x_test.shape)
# (398, 30) (171, 30)
print(y_train.shape, y_test.shape)
# (398,) (171,)

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
# 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))
# -0.09792843691148767 2.5798045602605866


# exit()

# 2. 모델구성
model = Sequential()
model.add(Dense(90, input_shape=(30,), activation = 'relu'))
model.add(Dense(180, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))                  #★ activation= 'sigmoid'  디폴트는'Linear'
                                                             #★ 이진분류는 마지막 activation에 반드시 'sigmoid'

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009


model.compile(loss='binary_crossentropy',                    #★ 이진분류는 무조건 binary_crossentropy 
              optimizer=Adam(learning_rate=learning_rate),
              metrics=['acc'], # metrics=['accuracy']
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

hist = model.fit(x_train, y_train,
                 epochs = 1000,
                 batch_size = 32,
                 verbose = 1,
                 validation_split = 0.3,
                 callbacks = [es, rlr],
                 )

end_time = time.time()

#4. 평가, 예측
print("================ keras53_reducel_cancer ====================")
loss = model.evaluate(x_test, y_test)
print("====================================================")
print('loss : ', loss[0])
print('acc : ' , round(loss[1], 4))
print("====================================================")

y_pred = model.predict(x_test)
# print(y_pred[:10])
y_pred = np.round(y_pred)
# print(y_pred[:10])

# from sklearn.metrics import accuracy_score
# acc_score = accuracy_score(y_test, y_pred)
# print('acc_score : ', acc_score)

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_pred)
# print("r2 = : ", r2 )

# mse = mean_squared_error(y_test, y_pred)
# print('r2 : ', r2)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test, y_pred)
# print('RMSE : ', rmse)


print("훈련시간 : ", round(end_time - start_time, 2),"sec")



'''
standardscaler
Epoch 24/1000
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - acc: 1.0000 - loss: 0.0018 - val_acc: 0.9750 - val_loss: 0.2672
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - acc: 0.9766 - loss: 0.0650 
====================================================
loss :  0.06504502892494202
acc :  0.9766
====================================================
6/6 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step 
acc_score :  0.9766081871345029
훈련시간 :  2.4 sec

---------------------------------------------------------
Epoch 37/10000
12/12 [==============================] - 0s 27ms/step - loss: 0.0814 - acc: 0.9827 - val_loss: 0.4691 - val_acc: 0.8878
============== keras52_learning_rate__santander ======================
1875/1875 [==============================] - 3s 1ms/step - loss: 0.2482 - acc: 0.9103
loss :  0.24819093942642212
acc :  0.91
1875/1875 [==============================] - 2s 956us/step
============================================================
acc_score :  0.9103333333333333
걸린시간 :  13.29 sec
============================================================

----------------------------------------------------------------------------
Epoch 43: early stopping
================ keras53_reducel_cancer ====================
6/6 [==============================] - 0s 800us/step - loss: 0.1306 - acc: 0.9649
====================================================
loss :  0.13060595095157623
acc :  0.9649
====================================================
6/6 [==============================] - 0s 0s/step
acc_score :  0.9649122807017544
훈련시간 :  2.68 sec
'''