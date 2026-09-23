'''
2026-09-23 (수)

52-1 카피

한도에 닿으면 러닝레이트를 절반으로 낮춰서 이어가겠다.
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1, 
    factor=0.5,
)

이론상으로는 러닝레이트를 크게 잡아야 리듀스엘알이 성능이 좋아지는데,
돌려보면 막상 그렇지는 않더라.
'''


import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing


#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target


# print(x.shape, y.shape) 
# (20640, 8) (20640,)



'''
MinMaxScaler

원값 - Min
-----------
Max  - Min

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# print(x)
# [[0.53966842 0.78431373 0.0435123  ... 0.00149943 0.5674814  0.21115538]
#  [0.53802706 0.39215686 0.03822395 ... 0.00114074 0.565356   0.21215139]
#  [0.46602805 1.         0.05275646 ... 0.00169796 0.5642933  0.21015936]
#  ...
#  [0.08276438 0.31372549 0.03090386 ... 0.0013144  0.73219979 0.31175299]
#  [0.09429525 0.33333333 0.03178269 ... 0.0011515  0.73219979 0.30179283]
#  [0.13025338 0.29411765 0.03125246 ... 0.00154886 0.72582359 0.30976096]]

# print(np.min(x), np.max(x))
#       0.0      1.0000000000000002

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=9984,
)



# 2. 모델 구성
model = Sequential()
model.add(Dense(32, input_shape=(8,)))    # input_dim = 8  -> input_shape = (8, )
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))


# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

learning_rate = 0.002
# learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

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

start_time = time.time()     # 현재 시간을 반환. 시작시간

hist = model.fit(x_train, y_train, 
                 epochs=5000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es, rlr],
                 verbose=1,
                 )

end_time = time.time()       # 현재 시간을 반환. 종료시간

print("============== keras53_reducel_california ===================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# 평가(보조)지표-R2
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("r2 = : ", r2 )

mse = mean_squared_error(y_test, y_predict)
print('mse : ', mse)

#RMSE함수의 정의
def RMSE(y_tset, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

print("훈련시간 :", round(end_time - start_time, 2),"sec")


'''
결과
- loss: 0.7383 - val_loss: 0.6921
loss :  0.6298893094062805
r2 = :  0.511302339524152
===============================================
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 617us/step - loss: 0.5445
loss :  0.5444505214691162
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 634us/step
r2 = :  0.5854050960043469
mse :  0.544450453266494
RMSE :  0.737868859125044
훈련시간 : 7.76 sec


Epoch 94: early stopping
============== keras53_reducel_california ===================
162/162 [==============================] - 0s 982us/step - loss: 0.5364
loss :  0.5363773107528687
162/162 [==============================] - 0s 614us/step
r2 = :  0.5915528365039178
mse :  0.5363771748221665
RMSE :  0.7323777541830216
훈련시간 : 57.11 sec

'''


















