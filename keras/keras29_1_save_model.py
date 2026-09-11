'''
2026-09-11 (금)
keras28_1  카피

#2. 모델

path = './_save/keras29/'
model.save(path + 'keras29_1_save_model.keras')

저장하는 위치에 따라 훈련 모델만 가져올지, 훈련이 완료된 가중치까지 가져올지 선택할 수 있다.
1. #2 모델만 가져올래.
2. #3 컴파일, 훈련까지 가져올래.

'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target


# print(x.shape, y.shape) 
# (20640, 8) (20640,)



x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.75,
    # test_size = 0.25,
    # shuffle = True
    random_state=20260910,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

print(x_train)

print(x_test)

print(np.min(x_train), np.max(x_train))
# 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))
# 0.0001681661481543765 1.0711971933203288

# exit()

# 2. 모델 구성
model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape=(8,)))    # input_dim = 8  -> input_shape = (8, )
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32))
model.add(Dense(1))

model.summary()

path = './_save/keras29/'
model.save(path + 'keras29_1_save_model.keras')


exit()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(              
    monitor = 'val_loss',        
    mode = 'min',                 
    patience = 30,                
    restore_best_weights = True,  
)


start_time = time.time()    
hist = model.fit(x_train, y_train, 
                 epochs=50000,
                 batch_size=600, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()      

print("===================keras29_california==========================")


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

# print("============================ history =================================")
# print(hist)
# print("========================= hist.histoy ==============================")
# print(hist.history)
# print("============================= loss =================================")
# print(hist.history['loss'])
# print("=========================== val_loss =================================")
# print(hist.history['val_loss'])
# print("============================ history =================================")

import matplotlib.pyplot as plt

# print(plt.rcParams['font.family'])         # 그래프 출력시 ['sans-serif']폰트 가 디폴트이기 때문에 한글이 출력되지 않는다.

plt.rcParams['font.family'] = 'Malgun Gothic'   # 그래서 맷플롯립 폰트를 한글을 지원하는 폰트로 삽입하여 그래프에 한글이 표시되도록 한다.

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][1:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][1:], c='blue', label='val_loss')
plt.legend(loc='upper right')    # 우측 상단에 라벨표시
plt.title('캘리포니아 Loss')
plt.xlabel('epochs')

plt.ylabel('loss')

plt.grid()    
plt.show()    

# exit()
'''
########### submission.csv 만들기 // count 컬럼에 값 넣어주기 #################
print(submission) #[715 rows x 1 columns]
y_submit = model.predict(test_csv)

submission['count'] = y_submit
print(submission) # [715 rows x 1 columns]
print(submission.shape)  # (715, 1)

submission.to_csv(path + "submit/" + "submit_0907_1014.csv")
'''

'''
===============================================
1차 시기
random : 14
train_size = 0.75
epoch = 300
batch_size = 32

결과
- loss: 0.7383 - val_loss: 0.6921
loss :  0.6298893094062805
r2 = :  0.511302339524152
===============================================

MinMaxScaler

162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 617us/step - loss: 0.5445
loss :  0.5444505214691162
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 634us/step
r2 = :  0.5854050960043469
mse :  0.544450453266494
RMSE :  0.737868859125044
훈련시간 : 7.76 sec

==============================================

StandardScaler

Epoch 36/50000
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.5290 - val_loss: 0.5440
=============================================
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 829us/step - loss: 0.6083
loss :  0.6083400249481201
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 903us/step
r2 = :  0.5378870088056662
mse :  0.6083399280323892
RMSE :  0.7799614913778687
훈련시간 : 7.73 sec

----------------------------------------------------------------

MaxAbsScaler

Epoch 349/50000
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.5361 - val_loss: 0.5564
=============================================
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.5216
loss :  0.5216047167778015
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 856us/step
r2 = :  0.6037736904987477
mse :  0.5216046490784881
RMSE :  0.7222220220115751
훈련시간 : 69.63 sec

------------------------------------------------------------

RobustScaler

Epoch 71/50000
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.5630 - val_loss: 0.5875
===================keras28 california==========================
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.7861  
loss :  0.7861367464065552
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
r2 = :  0.402827361716242
mse :  0.7861366521151975
RMSE :  0.8866434752002619
훈련시간 : 16.15 sec

'''




'''
2026-09-10 (목)
keras19_1  카피
#1. 데이터
훈련과정 x * w + b = y 를 실해중에
x(원 데이터) 의 값이 너무 클 때,
x(원 데이터들)를 동일한 비율로 크기를 작게(0~1) 만들어보자

scaler 는 머신러닝, 딥러닝 과정중 오버플로우를 막아줄 수 있고,
성능이 향상 될 수 있다.

min_max_scaler : 0 ~ 1로 한정
(x 최댓값 - x 최소값) 을 (x - x 최소값)으로 나눈 것

standardscaler : 평균을 0으로 한정

MaxAbsScaler : 절대값이 가장 큰 값의 절대값으로 나눠서 -1 ~ 1로 한정

RobustScaler : 데이터의 중앙값 = 0, IQR = 1이 되도록 스케일링하는 기법
이상치에 강한 Scaler
from sklearn.preprocessing import MinMaxScaler

'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target


# print(x.shape, y.shape) 
# (20640, 8) (20640,)



x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.75,
    # test_size = 0.25,
    # shuffle = True
    random_state=20260910,
)
'''
scaler.fit 은 x_train으로 한다.
MinMaxScaler의 기준은 train이다.
test 데이터는 train 데이터에 종속되면 안 된다.
과적합 주의.
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

'''
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

print(x_train)

print(x_test)

print(np.min(x_train), np.max(x_train))
# 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))
# 0.0001681661481543765 1.0711971933203288

# exit()

# 2. 모델 구성
model = Sequential()
model.add(Dense(128, input_shape=(8,)))    # input_dim = 8  -> input_shape = (8, )
model.add(Dense(256))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(               # 최소 혹은 최대값을 구해주는 코드 (if문으로 구성되어있는 조건문)
    monitor = 'val_loss',         # 모니터링 할게
    mode = 'min',                 # 최소점
    patience = 30,                # patience = n 번동안 갱신되지 않으면
    restore_best_weights = True,  # 기본값 = false - 원칙적으론 true가 좋으나, false가 나을때도 있음.
)


start_time = time.time()     # 현재 시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=50000,
                 batch_size=600, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()       # 현재 시간을 반환. 종료시간

print("===================keras28 california==========================")


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

# print("============================ history =================================")
# print(hist)
# print("========================= hist.histoy ==============================")
# print(hist.history)
# print("============================= loss =================================")
# print(hist.history['loss'])
# print("=========================== val_loss =================================")
# print(hist.history['val_loss'])
# print("============================ history =================================")

import matplotlib.pyplot as plt

# print(plt.rcParams['font.family'])         # 그래프 출력시 ['sans-serif']폰트 가 디폴트이기 때문에 한글이 출력되지 않는다.

plt.rcParams['font.family'] = 'Malgun Gothic'   # 그래서 맷플롯립 폰트를 한글을 지원하는 폰트로 삽입하여 그래프에 한글이 표시되도록 한다.

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][1:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][1:], c='blue', label='val_loss')
plt.legend(loc='upper right')    # 우측 상단에 라벨표시
plt.title('캘리포니아 Loss')
plt.xlabel('epochs')

plt.ylabel('loss')

plt.grid()    #격자 표시를 추가
plt.show()    #

# exit()
'''
########### submission.csv 만들기 // count 컬럼에 값 넣어주기 #################
print(submission) #[715 rows x 1 columns]
y_submit = model.predict(test_csv)

submission['count'] = y_submit
print(submission) # [715 rows x 1 columns]
print(submission.shape)  # (715, 1)

submission.to_csv(path + "submit/" + "submit_0907_1014.csv")
'''

'''
===============================================
1차 시기
random : 14
train_size = 0.75
epoch = 300
batch_size = 32

결과
- loss: 0.7383 - val_loss: 0.6921
loss :  0.6298893094062805
r2 = :  0.511302339524152
===============================================

MinMaxScaler

162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 617us/step - loss: 0.5445
loss :  0.5444505214691162
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 634us/step
r2 = :  0.5854050960043469
mse :  0.544450453266494
RMSE :  0.737868859125044
훈련시간 : 7.76 sec

==============================================

StandardScaler

Epoch 36/50000
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.5290 - val_loss: 0.5440
=============================================
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 829us/step - loss: 0.6083
loss :  0.6083400249481201
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 903us/step
r2 = :  0.5378870088056662
mse :  0.6083399280323892
RMSE :  0.7799614913778687
훈련시간 : 7.73 sec

----------------------------------------------------------------

MaxAbsScaler

Epoch 349/50000
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.5361 - val_loss: 0.5564
=============================================
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.5216
loss :  0.5216047167778015
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 856us/step
r2 = :  0.6037736904987477
mse :  0.5216046490784881
RMSE :  0.7222220220115751
훈련시간 : 69.63 sec

------------------------------------------------------------

RobustScaler

Epoch 71/50000
21/21 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.5630 - val_loss: 0.5875
===================keras28 california==========================
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - loss: 0.7861  
loss :  0.7861367464065552
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step  
r2 = :  0.402827361716242
mse :  0.7861366521151975
RMSE :  0.8866434752002619
훈련시간 : 16.15 sec

'''




