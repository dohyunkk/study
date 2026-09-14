'''
2026-09-14 (월)
keras30_1  카피

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

mcp = ModelCheckpoint()
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing, load_diabetes

path = './_save/keras31/'


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)  #(442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    random_state=95,
)

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
# 0.0 1.0
print(np.min(x_test), np.max(x_test))
# -0.10144927536231879 1.0144927536231885

# exit()

# 2. 모델 구성

model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape=(10,)))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))

# model.summary()

# exit()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint # ★ ModelCheckpoint

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(              
    monitor = 'val_loss',        
    mode = 'min',                 
    patience = 30,                
    restore_best_weights = True,
    verbose = 1,
    )

#★★★★★★★ mcp 세이브 파일명 만들기 시작 ★★★★★★★##
import datetime

date = datetime.datetime.now()
print(date)         # 2026-09-14 11:42:10.635267
print(type(date))   # <class 'datetime.datetime'>        # ★ calss
date = date.strftime("%m%d_%H%M")
print(date)         # 2026-09-14 11:48:27.764409
print(type(date))   # <class 'datetime.datetime'>

path = './_save/keras31/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'k31_', 'diabet',date, "-",filename])

# 내가 생각하는 파일명 ex)
# './_save/keras30/' + 'k30_' + '0914_1147' + '0530-0.001.keras'

#★★★★★★★ mcp 세이브 파일명 만들기 끝 ★★★★★★★##

mcp = ModelCheckpoint(                              # ★ mcp = ModelCheckpoint()
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath,
    verbose = 1,
)

start_time = time.time()    

hist = model.fit(x_train, y_train, 
                 epochs=50000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es, mcp ],              # ★callbacks = [mcp],
                 )

end_time = time.time()      

# model.save(path + 'keras29_3_save_model.keras')
# model.save_weights(path + 'keras29_5_save_model_2.weights.h5')


# model.load_weights(path + 'keras29_5_save_model_1.weights.h5')




print("===================keras31_MCP_diabetes==========================")


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

# print("훈련시간 :", round(end_time - start_time, 2),"sec")

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

# plt.rcParams['font.family'] = 'Malgun Gothic'   # 그래서 맷플롯립 폰트를 한글을 지원하는 폰트로 삽입하여 그래프에 한글이 표시되도록 한다.

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][1:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][1:], c='blue', label='val_loss')
# plt.legend(loc='upper right')    # 우측 상단에 라벨표시
# plt.title('캘리포니아 Loss')
# plt.xlabel('epochs')

# plt.ylabel('loss')

# plt.grid()    
# plt.show()    

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

r2 = :  0.7701598482056351
mse :  0.3025687311671747
RMSE :  0.5500624793304618

===============================================


'''


