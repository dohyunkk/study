'''
2026-09-14 (월)
keras29_6  카피

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
from sklearn.datasets import fetch_california_housing

path = './_save/keras30/'


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

scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)

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
model.add(Dense(128, activation = 'relu', input_shape=(8,)))
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
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint   # ★ ModelCheckpoint


model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(              
    monitor = 'val_loss',        
    mode = 'min',                 
    patience = 30,                
    restore_best_weights = True,
    verbose = 1,
    )

mcp = ModelCheckpoint(                              # ★ mcp = ModelCheckpoint()
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = path + 'keras30_mcp1.keras',
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




print("===================keras30_modelcheckpoint==========================")


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


