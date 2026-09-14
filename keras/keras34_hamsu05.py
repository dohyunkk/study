'''
2026-09-14 (월)

# https://www.kaggle.com/competitions/bike-sharing-demand/data
'''

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
#1. 데이터
path = "./_data/kaggle_bike/"   # 상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0)
#print(train_csv.shape) # (10886, 11)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
#print(test_csv.shape) #(6493, 8)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
#print(submission.shape) #(6493, 1)

#print(train_csv.info())
#print(test_csv.info())

#print(train_csv.describe())

########결측치 확인 ###############

#print(train_csv.isna().sum())   # 결손치 확인 코드
#print(test_csv.isnull().sum())  # 결손치 확인 코드

# exit()
########### x , y 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x) #  [10886 rows x 8 columns]
y = train_csv['count']
# print(y)
# print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=2414,
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
# 0.0 1.0

# exit()


# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input

# model = Sequential()
# model.add(Dense(16, activation='relu', input_shape=(8,)))    
# model.add(Dropout(0.3))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

# model.summary()

input1 = Input(shape=(8,))
dense1 = Dense(16, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(16, activation='relu')(drop1)
dense3 = Dense(8, activation='relu')(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)
# model.summary()

# exit()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(               
    monitor = 'val_loss',         
    mode = 'min',                 
    patience = 30,                
    restore_best_weights = True,  
)

#★★★★★★★ mcp 세이브 파일명 만들기 시작 ★★★★★★★##
# import datetime

# date = datetime.datetime.now()
# print(date)         # 2026-09-14 11:42:10.635267
# print(type(date))   # <class 'datetime.datetime'>        # ★ calss
# date = date.strftime("%m%d_%H%M")
# print(date)         # 2026-09-14 11:48:27.764409
# print(type(date))   # <class 'datetime.datetime'>

# path = './_save/keras31/'
# filename = '{epoch:04d}-{val_loss:.4f}.keras'
# filepath = "".join([path, 'k31_', 'kaggle_bike_', date, "-",filename])

# # 내가 생각하는 파일명 ex)
# # './_save/keras30/' + 'k30_' + '0914_1147' + '0530-0.001.keras'

# #★★★★★★★ mcp 세이브 파일명 만들기 끝 ★★★★★★★##

# mcp = ModelCheckpoint(
#     monitor = 'val_loss',
#     mode = 'auto',
#     save_best_only = True,
#     filepath = filepath,
#     verbose = 1,
# )

start_time = time.time()     

hist = model.fit(x_train, y_train, 
                 epochs=50000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es, ]# mcp],
                 )

end_time = time.time()   

print("====================keras34_kaggle_bike==============================")


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


########### submission.csv 만들기 // count 컬럼에 값 넣어주기 #################
# print(submission) #[715 rows x 1 columns]
# y_submit = model.predict(test_csv)

# submission['count'] = y_submit
# print(submission) # [715 rows x 1 columns]
# print(submission.shape)  # (715, 1)

# submission.to_csv(path + "submit/" + "submit_0907_1014.csv")

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

# import matplotlib.pyplot as plt

# print(plt.rcParams['font.family'])         # 그래프 출력시 ['sans-serif']폰트 가 디폴트이기 때문에 한글이 출력되지 않는다.

# plt.rcParams['font.family'] = 'Malgun Gothic'   # 그래서 맷플롯립 폰트를 한글을 지원하는 폰트로 삽입하여 그래프에 한글이 표시되도록 한다.

# plt.figure(figsize=(9,6))     # 그래프가 출력되는 창의 사이즈
# plt.plot(hist.history['loss'][5:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][5:], c='blue', label='val_loss')
# plt.legend(loc='upper right')    # 우측 상단에 라벨표시
# plt.title('대여량 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')

# plt.grid()    #격자 표시를 추가
# plt.show()    #



'''

====================keras31_kaggle_bike==============================
86/86 ━━━━━━━━━━━━━━━━━━━━ 0s 690us/step - loss: 22389.6914
loss :  22389.69140625
86/86 ━━━━━━━━━━━━━━━━━━━━ 0s 812us/step
r2 = :  0.3379707932472229
mse :  22389.689453125
RMSE :  149.63184638680698
훈련시간 : 48.19 sec

====================keras33_kaggle_bike==============================
86/86 ━━━━━━━━━━━━━━━━━━━━ 0s 757us/step - loss: 23891.2500
loss :  23891.25
86/86 ━━━━━━━━━━━━━━━━━━━━ 0s 814us/step
r2 = :  0.2935718894004822
mse :  23891.25
RMSE :  154.56794622430616
훈련시간 : 55.76 sec

====================keras34_kaggle_bike==============================
86/86 ━━━━━━━━━━━━━━━━━━━━ 0s 511us/step - loss: 23962.8359
loss :  23962.8359375
86/86 ━━━━━━━━━━━━━━━━━━━━ 0s 722us/step
r2 = :  0.29145514965057373
mse :  23962.83984375
RMSE :  154.7993534991345
훈련시간 : 16.1 sec
'''