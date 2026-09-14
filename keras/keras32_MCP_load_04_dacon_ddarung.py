'''
2026-09-14 (월)
31_4에 30_2 접목

https://dacon.io/competitions/open/235576/data
'''



import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import time

#1. 데이터

path = "c:/study/_data/ddarung/"       #절대경로 


train_csv = pd.read_csv(path + "train.csv", index_col=0)
# print(train_csv)   
# id열 포함 [1459 rows x 11 columns]
# id열 미포함 [1459 rows x 10 columns]

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv) 
# id열 미포함 [715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
# print(submission)
# id열 미포함 [715 rows x 1 columns]

# print(train_csv.shape) 
# (1459, 10)
# print(test_csv.shape)  
# (715, 9)
# print(submission.shape)
# (715, 1)

# print(train_csv.columns)

#print(train_csv.info())
   #   Column                  Non-Null Count  Dtype  
# ---  ------                  --------------  -----  
#  0   hour                    1459 non-null   int64  
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
#print(test_csv.info())

# exit() 

######################### 결측치 처리 1. 삭제 #########################

train_csv = train_csv.dropna()
#print(train_csv) # [1328 rows x 10 columns]

#################3###### train_csv를 x와 y로 분리######################

x = train_csv.drop(['count'], axis=1) # 카운트라는 열(컬럼)을 삭제하겠다.
#print(x) # [1328 rows x 9 columns]

y = train_csv['count']
#print(y)
#print(y.shape) # (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    random_state=23,
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
# -0.012345679012345678 1.1192660550458715

# exit()

#2. 모델 구성
model = Sequential()
model.add(Dense(18, input_shape=(9,)))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(1))


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')


es = EarlyStopping(               
    monitor = 'val_loss',         
    mode = 'min',                 
    patience = 15,                
    restore_best_weights = True,  
)

#★★★★★★★ mcp 세이브 파일명 만들기 시작 ★★★★★★★##
import datetime

date = datetime.datetime.now()
# print(date)         # 2026-09-14 11:42:10.635267
# print(type(date))   # <class 'datetime.datetime'>        # ★ calss
date = date.strftime("%m%d_%H%M")
# print(date)         # 2026-09-14 11:48:27.764409
# print(type(date))   # <class 'datetime.datetime'>

path = './_save/keras31/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'k31_', 'ddarung_', date, "-",filename])

# 내가 생각하는 파일명 ex)
# './_save/keras30/' + 'k30_' + '0914_1147' + '0530-0.001.keras'

#★★★★★★★ mcp 세이브 파일명 만들기 끝 ★★★★★★★##

mcp = ModelCheckpoint(
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
                 callbacks=[es, mcp],
                 )
end_time = time.time()       

print("===================== keras31_ddarung ================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# 평가지표(r2)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 = : ", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)

#RMSE함수의 정의
def RMSE(y_test, y_predict):    
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# random_state=334,
# model.add(Dense(9, input_dim=9))
# model.add(Dense(9))
# model.add(Dense(9))
# model.add(Dense(9))
# model.add(Dense(1))
# epochs=300, batch_size=32
# loss :  2679.29296875
# r2 = :  0.6040180951256893
# mse :  2679.2930744357695
# RMSE :  51.761888242564815

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

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'][5:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
# plt.plot(hist.history['val_loss'][5:], c='blue', label='val_loss')
# plt.legend(loc='upper right')    # 우측 상단에 라벨표시
# plt.title('대여량 Loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')

# plt.grid()    #격자 표시를 추가
# plt.show()    #


"""

===================== keras31_ddarung ================================
11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 2958.3503 
loss :  2958.350341796875
11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step 
r2 = :  0.5262083782889981
mse :  2958.3503296135264
RMSE :  54.390719149626314
훈련시간 : 7.15 sec

"""