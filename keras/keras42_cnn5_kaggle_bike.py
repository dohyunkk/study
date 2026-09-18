'''
2026-09-17 (목)

DNN모델을 CNN모델로 변환

# https://www.kaggle.com/competitions/bike-sharing-demand/data


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

x_train = x_train.reshape(-1, 8, 1, 1)
x_test = x_test.reshape(-1, 8, 1, 1)

# print(x_train.shape) # 

# exit()


# 2. 모델 구성

model = Sequential()
model.add(Conv2D(64, (2, 1), padding = 'same', activation = 'relu', input_shape = (8, 1, 1)))
model.add(Conv2D(64, (2, 1), padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))

model.add(Conv2D(32, (2, 1), padding = 'same', activation = 'relu'))
model.add(Conv2D(32, (2, 1), padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D())
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(               
    monitor = 'val_loss',         
    mode = 'min',                 
    patience = 30,                
    restore_best_weights = True,  
)


start_time = time.time()     

hist = model.fit(x_train, y_train, 
                 epochs=1000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es, ]# mcp],
                 )

end_time = time.time()   

print("====================keras42_cnn_kaggle_bike==============================")


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
Epoch 252/1000
205/205 [==============================] - 1s 3ms/step - loss: 21902.0703 - val_loss: 21635.4336
====================keras42_cnn_kaggle_bike==============================
86/86 [==============================] - 0s 2ms/step - loss: 22105.1895
loss :  22105.189453125
86/86 [==============================] - 0s 901us/step
r2 = :  0.3463830351829529
mse :  22105.189453125
RMSE :  148.67814046834525
훈련시간 : 149.67 sec
'''