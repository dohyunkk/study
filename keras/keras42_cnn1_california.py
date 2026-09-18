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

from sklearn.datasets import fetch_california_housing

path = './_save/keras30/'


#1. 데이터
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target


print(x.shape, y.shape) 
# (20640, 8) (20640,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.75,
    # test_size = 0.25,
    # shuffle = True
    random_state=99284,
)


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

# print(x_train)

# print(x_test)

print(np.min(x_train), np.max(x_train))
# 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))
# 0.0001681661481543765 1.0711971933203288

x_train = x_train.reshape(-1, 8, 1, 1)
x_test = x_test.reshape(-1, 8, 1, 1)

# print(x_train.shape) # (15480, 8, 1, 1)

# exit()


# 2. 모델 구성

model = Sequential()
model.add(Conv2D(64, (2, 1), padding = 'same', activation = 'relu', input_shape=(8, 1, 1)))
model.add(Conv2D(64, (2, 1), padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))

model.add(Conv2D(32, (2, 1), padding = 'same', activation = 'relu'))
model.add(Conv2D(32, (2, 1), padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))

model.add(Conv2D(16, (2, 1), padding = 'same', activation = 'relu'))
model.add(Conv2D(16, (2, 1), padding = 'same', activation = 'relu'))
model.add(Dropout(0.3))


model.add(GlobalAveragePooling2D())
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.summary()

# exit()
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(              
    monitor = 'val_loss',        
    mode = 'min',                 
    patience = 30,                
    restore_best_weights = True,
    verbose = 1,
    )


start_time = time.time()    

hist = model.fit(x_train, y_train, 
                 epochs=1000,
                 batch_size=64, 
                 validation_split=0.2,
                  callbacks=[es],  
                 )

end_time = time.time()      


print("===================keras42_cnn_california==========================")


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
--cpu----------------------------------

Epoch 30/30
387/387 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 0.3176 - val_loss: 0.2897
===================keras35_california==========================
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 647us/step - loss: 0.2969
loss :  0.29687678813934326
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 676us/step
r2 = :  0.7744835665757787
mse :  0.2968768536994423
RMSE :  0.5448640690112005
훈련시간 : 17.54 sec

--------------------------------------------
cnn
Epoch 203: early stopping
===================keras42_cnn_california==========================
162/162 [==============================] - 0s 2ms/step - loss: 0.3288
loss :  0.3288344442844391
162/162 [==============================] - 0s 1ms/step
r2 = :  0.7555503695492564
mse :  0.3288343591579846
RMSE :  0.5734408070219493
훈련시간 : 134.04 sec
'''


