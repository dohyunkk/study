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


# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train)

# print(x_test)

# print(np.min(x_train), np.max(x_train))
# -2.892958722886259 4.175175866528502
# print(np.min(x_test), np.max(x_test))
# -2.6602737306277326 3.809314198155913

x_train = x_train.reshape(-1, 10, 1, 1)
x_test = x_test.reshape(-1, 10, 1, 1)

# print(x_train.shape) # (331, 10, 1, 1)

# exit()

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(64, (2, 1), padding = 'same', activation = 'relu', input_shape = (10, 1, 1)))
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
    verbose = 1,
    )



start_time = time.time()    

hist = model.fit(x_train, y_train, 
                 epochs=3000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es ],
                 )

end_time = time.time()      


print("===================keras42_cnn_diabetes==========================")


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
--gpu-------------------------------------------
Epoch 30/30
9/9 [==============================] - 0s 4ms/step - loss: 2766.2183 - val_loss: 3779.1812
===================keras35_diabetes==========================
4/4 [==============================] - 0s 332us/step - loss: 3610.9565
loss :  3610.95654296875
4/4 [==============================] - 0s 0s/step
r2 = :  0.4211484340378284
mse :  3610.956781643874
RMSE :  60.09123714522671
훈련시간 : 2.04 sec

cnn
Epoch 612: early stopping
===================keras42_cnn_diabetes==========================
4/4 [==============================] - 0s 43ms/step - loss: 4041.0935
loss :  4041.093505859375
4/4 [==============================] - 0s 1ms/step
r2 = :  0.35219573734486764
mse :  4041.0933181533924
RMSE :  63.56959428967116
훈련시간 : 29.56 sec
'''


