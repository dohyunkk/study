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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import numpy as np
import time
# 1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape, x_test.shape) # (404, 13) (102, 13)
# print(y_train.shape, y_test.shape) # (404,) (102,)

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
# -0.0019120458891013214 1.1478180091225068

# exit()


# 2. 모델구성
model = Sequential()
model.add(Dense(64,input_shape=(13,)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))


# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
learning_rate = 0.005
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
                 epochs=50000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es, rlr],
                 )

end_time = time.time()       # 현재 시간을 반환. 종료시간

print("================= keras53_reducel_boston ========================")
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, )
print('loss(mse) : ', loss)

# 평가지표 (r2)
y_predict = model.predict(x_test)      # results = model.predict(x)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

mse = mean_squared_error(y_test, y_predict)
print("mse : ", mse)
#RMSE함수의 정의
def RMSE(y_test, y_predict):    
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


# loss(mse) :  23.699596405029297
# r2 =: 0.7152991240762334


# - loss: 37.3669 - val_loss: 69.8842
# loss(mse) :  54.49088668823242
# r2 : 0.34540641003351025

print("훈련시간 :", round(end_time - start_time, 2),"sec")


"""
훈련결과

===================================================
standardscaler

Epoch 51/50000
11/11 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 22.9807 - val_loss: 21.6166
===================================================
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 24.5732 
loss(mse) :  24.573232650756836
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step
r2 : 0.7048042150528292
mse :  24.57323320582577
RMSE :  4.9571396193597135
훈련시간 : 3.76 sec


----------------------------------------------------------------
Epoch 34/50000
11/11 [==============================] - 0s 3ms/step - loss: 23.7604 - val_loss: 21.8557
================= keras52_learning_rate_boston ========================
4/4 [==============================] - 0s 2ms/step - loss: 26.6346
loss(mse) :  26.634632110595703
4/4 [==============================] - 0s 0s/step
r2 : 0.6800408603302001
mse :  26.634630155198384
RMSE :  5.160874940860162
훈련시간 : 2.07 sec

----------------------------------------------------------------
Epoch 66: early stopping
================= keras53_reducel_boston ========================
4/4 [==============================] - 0s 0s/step - loss: 23.3966
loss(mse) :  23.39656639099121
4/4 [==============================] - 0s 3ms/step
r2 : 0.7189394097022672
mse :  23.39656520363006
RMSE :  4.836999607569765
훈련시간 : 3.03 sec

"""