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

from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time

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
scaler = MaxAbsScaler()
# scaler = RobustScaler()

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

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_shape=(10,)))
model.add(Dense(70))
model.add(Dense(140))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001   # 디폴트값
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
                 epochs=10000, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es, rlr],
                 )

end_time = time.time()       # 현재 시간을 반환. 종료시간

print("=============== keras53_reducel_diabetes =====================")

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


"""
훈련결과

MaxAbsScaler

Epoch 100/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 2659.1594 - val_loss: 2784.6941
=============================================
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 3625.2837 
loss :  3625.28369140625
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 12ms/step
r2 = :  0.4188517797515665
mse :  3625.2836313195485
RMSE :  60.21032827779258
훈련시간 : 7.18 sec

--------------------------------------------------------------------------

Epoch 52/10000
9/9 [==============================] - 0s 4ms/step - loss: 2901.1436 - val_loss: 2784.1646
=================== keras52_learning_rate_diabetes ==========================
4/4 [==============================] - 0s 1ms/step - loss: 3667.8240
loss :  3667.823974609375
4/4 [==============================] - 0s 0s/step
r2 = :  0.41203237781873525
mse :  3667.8240114516834
RMSE :  60.56256278801025
훈련시간 : 2.74 sec

--------------------------------------------------------------------------

Epoch 66: early stopping
=============== keras53_reducel_diabetes =====================
4/4 [==============================] - 0s 0s/step - loss: 3680.4204
loss :  3680.42041015625
4/4 [==============================] - 0s 0s/step
r2 = :  0.410013143737067
mse :  3680.4202752085357
RMSE :  60.66646746934039
훈련시간 : 3.22 sec

"""