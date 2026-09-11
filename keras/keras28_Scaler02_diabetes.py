'''
2026-09-10 (목)
# keras12_3 카피
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
                 epochs=10000, 
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )

end_time = time.time()       # 현재 시간을 반환. 종료시간

print("=================== keras28_diabetes ==========================")

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
plt.plot(hist.history['loss'][5:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][5:], c='blue', label='val_loss')
plt.legend(loc='upper right')    # 우측 상단에 라벨표시
plt.title('당뇨 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.grid()    #격자 표시를 추가
plt.show()    #

"""
훈련결과
===============================================

epochs=100, 
batch_size=32,
validation_split=0.25,

결과
- loss: 24292.8809 - val_loss: 24889.7012
loss :  24737.177734375
r2 = :  0.42228429517338506
===============================================

9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 2688.6184 - val_loss: 2849.6362
=============================================
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 3585.0813 
loss :  3585.081298828125
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
r2 = :  0.4252963293096209
훈련시간 : 6.74 sec

===================================================
standardscaler

Epoch 100/100
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 2801.0405 - val_loss: 2956.2419
=============================================
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 3752.1475 
loss :  3752.1474609375
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
r2 = :  0.39851492541511824
mse :  3752.147764374862
RMSE :  61.25477748204512
훈련시간 : 6.97 sec

-----------------------------------------------------------------

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

----------------------------------------------------------------------
RobustScaler

Epoch 74/10000
9/9 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 2720.4141 - val_loss: 2683.5386
=================== keras28_diabetes ==========================
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 3697.6072 
loss :  3697.607177734375
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
r2 = :  0.4072580069437586
mse :  3697.60720269247
RMSE :  60.80795344930193
훈련시간 : 5.64 sec

"""