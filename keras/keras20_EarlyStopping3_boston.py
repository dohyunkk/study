# 2026-09-07 월

# 19_3_boston 카피
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


# 2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=13))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(               # 최소 혹은 최대값을 구해주는 코드 (if문으로 구성되어있는 조건문)
    monitor = 'val_loss',         # 모니터링 할게
    mode = 'min',                 # 최소점
    patience = 15,                # patience = n 번동안 갱신되지 않으면
    restore_best_weights = True,  # 기본값 = false - 원칙적으론 true가 좋으나, false가 나을때도 있음.
)


start_time = time.time()     # 현재 시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=50000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()       # 현재 시간을 반환. 종료시간
print("===================================================")
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

print("============================ history =================================")
print(hist)
print("========================= hist.histoy ==============================")
print(hist.history)
print("============================= loss =================================")
print(hist.history['loss'])
print("=========================== val_loss =================================")
print(hist.history['val_loss'])
print("============================ history =================================")

import matplotlib.pyplot as plt

# print(plt.rcParams['font.family'])         # 그래프 출력시 ['sans-serif']폰트 가 디폴트이기 때문에 한글이 출력되지 않는다.

plt.rcParams['font.family'] = 'Malgun Gothic'   # 그래서 맷플롯립 폰트를 한글을 지원하는 폰트로 삽입하여 그래프에 한글이 표시되도록 한다.

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][5:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][5:], c='blue', label='val_loss')
plt.legend(loc='upper right')    # 우측 상단에 라벨표시
plt.title('보스턴 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.grid()    #격자 표시를 추가
plt.show()    #


"""
훈련결과
===============================================
es = EarlyStopping(               # 최소 혹은 최대값을 구해주는 코드 (if문으로 구성되어있는 조건문)
    monitor = 'val_loss',         # 모니터링 할게
    mode = 'min',                 # 최소점
    patience = 15,                # patience = n 번동안 갱신되지 않으면
    restore_best_weights = True,  # 기본값 = false - 원칙적으론 true가 좋으나, false가 나을때도 있음.
)


start_time = time.time()     # 현재 시간을 반환. 시작시간
hist = model.fit(x_train, y_train, 
                 epochs=50000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )

결과
- loss: 44.3854 - val_loss: 51.7248
loss(mse) :  48.42266845703125
RMSE :  6.958640088841112
r2 : 0.4183033011200942
===============================================
"""