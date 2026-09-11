# keras14_1  카피
# https://www.kaggle.com/competitions/bike-sharing-demand/data
# 훈련결과를 시각화하여 그래프를 보고 로스값을 확인하여 과적합(overfit)을 줄여보자.

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=99,
)


# 2. 모델 구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))    # relu는 음수를 없애주는 함수. -값을 0처리 한다.
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()     # 현재 시간을 반환. 시작시간
hist = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
end_time = time.time()       # 현재 시간을 반환. 종료시간

print("=============================================")


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
plt.plot(hist.history['loss'][3:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][3:], c='blue', label='val_loss')
plt.legend(loc='upper right')    # 우측 상단에 라벨표시
plt.title('캘리포니아 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.grid()    #격자 표시를 추가
plt.show()    #

exit()
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
1차 시기
random : 666
train_size = 0.75
epoch = 300
batch_size = 32

결과
RMSE :  49.92412160899653
r2 = :  0.5924579635041599
===============================================
'''