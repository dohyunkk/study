# keras12_3 카피
from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
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

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start_time = time.time()     # 현재 시간을 반환. 시작시간
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
end_time = time.time()       # 현재 시간을 반환. 종료시간

print("=============================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# 평가지표(r2)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2 = : ", r2)

# loss :  3158.30517578125
# r2 = :  0.43387182657837486

# - loss: 2756.9297 - val_loss: 3143.7991
# loss :  3208.415771484375
# r2 = :  0.42488947072663297
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

"""