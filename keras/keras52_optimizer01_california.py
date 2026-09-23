'''
2026-09-23 (수)

27_1 카피

지금까지 러닝레이트를 기본값으로 하고있었기 때문에
에포를 늘려 해결했다.
이제부터는 러닝레이트를 조절해서 모델 가동시간의 단축과, 성능 향상을 노려보자.

from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
# learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

optimizer=Adam(learning_rate=learning_rate)
'''


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


# print(x.shape, y.shape) 
# (20640, 8) (20640,)



'''
MinMaxScaler

원값 - Min
-----------
Max  - Min

'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# print(x)
# [[0.53966842 0.78431373 0.0435123  ... 0.00149943 0.5674814  0.21115538]
#  [0.53802706 0.39215686 0.03822395 ... 0.00114074 0.565356   0.21215139]
#  [0.46602805 1.         0.05275646 ... 0.00169796 0.5642933  0.21015936]
#  ...
#  [0.08276438 0.31372549 0.03090386 ... 0.0013144  0.73219979 0.31175299]
#  [0.09429525 0.33333333 0.03178269 ... 0.0011515  0.73219979 0.30179283]
#  [0.13025338 0.29411765 0.03125246 ... 0.00154886 0.72582359 0.30976096]]

# print(np.min(x), np.max(x))
#       0.0      1.0000000000000002

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=9984,
)



# 2. 모델 구성
model = Sequential()
model.add(Dense(256, input_shape=(8,)))    # input_dim = 8  -> input_shape = (8, )
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))


# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

learning_rate = 0.01
# learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009



model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(               # 최소 혹은 최대값을 구해주는 코드 (if문으로 구성되어있는 조건문)
    monitor = 'val_loss',         # 모니터링 할게
    mode = 'min',                 # 최소점
    patience = 30,                # patience = n 번동안 갱신되지 않으면
    restore_best_weights = True,  # 기본값 = false - 원칙적으론 true가 좋으나, false가 나을때도 있음.
)


start_time = time.time()     

hist = model.fit(x_train, y_train, 
                 epochs=5000,
                 batch_size=32, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()       

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
plt.plot(hist.history['loss'][1:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][1:], c='blue', label='val_loss')
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
random : 14
train_size = 0.75
epoch = 300
batch_size = 32

결과
- loss: 0.7383 - val_loss: 0.6921
loss :  0.6298893094062805
r2 = :  0.511302339524152
===============================================
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 617us/step - loss: 0.5445
loss :  0.5444505214691162
162/162 ━━━━━━━━━━━━━━━━━━━━ 0s 634us/step
r2 = :  0.5854050960043469
mse :  0.544450453266494
RMSE :  0.737868859125044
훈련시간 : 7.76 sec

Epoch 50/50
387/387 [==============================] - 1s 1ms/step - loss: 0.5451 - val_loss: 0.5327
=============================================
162/162 [==============================] - 0s 987us/step - loss: 0.5527
loss :  0.5527443885803223
162/162 [==============================] - 0s 685us/step
r2 = :  0.5790893739430685
mse :  0.5527443268906721
RMSE :  0.7434677712521721
훈련시간 : 31.1 sec

'''


















