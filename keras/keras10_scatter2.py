# 2026-09-02 수

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    #  train_size=0.7,
    #  test_size=0.3,
    #  shuffle=True,          #디폴트 설정값 = 섞는다.
      random_state=300,   #랜덤 난수표 중 고정값을 넣어 고정된 리턴을 뽑아낸다.
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

print("=============================================")

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=20)

print("=============================================")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

results= model.predict(x)
print(results)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, results)
plt.show()

# random_state=999, 
# loss :  8.157328605651855
# [[ 1.3274056]
#  [ 2.2529006]
#  [ 3.1783957]
#  [ 4.1038904]
#  [ 5.0293856]
#  [ 5.9548798]
#  [ 6.880375 ]
#  [ 7.80587  ]
#  [ 8.731365 ]
#  [ 9.6568575]
#  [10.582352 ]
#  [11.507848 ]
#  [12.433345 ]
#  [13.35884  ]
#  [14.284331 ]
#  [15.209828 ]
#  [16.135324 ]
#  [17.060822 ]
#  [17.986315 ]
#  [18.911808 ]]