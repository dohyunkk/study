# 머신러닝과 통계학에서 Validation(검증)은 학습 도중 모델을 조정하고 과적합(Overfitting)을 막는 과정이며, 
# Evaluation(평가)은 학습이 완전히 끝난 모델의 최종 성능을 수치로 측정하는 과정
# data를 Train, validation, test 로 삼등분하여 train(공부), validation(모의고사), test(수능).

# keras09_1 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])
y_val = np.array([7,8])

x_test = np.array([9,10])
y_test = np.array([9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=1))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=6,
          verbose = 1,
          validation_data = (x_val, y_val),
          )
# verbose = 0 : 침묵 ( 훈련과정을 안 보겠다. )
# verbose = 1 : 디폴트( 기본값)
# verbose = 2 : 프로그래스 바 삭제
# verbose = 3 : 에포만 나옴.
# verbose = 나머지 : verbose = 3과 같음.


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)

# 결과. loss :  0.028295153751969337
















