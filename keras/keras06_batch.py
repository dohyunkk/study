# 2026-08-31 월

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련 - 하이퍼 파라미터 튜닝(에포, 레이어, 덴스, 배치사이즈)을 통해 훈련의 질을 향상시킬 수 있다.
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=2 ) #batch_size를 설정하지 않으면 기본값 32로 작동
#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
# result = model.predict(np.array([1,2,3,4,5,6]))
# print("6의 예측값 : " ,result)