# 2026-09-02 수

# 09 카피
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [검색] train과 test 를 섞어서 7:3으로 나눈다 (힌트 : 사이킷런)
#  정확한 판단을 위해서 랜덤한 값을 훈련에 사용한다.
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    #  train_size=0.7,
    #  test_size=0.3,
    #  shuffle=True,          #디폴트 설정값 = 섞는다.
      random_state=999,   #랜덤 난수표 중 고정값을 넣어 고정된 리턴을 뽑아낸다.
)
print('x_train :', x_train)   # x_train
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(32,input_dim=1))
model.add(Dense(32))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=8)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = : ', loss)
results = model.predict(np.array([11]))
print('예측값 : ', results)

# loss = :  1.3959055650047958e-05
# 예측값 :  [[10.9936495]]