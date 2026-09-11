# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = "./_data/kaggle_bike/"   # 상대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0)
#print(train_csv.shape) # (10886, 11)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
#print(test_csv.shape) #(6493, 8)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
#print(submission.shape) #(6493, 1)

#print(train_csv.info())
#print(test_csv.info())

#print(train_csv.describe())

########결측치 확인 ###############

#print(train_csv.isna().sum())   # 결손치 확인 코드
#print(test_csv.isnull().sum())  # 결손치 확인 코드

# exit()
########### x , y 분리 ##########
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x) #  [10886 rows x 8 columns]
y = train_csv['count']
# print(y)
# print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=2414,
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=8))    # relu는 음수를 없애주는 함수. -값을 0처리 한다.
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32)

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


########### submission.csv 만들기 // count 컬럼에 값 넣어주기 #################
print(submission) #[715 rows x 1 columns]
y_submit = model.predict(test_csv)

submission['count'] = y_submit
print(submission) # [715 rows x 1 columns]
print(submission.shape)  # (715, 1)

submission.to_csv(path + "submit/" + "submit_0907_1014.csv")


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