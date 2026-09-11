'''
# 2026-09-10 (목)
keras21_sigmoid_metrics_cancer 카피

'''
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.datasets import load_breast_cancer    # 유방암 관련 데이터 셋 불러오기


#1. 데이터

datasets = load_breast_cancer()     #
# print(datasets.DESCR)             #
# print(datasets.feature_names)     #

x = datasets.data        # x = datasets['data']     #딕셔너리 형태의 데이터였구나
y = datasets.target

print(x.shape, y.shape)
# (569, 30) (569,)
print(type(x))           
# <class 'numpy.ndarray'> 넘파이로 구성되어있구나.

# print(y)      

### 0과 1의 개수가 몇개인지 찾아보기. -numpy            
print(np.unique(y))
# [0 1]
print(np.unique(y, return_counts=True))
# (array([0, 1]), array([212, 357])

### 0과 1의 개수가 몇개인지 찾아보기. -pandas
print(pd.DataFrame(y).value_counts())  
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

# train_test_나누기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.7, random_state = 7,
    stratify = y,         # 새로 추가된 부분 < 무조건 넣는건 아니고 분류모델에 넣으면 좋아질 수 있다.
)

print(np.unique(y_train, return_counts=True))  # y_train의 카테고리 확인
# (array([0, 1]), array([148, 250])
print(np.unique(y_test, return_counts=True))
# (array([0, 1]), array([ 64, 107])

print(x_train.shape, x_test.shape)
# (398, 30) (171, 30)
print(y_train.shape, y_test.shape)
# (398,) (171,)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

print(x_test)

print(np.min(x_train), np.max(x_train))
# 0.0 1.0000000000000002
print(np.min(x_test), np.max(x_test))
# -0.09792843691148767 2.5798045602605866


# exit()

# 2. 모델구성
model = Sequential()
model.add(Dense(90, input_shape=(30,), activation = 'relu'))
model.add(Dense(180, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))                  #★ activation= 'sigmoid'  디폴트는'Linear'
                                                             #★ 이진분류는 마지막 activation에 반드시 'sigmoid'

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',                    #★ 이진분류는 무조건 binary_crossentropy 
              optimizer='adam',
              metrics=['acc'], # metrics=['accuracy']
              )                                             

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 20,
    restore_best_weights = True,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs = 1000,
                 batch_size = 32,
                 verbose = 1,
                 validation_split = 0.3,
                 callbacks = [es],
                 )

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("====================================================")
print('loss : ', loss[0])
print('acc : ' , round(loss[1], 4))
print("====================================================")

y_pred = model.predict(x_test)
# print(y_pred[:10])
y_pred = np.round(y_pred)
# print(y_pred[:10])

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print('acc_score : ', acc_score)

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)
# print("r2 = : ", r2 )

# mse = mean_squared_error(y_test, y_predict)
# print('r2 : ', r2)

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test, y_predict)
# print('RMSE : ', rmse)

"""
print("훈련시간 : ", round(end_time - start_time, 2),"sec")

print("========================= history ==================================")
print(hist)
print("======================= hist.history ===============================")
print(hist.history)
print("========================== loss ====================================")
print(hist.history['loss'])
print("=========================== val_loss =================================")
print(hist.history['val_loss'])
print("============================ history =================================")

import matplotlib.pyplot as plt

# print(plt.rcParams['font.family'])         # 그래프 출력시 ['sans-serif']폰트 가 디폴트이기 때문에 한글이 출력되지 않는다.

plt.rcParams['font.family'] = 'Malgun Gothic'   # 그래서 맷플롯립 폰트를 한글을 지원하는 폰트로 삽입하여 그래프에 한글이 표시되도록 한다.

plt.figure(figsize=(9,6))     # 그래프가 출력되는 창의 사이즈
plt.plot(hist.history['loss'][5:], c='red', label='loss',)   #y값만 넣으면 시간순으로 그려줌.
plt.plot(hist.history['val_loss'][5:], c='blue', label='val_loss')
plt.legend(loc='upper right')    # 우측 상단에 라벨표시
plt.title('유방암 Loss')
plt.xlabel('epochs')
plt.ylabel('loss')

plt.grid()    #격자 표시를 추가
plt.show()    #
"""




