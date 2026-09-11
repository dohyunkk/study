'''
2026-09-10 (목) 
keras24_santander 카피
https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
'''

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


#1. 데이터
# path = './_data/kaggle_santander/'              # 상대경로
path = 'c:/study/_data/kaggle_santander/'         # 절대경로

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
'''
shape가 안 될 때

import numpy as np

# 판다스를 넘파이로 바꾸기
# y = np.array(y)
# y = y.to_numpy()

'''

# print(train_csv)            # [200000 rows x 201 columns]
# print(test_csv)             # [200000 rows x 200 columns]
# print(submission_csv)       # [200000 rows x 1 columns]

# print(train_csv.shape)      # (200000, 201)
# print(test_csv.shape)       # (200000, 200)
# print(submission_csv.shape) # (200000, 1)

# print(train_csv.info())       # 결측치, 이상치 확인
# print(train_csv.isna().sum())   # 결측치 확인
# print(test_csv.isnull().sum())  # 결측치 확인


x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
# print(x.shape, y.shape)      # (200000, 200) (200000,)

# print(np.unique(y, return_counts=True))     # 파일의 행렬 보기
# (array([0, 1]), array([179902,  20098]))

###### one-hot encoding ######
y = pd.get_dummies(y, dtype=int)
# print(y)
#               0  1
# ID_code           
# train_0       1  0
# train_1       1  0
# train_2       1  0
# train_3       1  0
# train_4       1  0
# ...          .. ..
# train_199995  1  0
# train_199996  1  0
# train_199997  1  0
# train_199998  1  0
# train_199999  1  0

# [200000 rows x 2 columns]

# exit()



x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=997,
    train_size=0.7,
    stratify=y,                 # y_train과 y_predict의 값이 치우치지 않게 해줌.
)

print(x_train.shape, x_test.shape)
# (140000, 200) (60000, 200)
print(y_train.shape, y_test.shape)
# (140000, 2) (60000, 2)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train)

print(x_test)

print(np.min(x_train), np.max(x_train))
# 0.0 1.0000000000000004
print(np.min(x_test), np.max(x_test))
# -0.08032267285709349 1.0867433267723332

# exit()

#2. 모델

model = Sequential()
model.add(Dense(1000, input_shape = (200,), activation = 'relu'))
model.add(Dense(800, activation = 'relu'))
model.add(Dense(1200, activation = 'relu'))
model.add(Dense(600, activation = 'relu'))
model.add(Dense(400, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))


#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 5,
    restore_best_weights = True,
)

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 10000,
          batch_size = 10000,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es],
          )

end_time = time.time()

#.4 평가, 예측

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', round(result[1],2))

y_predict = model.predict(x_test)

# print(y_predict)
# [[0.6418624  0.3581376 ]
#  [0.95958644 0.04041353]
#  [0.992097   0.00790298]
#  ...
#  [0.9274548  0.07254522]
#  [0.7445844  0.2554157 ]
#  [0.96017736 0.0398226 ]]

# print(y_predict.shape)
# (60000, 2)


y_predict = np.argmax(y_predict, axis=1)

y_test = np.argmax(y_test, axis=1)


accuracy_score = accuracy_score(y_test, y_predict)
print('============================================================')
print('acc_score : ', accuracy_score)
print('걸린시간 : ', round(end_time - start_time, 2),'sec')
print('============================================================')

# csv_만들기
# print(submission_csv)        # [200000 rows x 1 columns]
# print(submission_csv.shape)  # (200000, 1)
y_pred = model.predict(test_csv)
y_pred = np.argmax(y_pred, axis = 1)
print(np.unique(y_pred, return_counts=True)) 
# (array([0, 1]), array([193522,   6478]))
print(y_pred.shape)
# (200000,)
submission_csv['target'] = y_pred
submission_csv.to_csv(path + "submit/" + "submit_0910_1040.csv")

'''
20/20 ━━━━━━━━━━━━━━━━━━━━ 3s 142ms/step - acc: 0.9562 - loss: 0.1212 - val_acc: 0.8921 - val_loss: 0.3507
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - acc: 0.9105 - loss: 0.2473  
loss :  0.24729163944721222
acc :  0.91
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step   
============================================================
acc_score :  0.9104666666666666
걸린시간 :  111.1 sec
============================================================

Epoch 97/10000
12/12 ━━━━━━━━━━━━━━━━━━━━ 3s 263ms/step - acc: 0.9189 - loss: 0.2212 - val_acc: 0.9126 - val_loss: 0.2372
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - acc: 0.9159 - loss: 0.2293  
loss :  0.22926713526248932
acc :  0.92
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step   
============================================================
acc_score :  0.9159166666666667
걸린시간 :  305.51 sec
============================================================
6250/6250 ━━━━━━━━━━━━━━━━━━━━ 11s 2ms/step 




'''











