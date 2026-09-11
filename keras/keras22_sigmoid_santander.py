'''
2026-09-08 화
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

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=997,
    train_size=0.7,
    stratify=y,                 # y_train과 y_predict의 값이 치우치지 않게 해줌.
)

# 2. 모델구성
model = Sequential()
model.add(Dense(400,input_dim=200, activation = 'relu'))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dense(250, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights = True,
)

start_time = time.time()

hist = model.fit(x_train, y_train,
                 epochs = 10000,
                 batch_size = 3000,
                 verbose = 1,
                 validation_split = 0.3,
                 callbacks = [es],
                 )

end_time = time.time()


# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print("====================================================")
print('loss : ', loss[0])
print('acc : ' , round(loss[1], 4))
print("====================================================")

y_pred = model.predict(test_csv)
# print(y_pred[:10])
y_pred = np.round(y_pred)
print(y_pred[:20])
print(y_pred[-20:])
print(np.unique(y_pred, return_counts=True))



acc_score = accuracy_score(y, y_pred)
print('acc_score : ', acc_score)  

# csv_만들기
# print(submission_csv)        # [200000 rows x 1 columns]
# print(submission_csv)        # [200000 rows x 1 columns]
# print(submission_csv.shape)  # (200000, 1)
submission_csv['target'] = y_pred

submission_csv.to_csv(path + "submit/" + "submit_0908_1736.csv")


'''
====================================================
loss :  0.25371411442756653
acc :  0.9071
====================================================
'''