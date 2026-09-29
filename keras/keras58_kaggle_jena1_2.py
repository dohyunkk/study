'''
2026-09-29

https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

x = (n, 144, 13)
y = (n, 144, 1)

2016-12-31 00:00 ~ 2017-01-01 00:00의 wd(풍향)을 맞춰보자
'''

import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. 데이터
# ============================================================

path = './_data/kaggle_jena/'

datasets = pd.read_csv(
    path + 'jena_climate_2009_2016.csv',
    index_col=0
)

print('datasets.shape :', datasets.shape)
# (420551, 14)


# ------------------------------------------------------------
# 우리가 최종적으로 맞혀야 하는 실제 정답
# 마지막 144개의 wd(풍향)
# ------------------------------------------------------------

y_cor = datasets[-144:]['wd (deg)']

print('y_cor.shape :', y_cor.shape)
# (144,)


# ------------------------------------------------------------
# 마지막 하루를 예측하기 위한 입력 데이터
# 마지막 하루 바로 전 144개
# ------------------------------------------------------------

x_predict = datasets[-288:-144].drop(
    ['wd (deg)'],
    axis=1
)

print('x_predict.shape :', x_predict.shape)
# (144, 13)


# ============================================================
# 훈련 데이터 만들기
# ============================================================

start_time = time.time()


# x : 현재 144개 데이터를 사용
x_data = datasets[:-288].drop(
    ['wd (deg)'],
    axis=1
)

# y : x보다 144칸 뒤의 wd
y_data = datasets[144:-144]['wd (deg)']


print('x_data.shape :', x_data.shape)
# (420263, 13)

print('y_data.shape :', y_data.shape)
# (420263,)


size_x = 144
size_y = 144


# ------------------------------------------------------------
# 시계열 데이터 자르는 함수
# ------------------------------------------------------------

def split_x(dataset, size):
    aaa = []

    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size]
        aaa.append(subset)

    return np.array(aaa)


# ------------------------------------------------------------
# x : (n, 144, 13)
# y : (n, 144)
# ------------------------------------------------------------

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)


end_time = time.time()

print('x.shape :', x.shape)
print('y.shape :', y.shape)

print(
    '자르는 시간 :',
    round(end_time - start_time, 2),
    'sec'
)


# ============================================================
# 2. train / test 분리
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    shuffle=False
)


print('\n===== train / test =====')

print('x_train.shape :', x_train.shape)
print('x_test.shape  :', x_test.shape)

print('y_train.shape :', y_train.shape)
print('y_test.shape  :', y_test.shape)


# ============================================================
# 3. 스케일링
# ============================================================

scaler_x = StandardScaler()
scaler_y = StandardScaler()


# ------------------------------------------------------------
# x 스케일링
#
# (n, 144, 13)
#       ↓
# (n*144, 13)
# ------------------------------------------------------------

x_train_shape = x_train.shape
x_test_shape = x_test.shape


x_train = x_train.reshape(-1, 13)
x_test = x_test.reshape(-1, 13)


# train은 fit + transform
x_train = scaler_x.fit_transform(x_train)

# test는 transform만
x_test = scaler_x.transform(x_test)


# 다시 LSTM용 3차원으로 복구
x_train = x_train.reshape(x_train_shape)
x_test = x_test.reshape(x_test_shape)


# ------------------------------------------------------------
# y 스케일링
#
# (n, 144)
#     ↓
# (n*144, 1)
# ------------------------------------------------------------

y_train_shape = y_train.shape
y_test_shape = y_test.shape


y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


# 다시 원래 모양으로
y_train = y_train.reshape(y_train_shape)
y_test = y_test.reshape(y_test_shape)


# ------------------------------------------------------------
# LSTM 정답 모양
#
# (n, 144)
#     ↓
# (n, 144, 1)
# ------------------------------------------------------------

y_train = y_train.reshape(y_train.shape[0],y_train.shape[1],1)

y_test = y_test.reshape(y_test.shape[0],y_test.shape[1],1)


print('===== scaling 완료 =====')

print('x_train.shape :', x_train.shape)
print('x_test.shape  :', x_test.shape)

print('y_train.shape :', y_train.shape)
print('y_test.shape  :', y_test.shape)

# x_train : (n, 144, 13)
# y_train : (n, 144, 1)


# ============================================================
# x_predict도 train과 같은 scaler 적용
# ============================================================

x_predict = x_predict.values

# (144, 13)
x_predict = scaler_x.transform(x_predict)

# 모델에 넣기 위해 sample 차원 추가
x_predict = x_predict.reshape(1, 144, 13)


print('x_predict.shape :', x_predict.shape)
# (1, 144, 13)


# ============================================================
# 4. 모델 구성
# ============================================================

model = Sequential()

model.add(LSTM(units=32, input_shape=(144, 13), return_sequences=True))

model.add(LSTM(units=32, return_sequences=True))

model.add(Dense(32, activation='relu'))

# 각 timestep마다 풍향 1개씩 출력
model.add(Dense(1))


model.summary()

# 입력
# (None, 144, 13)
#
# 출력
# (None, 144, 1)


# ============================================================
# 5. 컴파일 / 훈련
# ============================================================

model.compile(
    loss='mse', optimizer=Adam(learning_rate=0.001))


es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    restore_best_weights=True
)


rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    patience=10,
    factor=0.5,
    verbose=1
)


model.fit(
    x_train,y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    verbose=1,
    callbacks=[es, rlr]
)


# ============================================================
# 6. 평가
# ============================================================

results = model.evaluate(
    x_test,
    y_test
)

print('test loss :', results)


# ============================================================
# 7. 2016-12-31 풍향 144개 예측
# ============================================================

y_predict = model.predict(x_predict)


print('예측 직후 y_predict.shape :')
print(y_predict.shape)

# (1, 144, 1)


# ============================================================
# 8. 스케일링된 예측값을 원래 풍향 값으로 복원
# ============================================================

y_predict = y_predict.reshape(-1, 1)

y_predict = scaler_y.inverse_transform(
    y_predict
)

y_predict = y_predict.reshape(-1)


print('최종 y_predict.shape :')
print(y_predict.shape)

# (144,)


# ============================================================
# 9. 실제값과 예측값 비교
# ============================================================

print('========================================')
print('실제 풍향 144개')
print('========================================')
print(y_cor.values)


print('========================================')
print('예측 풍향 144개')
print('========================================')
print(y_predict)


# ============================================================
# 10. RMSE
# ============================================================

rmse = np.sqrt(mean_squared_error(y_cor.values, y_predict))


print('========================================')
print('RMSE :', rmse)
print('========================================')