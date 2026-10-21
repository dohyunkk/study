# 삼성전자 주가맞추기
# 앙상블 쓸 것
# LSTM Con1D 쓸것

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ============================================================
# 1. 데이터 불러오기
# ============================================================

path = "c:/study/_data/samsung/"

samsung = pd.read_csv(
    path + "삼성전자261012.csv",
    index_col=0,
    encoding="cp949",
)

amore = pd.read_csv(
    path + "아모레261012.csv",
    index_col=0,
    encoding="cp949",
)

hynix = pd.read_csv(
    path + "하이닉스261012.csv",
    index_col=0,
    encoding="cp949",
)

print("samsung:", samsung.shape)
print("amore:", amore.shape)
print("hynix:", hynix.shape)


# ============================================================
# 2. 필요한 컬럼 선택과 전처리
# ============================================================

# 오래된 날짜부터 정렬
samsung.index = pd.to_datetime(samsung.index, errors="coerce")
amore.index = pd.to_datetime(amore.index, errors="coerce")
hynix.index = pd.to_datetime(hynix.index, errors="coerce")

samsung = samsung.loc[~samsung.index.isna()].sort_index()
amore = amore.loc[~amore.index.isna()].sort_index()
hynix = hynix.loc[~hynix.index.isna()].sort_index()


# 아모레와 하이닉스에서 사용할 컬럼 6개
feature_cols = ["시가", "고가", "저가", "종가", "등락률", "거래량"]

# 필요한 컬럼만 각각 따로 선택
amore_data = amore[feature_cols].copy()
hynix_data = hynix[feature_cols].copy()


# 쉼표가 포함된 숫자를 숫자 형식으로 변경
# 예: '260,000' -> 260000
for col in feature_cols:
    amore_data[col] = pd.to_numeric(
        amore_data[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    hynix_data[col] = pd.to_numeric(
        hynix_data[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

# 정답으로 사용할 삼성전자 시가도 숫자로 변경
samsung["시가"] = pd.to_numeric(
    samsung["시가"].astype(str).str.replace(",", "", regex=False),
    errors="coerce",
)

# 결측값이 있는 행 제거
amore_data = amore_data.dropna()
hynix_data = hynix_data.dropna()


# 세 종목 모두 데이터가 있는 날짜만 사용
common_dates = amore_data.index.intersection(hynix_data.index)
common_dates = common_dates.intersection(samsung.index)
common_dates = common_dates.sort_values()

# 각 날짜의 다음 거래일 삼성전자 시가
samsung_next_open = samsung["시가"].shift(-1)


# ============================================================
# 3. 최근 60거래일씩 묶기
# ============================================================

size = 60

amore_x = []
hynix_x = []
samsung_y = []

for i in range(size - 1, len(common_dates)):
    dates_60 = common_dates[i - size + 1 : i + 1]
    today = common_dates[i]

    # 다음 거래일 삼성전자 시가가 없으면 건너뜀
    if pd.isna(samsung_next_open.loc[today]):
        continue

    # 아모레와 하이닉스 데이터는 각각 별도로 보관
    amore_x.append(amore_data.loc[dates_60, feature_cols].to_numpy())

    hynix_x.append(hynix_data.loc[dates_60, feature_cols].to_numpy())

    # 정답은 다음 거래일 삼성전자 시가
    samsung_y.append(samsung_next_open.loc[today])


amore_x = np.array(amore_x, dtype=np.float32)
hynix_x = np.array(hynix_x, dtype=np.float32)
samsung_y = np.array(samsung_y, dtype=np.float32).reshape(-1, 1)

print("amore_x:", amore_x.shape)  # (샘플 수, 60, 6)
print("hynix_x:", hynix_x.shape)  # (샘플 수, 60, 6)
print("samsung_y:", samsung_y.shape)  # (샘플 수, 1)


# ============================================================
# 4. 시간 순서대로 학습 / 검증 / 테스트 분리
# ============================================================

# 앞 95% 학습, 그다음 3% 검증, 마지막 2% 테스트
train_end = int(len(samsung_y) * 0.95)
val_end = int(len(samsung_y) * 0.98)

amore_train = amore_x[:train_end]
amore_val = amore_x[train_end:val_end]
amore_test = amore_x[val_end:]

hynix_train = hynix_x[:train_end]
hynix_val = hynix_x[train_end:val_end]
hynix_test = hynix_x[val_end:]

samsung_train = samsung_y[:train_end]
samsung_val = samsung_y[train_end:val_end]
samsung_test = samsung_y[val_end:]

print("학습:", amore_train.shape, hynix_train.shape, samsung_train.shape)
print("검증:", amore_val.shape, hynix_val.shape, samsung_val.shape)
print("테스트:", amore_test.shape, hynix_test.shape, samsung_test.shape)


# ============================================================
# 5. 스케일링
# ============================================================

num_features = len(feature_cols)

# 아모레 입력 스케일링
amore_train_shape = amore_train.shape
amore_val_shape = amore_val.shape
amore_test_shape = amore_test.shape

scaler_amore = StandardScaler()

amore_train = (
    scaler_amore.fit_transform(amore_train.reshape(-1, num_features))
    .reshape(amore_train_shape)
    .astype(np.float32)
)

amore_val = (
    scaler_amore.transform(amore_val.reshape(-1, num_features))
    .reshape(amore_val_shape)
    .astype(np.float32)
)

amore_test = (
    scaler_amore.transform(amore_test.reshape(-1, num_features))
    .reshape(amore_test_shape)
    .astype(np.float32)
)


# 하이닉스 입력 스케일링
hynix_train_shape = hynix_train.shape
hynix_val_shape = hynix_val.shape
hynix_test_shape = hynix_test.shape

scaler_hynix = StandardScaler()

hynix_train = (
    scaler_hynix.fit_transform(hynix_train.reshape(-1, num_features))
    .reshape(hynix_train_shape)
    .astype(np.float32)
)

hynix_val = (
    scaler_hynix.transform(hynix_val.reshape(-1, num_features))
    .reshape(hynix_val_shape)
    .astype(np.float32)
)

hynix_test = (
    scaler_hynix.transform(hynix_test.reshape(-1, num_features))
    .reshape(hynix_test_shape)
    .astype(np.float32)
)


# 삼성전자 시가 정답 스케일링
samsung_test_original = samsung_test.copy()

scaler_y = StandardScaler()

samsung_train = scaler_y.fit_transform(samsung_train).astype(np.float32)
samsung_val = scaler_y.transform(samsung_val).astype(np.float32)
samsung_test_scaled = scaler_y.transform(samsung_test).astype(np.float32)


# ============================================================
# 6. 최근 샘플에 더 큰 가중치 부여
# ============================================================

# 학습 초반 샘플의 가중치는 0.5,
# 가장 최근 학습 샘플의 가중치는 2.0
sample_weight = np.linspace(
    0.5,
    2.0,
    len(samsung_train),
).astype(np.float32)

print("학습 샘플 가중치:", sample_weight[0], "~", sample_weight[-1])


# ============================================================
# 7. 모델 구성
# 아모레 입력과 하이닉스 입력을 각각 처리한 뒤 합침
# ============================================================

# # 아모레 입력 갈래: LSTM -> Conv1D
# input1 = Input(shape=(size, num_features), name='amore_input')

# lstm1 = LSTM(64, return_sequences=True)(input1)
# conv1 = Conv1D(
#     filters=32,
#     kernel_size=3,
#     padding='causal',
#     activation='relu',
# )(lstm1)
# flat1 = Flatten()(conv1)
# dense1 = Dense(32, activation='relu')(flat1)
# dense1 = Dropout(0.2)(dense1)


# # 하이닉스 입력 갈래: LSTM -> Conv1D
# input2 = Input(shape=(size, num_features), name='hynix_input')

# lstm2 = LSTM(64, return_sequences=True)(input2)
# conv2 = Conv1D(
#     filters=32,
#     kernel_size=3,
#     padding='causal',
#     activation='relu',
# )(lstm2)
# flat2 = Flatten()(conv2)
# dense2 = Dense(32, activation='relu')(flat2)
# dense2 = Dropout(0.2)(dense2)


# # 두 갈래의 결과를 합쳐 삼성전자 시가 예측
# merge = Concatenate()([dense1, dense2])
# dense3 = Dense(32, activation='relu')(merge)
# last_output = Dense(1, name='samsung_next_open')(dense3)

# model = Model(
#     inputs=[input1, input2],
#     outputs=last_output,
# )

# model.summary()


model = load_model("./save/keras70/" + "model.keras")
# ============================================================
# 8. 컴파일 및 훈련
# ============================================================

# model.compile(
#     loss='mse',
#     optimizer='adam',
#     metrics=['mae'],
# )

# checkpoint_path = './save/keras70'
# os.makedirs(checkpoint_path, exist_ok=True)

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=40,
#     restore_best_weights=True,
#     verbose=1,
# )

# cp = ModelCheckpoint(
#     filepath=checkpoint_path + 'amore_hynix_recent_best.weights.h5',
#     monitor='val_loss',
#     mode='min',
#     save_best_only=True,
#     save_weights_only=True,
#     verbose=1,
# )

# start_time = time.time()

# model.fit(
#     [amore_train, hynix_train],
#     samsung_train,
#     sample_weight=sample_weight,
#     epochs=10000,
#     batch_size=32,
#     validation_data=([amore_val, hynix_val], samsung_val),
#     shuffle=False,
#     callbacks=[es, cp],
#     verbose=1,
# )

# model.save('./save/keras70/'+'model.keras')

# print('훈련 시간:', round(time.time() - start_time, 2), '초')

# 검증 손실이 가장 낮았던 가중치 불러오기
# model.load_weights(
#     checkpoint_path + 'amore_hynix_recent_best.weights.h5'
# )


# ============================================================
# 9. 테스트 데이터 평가
# ============================================================

y_predict_scaled = model.predict([amore_test, hynix_test])

# 예측값을 실제 삼성전자 시가 단위로 되돌림
y_predict = scaler_y.inverse_transform(y_predict_scaled).reshape(-1)

samsung_test_original = samsung_test_original.reshape(-1)

mae = mean_absolute_error(samsung_test_original, y_predict)
rmse = np.sqrt(mean_squared_error(samsung_test_original, y_predict))

print("테스트 MAE:", round(mae, 2))
print("테스트 RMSE:", round(rmse, 2))


# ============================================================
# 10. 최신 데이터로 다음 거래일 삼성전자 시가 예측
# ============================================================

latest_dates = common_dates[-size:]

latest_amore = amore_data.loc[latest_dates, feature_cols].to_numpy(dtype=np.float32)

latest_hynix = hynix_data.loc[latest_dates, feature_cols].to_numpy(dtype=np.float32)

latest_amore = (
    scaler_amore.transform(latest_amore.reshape(-1, num_features))
    .reshape(1, size, num_features)
    .astype(np.float32)
)

latest_hynix = (
    scaler_hynix.transform(latest_hynix.reshape(-1, num_features))
    .reshape(1, size, num_features)
    .astype(np.float32)
)

latest_predict_scaled = model.predict([latest_amore, latest_hynix])

latest_predict = scaler_y.inverse_transform(latest_predict_scaled)[0, 0]

print("마지막 입력 날짜:", latest_dates[-1].strftime("%Y-%m-%d"))
print("다음 거래일 삼성전자 시가 예측:", round(float(latest_predict), 2))


"""
Epoch 65: val_loss did not improve from 0.06886
32/32 [==============================] - 0s 10ms/step - loss: 0.1612 - mae: 0.1728 - val_loss: 0.4589 - val_mae: 0.6282
Epoch 65: early stopping
훈련 시간: 30.63 초
1/1 [==============================] - 0s 360ms/step
테스트 MAE: 14115.2
테스트 RMSE: 16409.35
1/1 [==============================] - 0s 34ms/step
마지막 입력 날짜: 2026-10-12
다음 거래일 삼성전자 시가 예측: 252504.69
"""
