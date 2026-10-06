'''
2026-09-29

https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

x = (n, 144, 13)
y = (n, 144, 1)

2016-12-31 00:00~ 2017-01-01 00:00의 T (degC)을 맞춰보자
'''
# import os
# os.environ["TF_GRU_ALLOCATOR"] = "cuda_malloc_async"    # 메모리 모으기

import numpy as np
import pandas as pd
import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# 1. 데이터

path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# print(datasets.shape) # (420551, 14)

# 수정 : T (degC) < 이놈을 y로 잡는다.

y_cor = datasets[-144:]['T (degC)'] # 예측치 정답 데이터
# print(y_cor.shape)      # (144,)



#### 훈련할 데이터 자르기 ####
start_time = time.time()

x_data = datasets[:-288].drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)
y_data = datasets[144:-144]['T (degC)'].to_numpy(dtype=np.float32)

print(x_data.shape)      # (420263, 13)
print(y_data.shape)      # (420263),)

size_x = 144
size_y = 144

def split_x(dataset, size):                  
    aaa = []                                 
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]       
        aaa.append(subset)                   
    return np.array(aaa)

x = split_x(x_data, size_x)
y = split_x(y_data, size_y)

end_time = time.time()
#### 자르기 끝 ####

print('x, y:', x.shape, y.shape)
print("자르는시간:", round(end_time - start_time, 2),'sec')
# 자르는시간: 24.31 sec

# 2. 선생님 코드와 같은 train/test 분리.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False
)


#스케일링. fit은 train에만 한다.
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_shape = x_train.shape
x_test_shape = x_test.shape
x_train = scaler_x.fit_transform(x_train.reshape(-1, 13)).reshape(x_train_shape)
x_test = scaler_x.transform(x_test.reshape(-1, 13)).reshape(x_test_shape)

y_train_shape = y_train.shape
y_test_shape = y_test.shape
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train_shape)
y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test_shape)

print('x_train, y_train:', x_train.shape, y_train.shape)


# 2, 3. 모델구성, 컴파일 훈련 불러오기.


model = load_model('./_save/keras58/01/jena_easy_0930_1744-0080-0.0992.keras')



# 4. 평가, 예측
x_predict = datasets[-288:-144].drop(['T (degC)'], axis=1).to_numpy(dtype=np.float32)
x_predict = scaler_x.transform(x_predict).reshape(1, 144, 13)

y_predict = model.predict(x_predict).reshape(-1, 1)
y_predict = scaler_y.inverse_transform(y_predict).reshape(-1)

rmse = np.sqrt(mean_squared_error(y_cor.to_numpy(), y_predict))
print('실제 기온:', y_cor.to_numpy())
print('예측 기온:', y_predict)
print('RMSE (°C):', rmse)
# print('훈련 시간 : ', round(end_fit - start_fit, 2),'sec')



'''
실제 기온: [-4.17 -4.17 -4.09 -4.04 -4.14 -4.3  -4.26 -4.58 -4.68 -4.82 -4.96 -5.08
 -5.05 -5.25 -5.34 -5.15 -5.07 -4.96 -4.67 -4.17 -4.07 -3.97 -4.14 -4.15
 -4.47 -4.61 -4.86 -4.92 -5.12 -5.22 -4.9  -4.86 -4.99 -4.92 -5.07 -5.27
 -5.17 -5.37 -5.43 -5.62 -5.77 -6.09 -6.2  -6.04 -6.26 -6.35 -6.46 -6.8
 -6.84 -6.64 -6.87 -6.31 -6.42 -6.36 -6.31 -6.37 -6.09 -5.96 -5.6  -5.24
 -5.02 -4.48 -4.6  -4.41 -4.15 -3.76 -3.31 -2.96 -2.52 -1.92 -1.89 -1.19
 -0.71 -0.2   0.58  1.26  1.57  1.64  1.97  2.17  2.55  3.08  3.92  3.97
  4.15  4.41  4.78  5.19  5.21  5.09  4.97  4.6   4.42  3.98  3.56  2.87
  2.44  2.13  1.98  1.67  1.61  1.41  1.29  0.79  0.44  0.24 -0.06 -0.08
 -0.42 -0.95 -1.28 -1.67 -1.25 -1.03 -0.98 -1.4  -1.47 -1.61 -1.65 -1.52
 -1.4  -2.15 -3.19 -3.3  -3.46 -3.09 -2.75 -2.61 -2.51 -2.48 -2.48 -2.59
 -2.89 -3.22 -4.08 -4.45 -4.09 -3.76 -3.93 -4.05 -3.35 -3.16 -4.23 -4.82]
예측 기온: [-3.8751266e+00 -3.9560611e+00 -4.0208750e+00 -4.0807247e+00
 -4.1240635e+00 -4.1586809e+00 -4.2142143e+00 -4.2589931e+00
 -4.3106413e+00 -4.3190050e+00 -4.3563356e+00 -4.3731747e+00
 -4.3892469e+00 -4.4232502e+00 -4.4253073e+00 -4.4542513e+00
 -4.4756641e+00 -4.5020418e+00 -4.4981346e+00 -4.5149174e+00
 -4.5213356e+00 -4.5108194e+00 -4.5301781e+00 -4.5215654e+00
 -4.5283375e+00 -4.5605392e+00 -4.5697842e+00 -4.5944452e+00
 -4.6226282e+00 -4.6056967e+00 -4.6282663e+00 -4.6150513e+00
 -4.5999928e+00 -4.6223583e+00 -4.6001291e+00 -4.5980244e+00
 -4.6390047e+00 -4.6327944e+00 -4.6352415e+00 -4.7084007e+00
 -4.6847849e+00 -4.6747017e+00 -4.7095451e+00 -4.7469368e+00
 -4.7550306e+00 -4.7315559e+00 -4.7741165e+00 -4.7517443e+00
 -4.7702322e+00 -4.6926546e+00 -4.7158585e+00 -4.7294035e+00
 -4.6993656e+00 -4.6551723e+00 -4.5886068e+00 -4.5689268e+00
 -4.4816761e+00 -4.4421110e+00 -4.3308935e+00 -4.2096529e+00
 -4.1009216e+00 -3.9821289e+00 -3.8454168e+00 -3.7032688e+00
 -3.5264881e+00 -3.3613756e+00 -3.1143930e+00 -2.9104536e+00
 -2.7381113e+00 -2.5460413e+00 -2.2684925e+00 -2.0446842e+00
 -1.8015668e+00 -1.5619380e+00 -1.3164632e+00 -1.0682971e+00
 -8.1140500e-01 -5.9056169e-01 -2.4848539e-01 -6.3401007e-04
  2.2433680e-01  4.7954482e-01  7.3868006e-01  9.3878669e-01
  1.0988233e+00  1.3719275e+00  1.4508231e+00  1.6279814e+00
  1.8006976e+00  1.9558666e+00  2.0107133e+00  2.1324756e+00
  2.1696870e+00  2.2192161e+00  2.2348125e+00  2.3160226e+00
  2.3053133e+00  2.3096683e+00  2.3041050e+00  2.2704909e+00
  2.1936085e+00  2.1042168e+00  1.9885347e+00  1.8816855e+00
  1.8178408e+00  1.6910150e+00  1.5807984e+00  1.4272473e+00
  1.3106167e+00  1.2044671e+00  1.0262401e+00  8.0571765e-01
  6.6089553e-01  5.4457587e-01  3.9157695e-01  2.6547354e-01
  5.8700744e-02 -1.2638742e-01 -2.3527700e-01 -3.9777929e-01
 -5.3123742e-01 -6.8142873e-01 -8.1227857e-01 -9.5011503e-01
 -1.0369241e+00 -1.1781318e+00 -1.3027351e+00 -1.3672540e+00
 -1.4909256e+00 -1.5876567e+00 -1.6020038e+00 -1.6984642e+00
 -1.8462293e+00 -1.8644445e+00 -1.8920391e+00 -2.0319650e+00
 -2.0751064e+00 -2.1411397e+00 -2.2954853e+00 -2.3095386e+00
 -2.3949869e+00 -2.4017446e+00 -2.4435499e+00 -2.4814727e+00]
RMSE (°C): 1.5288290082473612

'''