'''
2026-09-29

목표: 로스는 0.1이하, 결과는 [101, 102, 103, 104, 105, 106] 의 근사치.

x_predict = np.array(range(96, 106))  # 101~106 까지 찾자.
'''

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 1. 데이터
a = np.array(range(1, 101))
x_predict = np.array(range(96, 106))

size = 6

def split_x(dataset, size):                  
    aaa = []                                 
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]       
        aaa.append(subset)                   
    return np.array(aaa)                     

bbb = split_x(a, size)
# print(bbb)
print('bbb.shape : ', bbb.shape)  # bbb.shape :  (95, 6)

x = bbb[:, :-1]

y = bbb[:,-1]

x_predict = split_x(x_predict, size-1)

print('x : ')
print(x)
print('y : ')
print(y)
# [  6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
#   24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41
#   42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59
#   60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77
#   78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
#   96  97  98  99 100]

print('x_predict : ')
print(x_predict)
# [[ 96  97  98  99 100]
#  [ 97  98  99 100 101]
#  [ 98  99 100 101 102]
#  [ 99 100 101 102 103]
#  [100 101 102 103 104]
#  [101 102 103 104 105]]

print('x.shape, y.shape : ')
print(x.shape, y.shape)
# (95, 5) (95,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print('x.reshape : ')
print(x.shape)
# (95, 5, 1)


#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=16, input_shape=(4, 2)))
model.add(LSTM(32, input_shape=(5, 1)))
# 3차원으로 들어가서 2차원 또는 1차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(         
    monitor = 'loss',        
    mode = 'auto',                 
    patience = 30,                
    verbose = 1,
    restore_best_weights = True,  
)

model.fit(x, y, 
          epochs = 10000,
          batch_size = 32,
          verbose = 1,
          callbacks=[es],
)


#4. 평가 예측
print('------------keras56_split3_1------------------')


results = model.evaluate(x, y)
print('loss : ', results)


x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)

y_predict = model.predict(x_predict)

print('[101, 102, 103, 104, 105, 106]의 예측값 : ')
print(y_predict)
'''
Epoch 216: early stopping
------------keras56_split3_1------------------
3/3 [==============================] - 0s 2ms/step - loss: 0.0204
loss :  0.020353879779577255
1/1 [==============================] - 0s 190ms/step
[101, 102, 103, 104, 105, 106]의 예측값 :  
[[100.1324  ]
 [100.729034]
 [101.277115]
 [101.768486]
 [102.22819 ]
 [102.62313 ]]

 Epoch 264: early stopping
------------keras56_split3_1------------------
3/3 [==============================] - 0s 0s/step - loss: 0.0122
loss :  0.012208903208374977
1/1 [==============================] - 0s 192ms/step
[101, 102, 103, 104, 105, 106]의 예측값 : 
[[100.44954]
 [101.25386]
 [102.03418]
 [102.80687]
 [103.54563]
 [104.24948]]

'''