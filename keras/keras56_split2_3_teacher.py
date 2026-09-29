'''
2026-09-28 (월)

'''
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

a = np.array([[1,2,3,4,5,6,7,8,9,10],        # 삼성전자 주가
              [9,8,7,6,5,4,3,2,1,0],         # 해당 날짜의 온도
              ]).T   
# print(a.shape)    # (10, 2)

size = 5

def split_x(dataset, size):                  
    aaa = []                                 
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i+size)]       
        aaa.append(subset)                   
    return np.array(aaa)                     

bbb = split_x(a, size)
# print(bbb)
print('bbb.shape : ', bbb.shape)       # (6, 5, 2)  

x = bbb[:, :-1, :]                 # batch_size
# x = bbb[:, :-1]

y = bbb[:, -1, 0]                  # batch_size  
# y = bbb[:, -1, -1]

print('x : ')
print(x)
print('y : ')
print(y)
# [5 4 3 2 1 0]
# [5 4 3 2 1 0]
print(x.shape, y.shape)   # (6, 4, 2) (6,)

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=16, input_shape=(4, 2)))
model.add(LSTM(32, input_shape=(4, 2)))
# 3차원으로 들어가서 2차원 또는 1차원으로 나옴 -> 바로 Dense와 연결가능
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
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
print('------------keras56_split2_3_teacher------------------')


results = model.evaluate(x, y)
print('loss : ', results)


x_predict = np.array([
    [7, 3],
    [8, 2],
    [9, 1],
    [10, 0]
]).reshape(1, 4, 2)
y_predict = model.predict(x_predict)


# print('다음 날 온도 예측값 : ', y_predict) # y = bbb[:, -1, 1]
print('다음 날 주가 예측값 : ', y_predict)  # y = bbb[:, -1, 0]


'''
Epoch 744: early stopping
1/1 [==============================] - 0s 202ms/step - loss: 9.7215e-05
loss :  9.721534297568724e-05
1/1 [==============================] - 0s 187ms/step
다음 날 주가 예측값 :  [[10.5769005]]

'''