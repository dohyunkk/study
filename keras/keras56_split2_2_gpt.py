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

size = 3

def split_xy(dataset, size):
    x = []
    y = []

    for i in range(len(dataset) - size):
        x.append(dataset[i:i+size])
        y.append(dataset[i+size, 0])

    return np.array(x), np.array(y)

x, y = split_xy(a, size)
# print(bbb)




print('x : ')
print(x)

print('y : ')
print(y)

print('x.shape :', x.shape)    # x.shape : (7, 3, 2)
print('y.shape :', y.shape)    # y.shape : (7,)


# exit()

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, input_shape=(2, 1)))
model.add(LSTM(16, input_shape=(3, 2)))
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
print('------------keras56_split2_2_gpt------------------')

results = model.evaluate(x, y)
print('loss : ', results)


x_predict = np.array([
    [8, 2],
    [9, 1],
    [10, 0]
]).reshape(1, 3, 2)
y_predict = model.predict(x_predict)

# print('다음 날 온도 예측값 : ', y_predict) # y = bbb[:, -1, 1]
print('다음 날 주가 예측값 : ', y_predict)  # y = bbb[:, -1, 0]

'''
Epoch 1039: early stopping
1/1 [==============================] - 0s 208ms/step - loss: 1.3834e-08
loss :  1.383438874569265e-08
1/1 [==============================] - 0s 187ms/step
다음 날 주가 예측값 :  [[10.391408]]

'''
