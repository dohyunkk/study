# 실습
# acc 기준 0.6
# Embedding + LSTM/GRU 
# Embedding + Bidirectional  
# Embedding + Flatten + DNN 
# 비교


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, GRU, Bidirectional, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tensorflow.keras.datasets import imdb



(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = 1000,        # 단어사전의 갯수, 빈도수가 높은 단어 순으로 1000개 뽑겠다.
    # maxlen = 1000,         # 단어의 최대 길이 제한. 
    # test_split = 0.2,
)

print('x_train :')
print(x_train)
print('x_train.shape, y_train.shape :')
print(x_train.shape, y_train.shape)    # (25000,) (25000,)
print('x_test.shape, y_test.shape :')
print(x_test.shape, y_test.shape)      # (25000,) (25000,)
print('y_train :')
print(y_train)
# [1 0 0 ... 0 1 0]
print(np.unique(y_train))
# [0 1]

print(type(x_train))           # <class 'numpy.ndarray'>
print(type(x_train[0]))        # <class 'list'>
print(len(x_train[0]),len(x_train[1]))  # 218 189

print(" 최소길이 :", min(len(i) for i in x_train)) #  최소길이 : 11
print(" 최대길이 :", max(len(i) for i in x_train)) #  최대길이 : 2494
print(" 평균길이 :", sum(map(len, x_train))/ len(x_train)) # 평균길이 : 238.71364

# plt.hist([len(sample) for sample in x_train], bins=50)
# plt.xlabel('length of samples')
# plt.ylabel('number of samples')
# plt.show()


# 전처리 (패드 시퀀스)
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 문장마다 단어 수가 다름 -> 모델은 같은 길이만 받을 수 있어서 모자란 자리를 0 으로 채움
x_train = pad_sequences(x_train,                # padding 할 대상
                         padding='post',   # post -> 뒤를 0으로 채움 / default 는 pre (앞을 0으로 채움)
                         maxlen=250,         # 문장 길이를 200로 맞춤 -> 200보다 긴 문장은 잘림
                         truncating='post' # 잘라낼 때 뒤쪽을 자름 / default 는 pre (앞쪽 잘림)
                         )
x_test = pad_sequences(x_test,
                       maxlen=250,
                       padding='post',
                       truncating='post'
                       )

print(x_train)                            # [[ 2  3  0  0  0] [ 1  4  0  0  0] ... -> 모자란 뒷자리가 0 으로 채워짐
print(x_train.shape)                      # (25000, 200)
print(x_test)                           
print(x_test.shape)                       # (25000, 200)

print(np.unique(y_train, return_counts=True)) # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
print(np.unique(y_test, return_counts=True)) # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))


# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
# model = Sequential()
# model.add(Embedding(input_dim=1000, output_dim=100))
# model.add(LSTM(100))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))    

# model = Sequential()
# model.add(Embedding(input_dim=1000, output_dim=100))
# model.add(GRU(100))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))  

# model = Sequential()
# model.add(Embedding(input_dim=1000, output_dim=100))
# model.add(Bidirectional(LSTM(100)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))  


# model = Sequential()
# model.add(Embedding(input_dim=1000, output_dim=100, input_length=250))
# model.add(GRU(100, return_sequences=True))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100, input_length=250))
# model.add(GRU(100, return_sequences=True))
model.add(Flatten())
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# exit()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc'],
              )

# patience 를 크게 줘서 loss 를 끝까지 낮춤 -> 예측값이 0.5 근처에서 멈추지 않고 0 또는 1 쪽으로 확실해짐
es = EarlyStopping(
    monitor='val_loss',                    
    mode='min',                            # loss 는 낮을수록 좋음
    patience=20,                           # 최저 loss 가 갱신되지 않으면 멈춤
    restore_best_weights=True,             # 멈춘 시점이 아니라 loss 가 가장 낮았던 epoch 의 가중치로 되돌림
)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=128,   
          verbose=1,
          callbacks=[es],
          validation_split=0.1,
          )
end_time = time.time()

#4. 평가 예측
loss = model.evaluate(x_test, y_test)      # [loss, acc] 리스트로 돌려줌 -> loss[0] = loss, loss[1] = acc
print("loss :", loss[0])
print("acc :", round(loss[1], 4))

y_predict = model.predict(x_test)          # test 5문장의 sigmoid 확률 (0 ~ 1)
y_predict = np.round(y_predict)            # 0.5 기준 반올림 -> 0 / 1
acc_score = accuracy_score(y_test, y_predict)   # 직접 계산한 정확도 -> 위의 evaluate acc 와 같은 값

print("acc_score :", acc_score)
print("걸린 시간 :", round(end_time - start_time, 2), "초")

'''
# Embedding + LSTM
# loss : 0.6193777918815613
# acc : 0.6611
# 782/782 [==============================] - 5s 6ms/step
# acc_score : 0.66108
# 걸린 시간 : 80.57 초

# Embedding + GRU 
Epoch 27/1000
176/176 [==============================] - 3s 20ms/step - loss: 0.1468 - acc: 0.9505 - val_loss: 0.4997 - val_acc: 0.8348
782/782 [==============================] - 5s 6ms/step - loss: 0.3217 - acc: 0.8600
loss : 0.32173028588294983
acc : 0.86
782/782 [==============================] - 5s 6ms/step
acc_score : 0.85996
걸린 시간 : 100.08 초

# Embedding + Bidirectional
Epoch 22/1000
176/176 [==============================] - 6s 36ms/step - loss: 0.1174 - acc: 0.9568 - val_loss: 0.7164 - val_acc: 0.8184
782/782 [==============================] - 9s 12ms/step - loss: 0.3506 - acc: 0.8483
loss : 0.35060596466064453
acc : 0.8483
782/782 [==============================] - 9s 11ms/step
acc_score : 0.84832
걸린 시간 : 147.3 초

# Embedding + GRU + Flatten
Epoch 22/1000
176/176 [==============================] - 4s 20ms/step - loss: 5.1702e-05 - acc: 1.0000 - val_loss: 1.6299 - val_acc: 0.8136
782/782 [==============================] - 5s 7ms/step - loss: 0.3571 - acc: 0.8417
loss : 0.35706040263175964
acc : 0.8417
782/782 [==============================] - 5s 6ms/step
acc_score : 0.84172
걸린 시간 : 84.13 초

# Embedding + Flatten
Epoch 22/1000
176/176 [==============================] - 0s 3ms/step - loss: 3.7437e-04 - acc: 1.0000 - val_loss: 1.0524 - val_acc: 0.8148
782/782 [==============================] - 1s 1ms/step - loss: 0.3804 - acc: 0.8262
loss : 0.38043057918548584
acc : 0.8262
782/782 [==============================] - 1s 750us/step
acc_score : 0.82616
걸린 시간 : 11.68 초

'''