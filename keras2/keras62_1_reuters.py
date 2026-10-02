from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 1000,    # 단어사전의 갯수, 빈도수가 높은 단어 순으로 1000개 뽑겠다.
    # maxlen = 1000,         # 단어의 최대 길이 제한. 
    test_split = 0.2,
)

print('x_train :')
print(x_train)
# print('x_train.shape, y_train.shape :')
# print(x_train.shape, y_train.shape)    # (8982,) (8982,)
# print('x_test.shape, y_test.shape :')
# print(x_test.shape, y_test.shape)      # (2246,) (2246,)
print('y_train :')
print(y_train)
# [ 3  4  3 ... 25  3 25]
# print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# print(type(x_train))           # <class 'numpy.ndarray'>
# print(type(x_train[0]))        # <class 'list'>
# print(len(x_train[0]),len(x_train[1]))  # 87 56

# print("뉴스기사의 최소길이 :", min(len(i) for i in x_train)) # 뉴스기사의 최소길이 : 13
# print("뉴스기사의 최대길이 :", max(len(i) for i in x_train)) # 뉴스기사의 최대길이 : 2376
# print("뉴스기사의 평균길이 :", sum(map(len, x_train))/ len(x_train)) # 뉴스기사의 평균길이 : 145.5398574927633

plt.hist([len(sample) for sample in x_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
# 전처리 (패드 시퀀스)
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 문장마다 단어 수가 다름 -> 모델은 같은 길이만 받을 수 있어서 모자란 자리를 0 으로 채움
x_train = pad_sequences(x_train,                # padding 할 대상
                         padding='post',   # post -> 뒤를 0으로 채움 / default 는 pre (앞을 0으로 채움)
                         maxlen=250,         # 문장 길이를 5로 맞춤 -> 5보다 긴 문장은 잘림
                         truncating='post' # 잘라낼 때 뒤쪽을 자름 / default 는 pre (앞쪽 잘림)
                         )
x_test = pad_sequences(x_test,
                       maxlen=250,
                       padding='post',
                       truncating='post'
                       )

print(x_train)                            # [[ 2  3  0  0  0] [ 1  4  0  0  0] ... -> 모자란 뒷자리가 0 으로 채워짐
print(x_train.shape)                      # (15, 5)
print(x_test)                           
print(x_test.shape)                            


# y 원핫
y_train = to_categorical(y_train, num_classes=46)
y_test = to_categorical(y_test, num_classes=46)

print(x_train.shape, y_train.shape) # (8982, 200) (8982, 46)
print(x_test.shape, y_test.shape) # (2246, 200) (2246, 46)

# 2. 모델구성
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=100, input_length=200))
model.add(LSTM(100))  
model.add(Dense(46, activation='softmax'))                       
model.summary()

# exit()
# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss',                        # 훈련 loss 를 감시
    mode='min',                            # loss 는 낮을수록 좋음
    patience=20,                          # 100 epoch 동안 최저 loss 가 갱신되지 않으면 멈춤
    restore_best_weights=True,             # 멈춘 시점이 아니라 loss 가 가장 낮았던 epoch 의 가중치로 되돌림
)

start_time = time.time()

model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=64,
    validation_split=0.2,
    callbacks = [es]
)

end_time = time.time()

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('test loss:', loss)
print('test acc:', acc)

y_predict = model.predict(x_test)
y_predcit = np.argmax(y_predict)
acc_score = accuracy_score(y_test, y_predict)

print("acc_score :", acc_score)
print("걸린 시간 :", round(end_time - start_time, 2), "초")


### acc 0.67 이상
# test acc: 0.7097061276435852