from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding

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

# 전처리 (패드 시퀀스)
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 문장마다 단어 수가 다름 (1개 ~ 5개) -> 모델은 같은 길이만 받을 수 있어서 모자란 자리를 0 으로 채움
padded_x_train = pad_sequences(x_train,                # padding 할 대상
                         padding='post',   # post -> 뒤를 0으로 채움 / default 는 pre (앞을 0으로 채움)
                         maxlen=5,         # 문장 길이를 5로 맞춤 -> 5보다 긴 문장은 잘림
                         truncating='post' # 잘라낼 때 뒤쪽을 자름 / default 는 pre (앞쪽 잘림)
                         )

print(padded_x_train)                            # [[ 2  3  0  0  0] [ 1  4  0  0  0] ... -> 모자란 뒷자리가 0 으로 채워짐
print(padded_x_train.shape)                      # (15, 5)
# y 원핫



# 2. 모델구성
model = Sequential()
model.add(Embedding(input_dim=32, output_dim=100, input_length=5))
                    # 단어 사전 크기 (단어 종류 + 1), 출력 차원 (단어 하나를 몇 칸짜리 벡터로 만들지), 입력 시퀀스 길이 (padding 한 길이)
model.add(LSTM(10))                   # Embedding 출력 (None, 5, 100) 이 3차원이라 RNN 계열에 바로 넣을 수 있음 (reshape 필요 없음)
model.add(Dense(1, activation='softmax'))                       
model.summary()


# softmax, categorical_crossentropy


### acc 0.67 이상