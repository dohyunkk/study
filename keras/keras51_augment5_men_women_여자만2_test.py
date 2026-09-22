'''
2026-09-22 (화)

!! 실습
!! 여자 데이터만 증폭해서, 성능 올려봐
!! 기존 keras47 모델을 불러와서 추가 학습

http
'''

import numpy as np
import time
import datetime

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np_path = './_data/save_men-women_npy/'

x = np.load(np_path + 'x.npy')
y = np.load(np_path + 'y.npy')

# print('x shape :', x.shape)             # x shape : (27167, 100, 100, 3)
# print('y shape :', y.shape)             # y shape : (27167,)
# print('남자 데이터 :', np.sum(y == 0))  # 남자 데이터 : 17678
# print('여자 데이터 :', np.sum(y == 1))  # 여자 데이터 : 9489


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=2414,
    stratify=y
)

# print('x_train :', x_train.shape) # x_train : (21733, 100, 100, 3)
# print('y_train :', y_train.shape) # y_train : (21733,)
# print('x_test  :', x_test.shape)  # x_test  : (5434, 100, 100, 3)
# print('y_test  :', y_test.shape)  # y_test  : (5434,)
# print('학습 남자 :', np.sum(y_train == 0)) # 학습 남자 : 14142
# print('학습 여자 :', np.sum(y_train == 1)) # 학습 여자 : 7591
# print('테스트 남자 :', np.sum(y_test == 0)) # 테스트 남자 : 3536
# print('테스트 여자 :', np.sum(y_test == 1)) # 테스트 여자 : 1898


# 여자 데이터만 선택
x_woman = x_train[y_train == 1]
y_woman = y_train[y_train == 1]

# 남자 데이터
x_man = x_train[y_train == 0]
y_man = y_train[y_train == 0]

# print('남자 데이터 :', len(x_man))     # 남자 데이터 : 14142
# print('여자 데이터 :', len(x_woman))   # 여자 데이터 : 7591


# 여자 데이터 증강 설정
woman_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


# 여자 데이터 증강 제너레이터
woman_generator = woman_datagen.flow(
    x_woman,
    y_woman,
    batch_size=32,
    shuffle=True
)


# 남자 데이터 수만큼 여자 데이터 증강
x_woman_aug = []
y_woman_aug = []

while len(x_woman_aug) * 32 < len(x_man):

    batch_x, batch_y = next(woman_generator)

    x_woman_aug.append(batch_x)
    y_woman_aug.append(batch_y)


# 리스트를 numpy 배열로 변환
x_woman_aug = np.concatenate(x_woman_aug, axis=0)
y_woman_aug = np.concatenate(y_woman_aug, axis=0)


# 남자 데이터와 동일한 개수만 사용
x_woman_aug = x_woman_aug[:len(x_man)]
y_woman_aug = y_woman_aug[:len(x_man)]


# print('증강된 여자 데이터 :', len(x_woman_aug))  # 증강된 여자 데이터 : 14119

# ----------------------------------------------------
# 남자 원본 + 증강된 여자 데이터 합치기
# ----------------------------------------------------

x_train_aug = np.concatenate(
    [x_man, x_woman_aug],
    axis=0
)

y_train_aug = np.concatenate(
    [y_man, y_woman_aug],
    axis=0
)

# ----------------------------------------------------
# 데이터 섞기
# ----------------------------------------------------

shuffle_index = np.random.permutation(
    len(x_train_aug)
)

x_train_aug = x_train_aug[shuffle_index]
y_train_aug = y_train_aug[shuffle_index]


# ----------------------------------------------------
# 결과 확인
# ----------------------------------------------------

# print('최종 학습 데이터 :', x_train_aug.shape) # 최종 학습 데이터 : (28261, 100, 100, 3)
# print('최종 라벨 데이터 :', y_train_aug.shape) # 최종 라벨 데이터 : (28261,)

# print('최종 남자 데이터 :', np.sum(y_train_aug == 0)) # 최종 남자 데이터 : 14142
# print('최종 여자 데이터 :', np.sum(y_train_aug == 1)) # 최종 여자 데이터 : 14119



# 기존에 학습했던 모델 불러오기
model = load_model(
    './_save/keras47/03/men-women0921_1702-0029-0.1487.keras'
)

# model.summary()

# ----------------------------------------------------
# 6. 기존 모델 컴파일
# ----------------------------------------------------

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)


# ----------------------------------------------------
# 7. EarlyStopping
# ----------------------------------------------------

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# ----------------------------------------------------
# 8. 추가 학습
# ----------------------------------------------------

start_time = time.time()

model.fit(
    x_train_aug,
    y_train_aug,
    epochs=100,
    batch_size=32,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[es]
)

end_time = time.time()

# ----------------------------------------------------
# 최종 평가
# ----------------------------------------------------

print("------------------ keras51_aug_men_women_test --------------------")

loss = model.evaluate(
    x_test,
    y_test,
    verbose=1,
    batch_size=32
)

print('최종 loss :', loss[0])
print('최종 acc  :', loss[1])


# ----------------------------------------------------
# Accuracy Score
# ----------------------------------------------------

y_predict = model.predict(x_test)

y_predict = np.round(y_predict).ravel().astype(int)

acc_score = accuracy_score(
    y_test,
    y_predict
)

print('accuracy_score :', acc_score)
print('학습 시간 :', round(end_time - start_time, 2), 'sec')

#★★★★★★★ mcp 세이브 파일명 만들기 시작 ★★★★★★★##
import datetime

date = datetime.datetime.now()
print(date)         # 2026-09-14 11:42:10.635267
print(type(date))   # <class 'datetime.datetime'>        # ★ calss
date = date.strftime("%m%d_%H%M")
print(date)         # 2026-09-14 11:48:27.764409
print(type(date))   # <class 'datetime.datetime'>

path = './_save/keras51/05/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'men_women_test', date, "-",filename])

#★★★★★★★ mcp 세이브 파일명 만들기 끝 ★★★★★★★##

'''
------------------ keras51_aug_men_women --------------------
170/170 [==============================] - 1s 4ms/step - loss: 0.1377 - acc: 0.9468
최종 loss : 0.13769328594207764
최종 acc  : 0.9468163251876831
accuracy_score : 0.9468163415531836
학습 시간 : 68.49 sec
'''