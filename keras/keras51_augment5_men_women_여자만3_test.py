'''
2026-09-22 (화)
'''

import numpy as np
import time
import datetime

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. 데이터
# ============================================================

start1 = time.time()

np_path = './_save/keras/'

x = np.load(np_path + 'keras46_03_x_train.npy')
y = np.load(np_path + 'keras46_03_y_train.npy')

print('전체 x :', x.shape)
print('전체 y :', y.shape)
print('전체 데이터 분포 :', np.unique(y, return_counts=True))


# ============================================================
# train / test 분리
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.1,
    random_state=2414,
    stratify=y
)

print('x_train :', x_train.shape)
print('y_train :', y_train.shape)
print('x_test  :', x_test.shape)
print('y_test  :', y_test.shape)

print('학습 데이터 분포 :', np.unique(y_train, return_counts=True))
print('테스트 데이터 분포 :', np.unique(y_test, return_counts=True))


# ============================================================
# 여자 데이터만 추출
# ============================================================

x_train_woman = x_train[np.where(y_train > 0.0)]
y_train_woman = y_train[np.where(y_train > 0.0)]

print('여자 데이터 :', x_train_woman.shape)
print('여자 라벨 :', y_train_woman.shape)
print('여자 데이터 분포 :', np.unique(y_train_woman, return_counts=True))


end1 = time.time()

print('데이터 걸린시간 :', round(end1 - start1, 2), 'sec')


# ============================================================
# 데이터 증강 설정
# ============================================================

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    rotation_range=5,
    fill_mode='nearest'
)


# ============================================================
# 증강할 데이터 개수
# ============================================================

augment_size = 8000

print('여자 데이터 개수 :', len(x_train_woman))
print('학습 데이터 분포 :', np.unique(y_train, return_counts=True))


# ============================================================
# 여자 데이터에서 랜덤하게 8000개 선택
# ============================================================

randidx = np.random.randint(
    x_train_woman.shape[0],
    size=augment_size
)

print('randidx 개수 :', len(randidx))
print('최소 index :', np.min(randidx))
print('최대 index :', np.max(randidx))


# ============================================================
# 증강할 이미지와 라벨 생성
# ============================================================

x_augmented = x_train_woman[randidx].copy()
y_augmented = y_train_woman[randidx].copy()

print('x_augmented :', x_augmented.shape)
print('y_augmented :', y_augmented.shape)


# ============================================================
# 이미지 형태 확인 / reshape
# ============================================================

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],
    3
)

print('reshape 후 x_augmented :', x_augmented.shape)


# ============================================================
# 이미지 증강
# ============================================================

x_augmented = train_datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()[0]

print('증강 후 x_augmented :', x_augmented.shape)


# ============================================================
# 기존 데이터 reshape
# ============================================================

x_train = x_train.reshape(
    x_train.shape[0],
    100,
    100,
    3
)

x_test = x_test.reshape(
    x_test.shape[0],
    100,
    100,
    3
)

print('reshape 후')
print('x_train :', x_train.shape)
print('x_test  :', x_test.shape)


# ============================================================
# numpy에서 데이터 합치기
# ============================================================

x_train = np.concatenate(
    (x_train, x_augmented)
)

y_train = np.concatenate(
    (y_train, y_augmented)
)

print('최종 x_train :', x_train.shape)
print('최종 y_train :', y_train.shape)

print('최종 데이터 분포 :', np.unique(y_train, return_counts=True))


# ============================================================
# 데이터 정규화
# ============================================================

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print('x_train 최소값 :', np.min(x_train))
print('x_train 최대값 :', np.max(x_train))
print('x_test 최소값 :', np.min(x_test))
print('x_test 최대값 :', np.max(x_test))


# ============================================================
# 2. 모델구성
# ============================================================

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()


# ============================================================
# 3. 컴파일
# ============================================================

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)


# ============================================================
# EarlyStopping
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=15,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# ModelCheckpoint
# ============================================================

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path = './_save/keras51/05/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = path + 'men-women' + date + '-' + filename

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    filepath=filepath,
    verbose=1
)


# ============================================================
# 4. 훈련
# ============================================================

start_time = time.time()

model.fit(
    x_train,
    y_train,
    epochs=300,
    batch_size=32,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, mcp]
)

end_time = time.time()


# ============================================================
# 5. 평가
# ============================================================

print('------------------ keras51_men-women --------------------')

loss = model.evaluate(
    x_test,
    y_test,
    verbose=1,
    batch_size=8
)

print('loss :', loss[0])
print('acc :', loss[1])


# ============================================================
# 6. 예측
# ============================================================

y_predict = model.predict(x_test)

y_predict = np.round(y_predict)

acc_score = accuracy_score(
    y_test,
    y_predict
)

print('accuracy_score :', acc_score)
print('걸린 시간 :', round(end_time - start_time, 2), 'sec')