'''
2026-09-22 (화)
'''

import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import fashion_mnist


# =========================================================
# 1. 데이터
# =========================================================

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# =========================================================
# 데이터 증강
# =========================================================

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,       # 수평 뒤집기
    # vertical_flip=True,       # 수직 뒤집기
    width_shift_range=0.1,      # 가로 평행이동
    # height_shift_range=0.1,   # 세로 평행이동
    rotation_range=30,          # 회전
    # zoom_range=1.1,            # 확대/축소
    # shear_range=0.7,           # 찌그러뜨리기
    fill_mode='nearest',
)


augment_size = 40000


# =========================================================
# 4만 개 랜덤 추출
# =========================================================

# replace=False → 중복 없이 4만 개 추출
randidx = np.random.choice(
    x_train.shape[0],
    size=augment_size,
    replace=False
)

print('randidx 최소값 :', np.min(randidx))
print('randidx 최대값 :', np.max(randidx))
print('randidx shape :', randidx.shape)


# =========================================================
# 증강할 데이터 복사
# =========================================================

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print('증강 전 :', x_augmented.shape, y_augmented.shape)


# =========================================================
# CNN 입력 형태로 변경
# (40000, 28, 28)
#        ↓
# (40000, 28, 28, 1)
# =========================================================

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],
    1
)

print('reshape 후 :', x_augmented.shape)


# =========================================================
# 데이터 증강 실행
# =========================================================

xy_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False
).next()


# flow에서 나온 결과
x_augmented = xy_augmented[0]
y_augmented = xy_augmented[1]


print('증강 완료 :', x_augmented.shape)
print('증강 label :', y_augmented.shape)


# =========================================================
# 원본 데이터 reshape
# =========================================================

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


# =========================================================
# 정규화
#
# 원본 데이터는 /255
# 증강 데이터는 ImageDataGenerator에서 이미 /255
# =========================================================

x_train = x_train / 255.
x_test = x_test / 255.


# =========================================================
# 원본 + 증강 데이터 합치기
# =========================================================

x_train = np.concatenate(
    (x_train, x_augmented),
    axis=0
)

y_train = np.concatenate(
    (y_train, y_augmented),
    axis=0
)


print('최종 x_train :', x_train.shape)
print('최종 y_train :', y_train.shape)

# 결과
# (100000, 28, 28, 1)
# (100000,)


# =========================================================
# 클래스별 데이터 개수 확인
# =========================================================

print('클래스별 개수 :')
print(np.unique(y_train, return_counts=True))


# =========================================================
# One-hot encoding
# =========================================================

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# =========================================================
# 2. 모델 구성
# =========================================================

model = Sequential()


# 첫 번째 Conv Block
model.add(
    Conv2D(
        32,
        (3, 3),
        padding='same',
        activation='relu',
        input_shape=(28, 28, 1)
    )
)

model.add(MaxPooling2D())
model.add(Dropout(0.2))


# 두 번째 Conv Block
model.add(
    Conv2D(
        64,
        (3, 3),
        padding='same',
        activation='relu'
    )
)

model.add(MaxPooling2D())
model.add(Dropout(0.2))


# 세 번째 Conv Block
model.add(
    Conv2D(
        128,
        (3, 3),
        padding='same',
        activation='relu'
    )
)

model.add(MaxPooling2D())
model.add(Dropout(0.3))


# 네 번째 Conv Block
model.add(
    Conv2D(
        256,
        (3, 3),
        padding='same',
        activation='relu'
    )
)

model.add(MaxPooling2D())
model.add(Dropout(0.3))


# Fully Connected
model.add(Flatten())

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        10,
        activation='softmax'
    )
)


model.summary()


# =========================================================
# 3. 컴파일
# =========================================================

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)


# =========================================================
# EarlyStopping
# =========================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=15,
    restore_best_weights=True,
    verbose=1
)


# =========================================================
# 훈련
# =========================================================

start_time = time.time()


history = model.fit(
    x_train,
    y_train,
    epochs=500,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[es]
)


end_time = time.time()


# =========================================================
# 4. 평가
# =========================================================

print(
    "------------------ 51_fashion_model.evaluate --------------------"
)

loss = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print('loss :', loss[0])
print('acc  :', loss[1])


# =========================================================
# 예측
# =========================================================

y_predict = model.predict(x_test)


# softmax 결과 → class 번호
y_predict = np.argmax(
    y_predict,
    axis=1
)

y_test = np.argmax(
    y_test,
    axis=1
)


# =========================================================
# accuracy_score
# =========================================================

acc_score = accuracy_score(
    y_test,
    y_predict
)

print('accuracy_score :', acc_score)


# =========================================================
# 걸린 시간
# =========================================================

print('걸린 시간 :', 
      round(end_time - start_time, 2), 
      'sec'
)

'''
cnn
Epoch 75: early stopping
------------------ 40_fashion_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.2131 - acc: 0.9280
loss :  0.2131367176771164
acc :  0.9279999732971191
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.928
걸린 시간 :  114.16 sec

Epoch 00058: early stopping
------------------ 51_fashion_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.2077 - acc: 0.9291
loss :  0.2077203392982483
acc :  0.929099977016449
accuracy_score :  0.9291
걸린 시간 :  154.22 sec

'''