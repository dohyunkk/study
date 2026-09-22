'''
2026-09-22 (화)

!! 실습 만들어보기.
!! 목표 acc : 성능개선
'''

import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

from sklearn.metrics import accuracy_score


# ============================================================
# 1. 데이터
# ============================================================

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ============================================================
# 2. 데이터 증강
# ============================================================

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=False,
    height_shift_range=0.1,
    width_shift_range=0.1,
    rotation_range=10,
    fill_mode='nearest'
)

augment_size = 40000


# 60,000개 중 랜덤하게 40,000개 선택
randidx = np.random.randint(
    x_train.shape[0],
    size=augment_size
)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()


# CNN 입력 형태로 변경
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    28,
    28,
    1
)


# 데이터 증강
generator = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False
)

x_augmented = next(generator)[0]


# ============================================================
# 3. 원본 데이터 형태 변경 및 스케일링
# ============================================================

x_train = x_train.reshape(60000, 28, 28, 1)

x_test = x_test.reshape(10000, 28, 28, 1)


# 원본 데이터 스케일링
x_train = x_train / 255.0
x_test = x_test / 255.0


# ============================================================
# 4. 원본 + 증강 데이터 합치기
# ============================================================

x_train = np.concatenate(
    (x_train, x_augmented),
    axis=0
)

y_train = np.concatenate(
    (y_train, y_augmented),
    axis=0
)


print()
print("==========================================")
print("최종 Train 구조")
print("==========================================")
print("x_train :", x_train.shape)
print("y_train :", y_train.shape)

print()
print("라벨 분포")
print(np.unique(y_train, return_counts=True))


# ============================================================
# 5. 원-핫 인코딩
# ============================================================

y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)


# ============================================================
# 6. 모델 구성
# ============================================================

model = Sequential()

# 블록 1
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

# 블록 2
model.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

# Fully Connected
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.summary()


# ============================================================
# 7. 컴파일
# ============================================================

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)


# ============================================================
# 8. EarlyStopping
# ============================================================

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 9. 학습
# ============================================================

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


# ============================================================
# 10. 평가
# ============================================================

print()
print("==========================================")
print("MNIST MODEL EVALUATE")
print("==========================================")


loss = model.evaluate(
    x_test,
    y_test,
    verbose=1
)


print()
print("loss :", loss[0])
print("acc  :", loss[1])


# ============================================================
# 11. 예측
# ============================================================

y_predict = model.predict(
    x_test
)


# 확률값 → 실제 숫자
y_predict = np.argmax(
    y_predict,
    axis=1
)

y_test = np.argmax(
    y_test,
    axis=1
)


# ============================================================
# 12. Accuracy Score
# ============================================================

acc_score = accuracy_score(
    y_test,
    y_predict
)


print()
print("accuracy_score :", acc_score)


# ============================================================
# 13. 학습 시간
# ============================================================

print("걸린 시간 :", round(end_time - start_time, 2), "sec")

'''
Epoch 00038: early stopping
------------------ 51_fashion_model.evaluate --------------------
313/313 [==============================] - 0s 2ms/step - loss: 0.2118 - acc: 0.9263
loss : 0.2117992639541626
acc  : 0.9262999892234802
accuracy_score : 0.9263
걸린 시간 : 101.63 sec

'''