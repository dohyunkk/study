"""
47 카피

# 동결

전이학습 : Transfer Learning

Feature Extraction
 - 사전 학습된 모델을 고정된 특징 추출기로 활용

Pre-trained Model
사전 학습 모델

사전학습 모델 기반의 파인 튜닝(Fine-tuning)

어쩌구 저쩌구... 이제 위에 용어들이 뭔 말인지 알것조?
"""

import random
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#! 랜덤값 고정
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


np_path = "./_data/save_horse-human_npy/"


x_train = np.load(np_path + "keras46_01_x_train.npy")
y_train = np.load(np_path + "keras46_01_y_train.npy")
x_test = np.load(np_path + "keras46_01_x_test.npy")
y_test = np.load(np_path + "keras46_01_y_test.npy")

# print(x_train.shape, y_train.shape) # (821, 300, 300, 3) (821,)
# print(x_test.shape, y_test.shape)   # (206, 300, 300, 3) (206,)

# exit()

# 2. 모델
vgg16 = VGG16(
    weights="imagenet",
    # include_top=True,
    # input_shape=(224,224,3),
    include_top=False,
    input_shape=(300, 300, 3),
)

vgg16.trainable = False  #! 동결o
# vgg16.trainable=True    #! 동결x

# vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(512))
model.add(Dense(1, activation="sigmoid"))

# model.summary()


# exit()

# 3. 컴파일, 훈련

# learning_rate = 0.01
learning_rate = 0.001  # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=learning_rate),
    metrics=["acc"],
)


es = EarlyStopping(
    monitor="val_loss",
    mode="auto",
    patience=10,
    verbose=1,
    restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor="val_loss",
    mode="auto",
    patience=5,
    verbose=1,
    factor=0.5,
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=16,
    verbose=1,
    validation_split=0.2,
    callbacks=[es, rlr],
)

end_time = time.time()


# 4. 평가, 예측
print("------------------ keras73_vgg16_man_horse --------------------")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_predict = model.predict(x_test)

# sigmoid 결과가 0.5 이상이면 1, 미만이면 0
y_predict = (y_predict >= 0.5).astype("int32").ravel()
y_test = y_test.astype("int32").ravel()

acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "sec")

"""
Epoch 00070: early stopping
------------------ keras47_horse-human_load --------------------
26/26 [==============================] - 0s 18ms/step - loss: 0.0076 - acc: 0.9951
loss :  0.007632194086909294
acc :  0.9951456189155579
accuracy_score :  0.9951456310679612
걸린 시간 :  149.9 sec

Epoch 15: early stopping
------------------ keras73_vgg16_man_horse --------------------
7/7 [==============================] - 5s 381ms/step - loss: 7.6632e-12 - acc: 1.0000
loss :  7.663179986405755e-12
acc :  1.0
7/7 [==============================] - 1s 186ms/step
accuracy_score :  1.0
걸린 시간 :  81.74 sec


Epoch 26: early stopping
------------------ keras73_vgg16_rps --------------------
52/52 [==============================] - 4s 78ms/step - loss: 7.5596e-09 - acc: 1.0000
loss :  7.559608938834117e-09
acc :  1.0
13/13 [==============================] - 7s 348ms/step
accuracy_score :  1.0
걸린 시간 :  290.87 sec

"""
