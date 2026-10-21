import random
import time

import numpy as np
import tensorflow as tf
from keras.applications import VGG16
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#! 랜덤값 고정
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# plt.imshow(x_train[10], )
# plt.show()

# print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)

# print(np.min(x_train), np.max(x_train)) # 0 255
# print(np.min(x_test), np.max(x_test))   # 0 255
# print(np.min(y_train), np.max(y_train)) # 0 9
# print(np.min(y_test), np.max(y_test))   # 0 9

############### 스케일링 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# print(np.min(x_train), np.max(x_train))   # 0.0 1.0
# print(np.min(x_test), np.max(x_test))     # 0.0 1.0

################ reshape
# x_train = x_train.reshape(-1, 32 * 32 * 3)
# x_test = x_test.reshape(-1, 32 * 32 * 3)
# print(x_train.shape, x_test.shape)
# (50000, 32, 32, 3) (10000, 32, 32, 3)

ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)
# (50000, 10) (10000, 10)

# exit()

# 2. 모델
vgg16 = VGG16(
    weights="imagenet",
    # include_top=True,
    # input_shape=(224,224,3),
    include_top=False,
    input_shape=(32, 32, 3),
)

vgg16.trainable = False  #! 동결o
# vgg16.trainable=True    #! 동결x

# vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(10, activation="softmax"))

# model.summary()


# 3. 컴파일, 훈련

# learning_rate = 0.01
learning_rate = 0.001  # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=learning_rate),
    metrics=["acc"],
)

es = EarlyStopping(
    monitor="val_loss", mode="min", patience=30, restore_best_weights=True, verbose=1
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    epochs=5000,
    batch_size=128,
    verbose=1,
    validation_split=0.2,
    callbacks=[es],
)

end_time = time.time()


# 4. 평가, 예측
print("------------------ keras73_vgg16_cifar10 --------------------")
loss = model.evaluate(x_test, y_test, verbose=1)
print("loss : ", loss[0])
print("acc : ", loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)  # .reshape(-1, 1)
y_test = np.argmax(y_test, axis=1)  # .reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc_score)
print("걸린 시간 : ", round(end_time - start_time, 2), "sec")

"""
# 실습 # 
# 비교할거
#1. 이전에 본인이 한 최상의 훈련결과
#2. 가중치를 동결하고 훈련시켰을때,   #* vgg16.trainable=False  #! 동결o
#3. 가중치를 동결하지 않고 훈련시켰을때, #* vgg16.trainable=True   #! 동결x 
#4. 시간까지 비교.


cnn
Epoch 31: early stopping
------------------ 40_cifar10_model.evaluate --------------------
313/313 [==============================] - 1s 4ms/step - loss: 0.6181 - acc: 0.7989
loss :  0.6181313395500183
acc :  0.7989000082015991
313/313 [==============================] - 1s 3ms/step
accuracy_score :  0.7989
걸린 시간 :  469.75 sec


#* vgg16.trainable=True    #! 동결x
Epoch 38: early stopping
------------------ keras73_vgg16_cifar10 --------------------
313/313 [==============================] - 2s 4ms/step - loss: 0.7154 - acc: 0.7776
loss :  0.715376615524292
acc :  0.7775999903678894
313/313 [==============================] - 1s 4ms/step
accuracy_score :  0.7776
걸린 시간 :  424.75 sec

#* vgg16.trainable=False  #! 동결o
Epoch 49: early stopping
------------------ keras73_vgg16_cifar10 --------------------
313/313 [==============================] - 7s 20ms/step - loss: 1.2012 - acc: 0.5856
loss :  1.201150894165039
acc :  0.5856000185012817
313/313 [==============================] - 5s 15ms/step
accuracy_score :  0.5856
걸린 시간 :  331.49 sec

"""
