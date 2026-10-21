"""
75 카피

"""

import random
import warnings

import numpy as np
import tensorflow as tf
from keras.applications import VGG19, DenseNet201, EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")


model_list = [
    VGG19(include_top=False, input_shape=(32, 32, 3)),  # * 0
    ResNet50(include_top=False, input_shape=(32, 32, 3)),  # * 1
    DenseNet201(include_top=False, input_shape=(32, 32, 3)),  # * 2
    # InceptionV3(include_top=False, input_shape=(32, 32, 3)),
    # InceptionResNetV2(include_top=False, input_shape=(32, 32, 3)),
    MobileNetV2(include_top=False, input_shape=(32, 32, 3)),  # * 3
    EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),  # * 4
]

# model = model_list[0]()  # 첫 번째 모델인 VGG19
# model = model_list[1]()  # 두 번째 모델인 ResNet50
# model = model_list[2]()  # 세 번째 모델인 DenseNet201
# model = model_list[3]()  # 네 번째 모델인 MobileNetV2
# model = model_list[4]()  # 네 번째 모델인 EfficientNetB0


random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)


for model_i in model_list:
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())  #! 얘가 풀리커넥티드임
    model.add(Dense(32))
    model.add(Dense(16))
    # model.add(Dense(10, activation='softmax'))
    model.add(Dense(100, activation="softmax"))

    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # * 스케일링
    # x_train = x_train/255.
    # x_test = x_test/255.

    # * 3. 컴파일, 훈련
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["acc"],
    )

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # * 얼리스탑
    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20,
        verbose=0,
        restore_best_weights=True,
    )

    # * 리두스 엘알
    rlr = ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        patience=5,
        factor=0.5,
        verbose=0,
    )

    model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size=128,
        verbose=0,
        validation_split=0.2,
        callbacks=[es, rlr],
    )

    # * 4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("=======================================")
    print("모델명 :", model_i.name, "/", "loss :", loss[0], "/", "acc :", loss[1])
