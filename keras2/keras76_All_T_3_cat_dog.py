"""
75 카피

"""

import random
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import (
    VGG19,
    DenseNet201,
    EfficientNetB0,
    MobileNetV2,
    ResNet50,
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")


model_list = [
    VGG19,  # * 0
    ResNet50,  # * 1
    DenseNet201,  # * 2
    # InceptionV3,
    # InceptionResNetV2,
    MobileNetV2,  # * 3
    EfficientNetB0,  # * 4
]

# model = model_list[0]()  # 첫 번째 모델인 VGG19
# model = model_list[1]()  # 두 번째 모델인 ResNet50
# model = model_list[2]()  # 세 번째 모델인 DenseNet201
# model = model_list[3]()  # 네 번째 모델인 MobileNetV2
# model = model_list[4]()  # 네 번째 모델인 EfficientNetB0


random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

# 1. 데이터
np_path = "./_data/save_kaggle_cat_dog_npy/"

x_train = np.load(np_path + "keras45_03_x_train.npy")
y_train = np.load(np_path + "keras45_03_y_train.npy")
x_test = np.load(np_path + "keras45_03_x_test.npy")
y_test = np.load(np_path + "keras45_03_y_test.npy")

# print(x_train.shape, y_train.shape) # (8005, 150, 150, 3) (8005,)
# print(x_test.shape, y_test.shape)   # (2023, 150, 150, 3) (2023,)

# exit()

for model_class in model_list:
    model_i = model_class(
        weights="imagenet",
        include_top=False,
        input_shape=(200, 200, 3),
    )
    model_i.trainable = False

    model = Sequential()
    model.add(model_i)
    model.add(GlobalAveragePooling2D())  #! 얘가 풀리커넥티드임
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # * 스케일링
    # x_train = x_train/255.
    # x_test = x_test/255.

    # * 3. 컴파일, 훈련
    model.compile(
        loss="binary_crossentropy",
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
        batch_size=16,
        verbose=0,
        validation_split=0.2,
        callbacks=[es, rlr],
    )

    # * 4. 평가, 예측
    loss = model.evaluate(x_test, y_test, verbose=0)
    print("=======================================")
    print("모델명 :", model_i.name, "/", "loss :", loss[0], "/", "acc :", loss[1])


"""


"""
