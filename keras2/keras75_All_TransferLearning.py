import warnings

from keras.applications import (
    VGG16,
    VGG19,
    DenseNet121,
    DenseNet169,
    DenseNet201,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    InceptionResNetV2,
    InceptionV3,
    MobileNet,
    MobileNetV2,
    MobileNetV3Large,
    MobileNetV3Small,
    NASNetLarge,
    NASNetMobile,
    ResNet50,
    ResNet50V2,
    ResNet101,
    ResNet101V2,
    ResNet152,
    ResNet152V2,
    Xception,
)

warnings.filterwarnings("ignore")


model_list = [
    VGG16(),
    VGG19(),
    ResNet50(),
    ResNet50V2(),
    ResNet101(),
    ResNet101V2(),
    ResNet152(),
    ResNet152V2(),
    DenseNet201(),
    DenseNet169(),
    DenseNet121(),
    InceptionV3(),
    InceptionResNetV2(),
    MobileNet(),
    MobileNetV2(),
    MobileNetV3Small(),
    MobileNetV3Large(),
    NASNetMobile(),
    NASNetLarge(),
    EfficientNetB0(),
    EfficientNetB1(),
    EfficientNetB2(),
    Xception(),
]

# model = model_list[0]()  # 첫 번째 모델인 VGG16 생성
# model = model_list[1]()  # 두 번째 모델인 VGG19 생성
# model = model_list[2]()  # 세 번째 모델인 ResNet50 생성
model = Xception()

model.trainable = True

model.summary()

print("=========================")
print("모델명 :", model.name)
print("전체 가중치 갯수 :", len(model.weights))
print("전체 훈련가능 가중치 갯수 :", len(model.trainable_weights))

for model in model_list:
    model.trainable = False
    print("=========================")
    print("모델명 :", model.name)
    print("전체 가중치 갯수 :", len(model.weights))
    print("전체 훈련가능 가중치 갯수 :", len(model.trainable_weights))

#     =========================
# * 모델명 : xception
# 전체 가중치 갯수 : 236
# 전체 훈련가능 가중치 갯수 : 156
# =========================
# * 모델명 : vgg16
# 전체 가중치 갯수 : 32
# 전체 훈련가능 가중치 갯수 : 32
# =========================
# * 모델명 : vgg19
# 전체 가중치 갯수 : 38
# 전체 훈련가능 가중치 갯수 : 38
# =========================
# * 모델명 : resnet50
# 전체 가중치 갯수 : 320
# 전체 훈련가능 가중치 갯수 : 214
# =========================
# * 모델명 : resnet50v2
# 전체 가중치 갯수 : 272
# 전체 훈련가능 가중치 갯수 : 174
# =========================
# * 모델명 : resnet101
# 전체 가중치 갯수 : 626
# 전체 훈련가능 가중치 갯수 : 418
# =========================
# * 모델명 : resnet101v2
# 전체 가중치 갯수 : 544
# 전체 훈련가능 가중치 갯수 : 344
# =========================
# * 모델명 : resnet152
# 전체 가중치 갯수 : 932
# 전체 훈련가능 가중치 갯수 : 622
# =========================
# * 모델명 : resnet152v2
# 전체 가중치 갯수 : 816
# 전체 훈련가능 가중치 갯수 : 514
# =========================
# * 모델명 : densenet201
# 전체 가중치 갯수 : 1006
# 전체 훈련가능 가중치 갯수 : 604
# =========================
# * 모델명 : densenet169
# 전체 가중치 갯수 : 846
# 전체 훈련가능 가중치 갯수 : 508
# =========================
# * 모델명 : densenet121
# 전체 가중치 갯수 : 606
# 전체 훈련가능 가중치 갯수 : 364
# =========================
# * 모델명 : inception_v3
# 전체 가중치 갯수 : 378
# 전체 훈련가능 가중치 갯수 : 190
# =========================
# * 모델명 : inception_resnet_v2
# 전체 가중치 갯수 : 898
# 전체 훈련가능 가중치 갯수 : 490
# =========================
# * 모델명 : mobilenet_1.00_224
# 전체 가중치 갯수 : 137
# 전체 훈련가능 가중치 갯수 : 83
# =========================
# * 모델명 : mobilenetv2_1.00_224
# 전체 가중치 갯수 : 262
# 전체 훈련가능 가중치 갯수 : 158
# =========================
# * 모델명 : MobilenetV3small
# 전체 가중치 갯수 : 210
# 전체 훈련가능 가중치 갯수 : 142
# =========================
# * 모델명 : MobilenetV3large
# 전체 가중치 갯수 : 266
# 전체 훈련가능 가중치 갯수 : 174
# =========================
# * 모델명 : NASNet
# 전체 가중치 갯수 : 1126
# 전체 훈련가능 가중치 갯수 : 742
# =========================
# * 모델명 : NASNet
# 전체 가중치 갯수 : 1546
# 전체 훈련가능 가중치 갯수 : 1018
# =========================
# * 모델명 : efficientnetb0
# 전체 가중치 갯수 : 314
# 전체 훈련가능 가중치 갯수 : 213
# =========================
# * 모델명 : efficientnetb1
# 전체 가중치 갯수 : 442
# 전체 훈련가능 가중치 갯수 : 301
# =========================
# * 모델명 : efficientnetb2
# 전체 가중치 갯수 : 442
# 전체 훈련가능 가중치 갯수 : 301
# =========================
# * 모델명 : xception
# 전체 가중치 갯수 : 236
# 전체 훈련가능 가중치 갯수 : 156
