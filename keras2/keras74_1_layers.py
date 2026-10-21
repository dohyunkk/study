from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3),
)

vgg16.trainable = False

model = Sequential()
model.add(vgg16)  # vgg16이 13개의 레이어로 구성
model.add(Flatten())
model.add(Dense(100))  # vgg16 이후에 층 두개.
model.add(Dense(10, activation="softmax"))

print(len(model.weights))  # 30 : 13(vgg16) * 2 (w, b) + 2(Dense) * 2
print(len(model.trainable_weights))  # 4

print(model.weights)
