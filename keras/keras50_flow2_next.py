'''
2026-09-22 (화)

세이브 한 데이터를 증폭해보자.

50-1 카피
'''
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


##### ★ 여기부터 증폭 ★ #####
datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,     # 수평 뒤집기  (좌우반전)
    # vertical_flip = True,       # 수직 뒤집기  (상하반전)
    width_shift_range = 0.1,    # 평행이동
    # height_shift_range = 0.1,   # 수직이동
    rotation_range = 15,         # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.1,           # 확대
    # shear_range = 0.7,          # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한 마디로 찌부)
    fill_mode = 'nearest',
)


augment_size = 100

# print(x_train.shape)      # (60000, 28, 28)
# print(x_train[0].shape)   # (28, 28)

# aaa = np.tile(x_train[0], augment_size)
# print(aaa.shape)          # (28, 2800)


aaa = np.tile(x_train[0], augment_size).reshape(-1, 28, 28, 1)
print(aaa.shape)          # (100, 28, 28, 1)
### 단순 복붙 주의 ###

xy_data = datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),
    np.zeros(augment_size),                        # 0을 채워 넣어라, 해서 100개의 데이터에 0이 들어갔다.
    batch_size = augment_size,
    shuffle = False,
).next()


print(xy_data)
print(type(xy_data))       # <class 'tuple'>

# print(xy_data.shape)
print(len(xy_data))        # 2, 왜냐하면 x와 y 하나씩

print(xy_data[0].shape)    # (100, 28, 28, 1)
print(xy_data[1].shape)    # (100,)

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap = 'gray')
plt.show()









