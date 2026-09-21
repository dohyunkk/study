'''
2026-09-21 (월)

세이브 한 데이터를 증폭해보자.
'''
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt


path = 'c:/study/_data/image/'

img = load_img(path + 'me.png', target_size = (100, 100))

# print(img) 
# # <PIL.Image.Image image mode=RGB size=200x200 at 0x1A9180DD660>
# print(type(img)) # <class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
# print(arr)
# print(arr.shape) # (200, 200, 3)
# print(type(arr)) # <class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis = 0)     # 차원 증가
# print(arr)
# print(arr.shape)  # (1, 200, 200, 3), predict에 넣을 수 있는 데이터가 되었다.

# 세이브
# np_path = './_data/save_img/'
# np.save(np_path + 'keras48_me.npy', arr=arr)

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

it = datagen.flow(arr, 
             batch_size = 1, 
)
# print(it)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x0000024E03F67F70>

# print(it.next())  # python 3.10까지,
# print(next(it))   # python 3.11이후
print(next(it).shape) # (1, 200, 200, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):
    # batch = it.next()  # 파이썬 구버전
    batch = next(it)     # 파이썬 신버전
    # print(batch.shape)
    batch = batch.reshape(100,100,3)

    ax[i].imshow(batch)
    # ax[i].axis('off')
plt.show()