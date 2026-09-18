'''
2026-09-18 (금)

for문은 이터레이터의 정보를 읽기 좋다.

from keras.preprocessing.image import ImageDataGenerator
'''

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,     # 수평 뒤집기  (좌우반전)(상하반전)
    # 이미지의 데이터를 증폭, 변환할 수 있다.
    vertical_flip = True,       # 수직 뒤집기  (상하반전)(좌우반전)
    width_shift_range = 0.1,    # 평행이동
    height_shift_range = 0.1,   # 수직이동
    rotation_range = 5,         # 각도조절(정해진 각도만큼 이미지 회전)
    zoom_range = 1.2,           # 확대
    shear_range = 0.7,          # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한 마디로 찌부)
    fill_mode = 'nearest',
)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
)

# 경로지정
path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'


xy_train = train_datagen.flow_from_directory(
    path_train,                      # 경로
    target_size = (100, 100),
    batch_size = 10,
    class_mode = 'binary',           # 이진분류
    color_mode = 'grayscale',         # 흑백
    shuffle = True,
)
# Found 160 images belonging to 2 classes.


xy_test = test_datagen.flow_from_directory(
    path_test,                        # 경로
    target_size = (100, 100),
    batch_size = 10,
    class_mode = 'binary',            # 이진분류
    color_mode = 'grayscale',         # 흑백
    shuffle = True,
)
# Found 120 images belonging to 2 classes.


# print(xy_train)                  # <keras.preprocessing.image.DirectoryIterator object at 0x000001F190997FA0>
#                                                                      ★Iterator : 프로그래밍에서 데이터 모음의 항목들에 
#                                                                                   순서대로 접근할 수 있게 도와주는 객체
# print(xy_train.next())           # 이터레이터의 첫번째를 보여줘
# print(xy_train.next())           # 이터레이터의 두번째를 보여줘

# print(xy_train[0])               # 이터레이터의 첫번째만 보여줘
# print(xy_train[1])               # 이터레이터의 두번째만 보여줘
# print(xy_train[2])               # 이터레이터의 세번째만 보여줘

# print(xy_train[0][0])            # 첫번째 배치의 x데이터.
# print(xy_train[0][1])            # 첫번째 배치의 y데이터.

# print(xy_train[0][0].shape)      # (10, 100, 100, 1)
# print(xy_train[0][1].shape)      # (10,)

# print(xy_train[16][0])           # 여기부터 에러. 이유는 160장이다 !! 배치는 10개니까

# print(type(xy_train))            # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))         # <class 'tuple'>   # tuple, list와 비슷한데 저장이 안 된다.
# print(type(xy_train[0][0]))      # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))      # <class 'numpy.ndarray'>

# 수치화 과정을 거쳐 데이터가 수치화 되었다.

