'''
2026-09-22 (화)
'''


import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# 1. 데이터

start1 = time.time()

np_path = './_save/keras/'
# np.save(np_path + 'keras46_03_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras46_03_y_train.npy', arr=xy_train[0][1])


x_train = np.load(np_path + 'keras46_03_x_train.npy')
y_train = np.load(np_path + 'keras46_03_y_train.npy')

print(x_train.shape)
print(y_train.shape)

# exit()

# ============================================================
# 여자 데이터만 추출

x_train_woman = x_train[np.where(y_train > 0.0)]
y_train_woman = y_train[np.where(y_train > 0.0)]

print(x_train_woman.shape, y_train_woman.shape)
print(np.unique(y_train_woman, return_counts=True))

# ============================================================
# train / test 분리

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, 
    test_size=0.1,
    random_state=2414
)

end1 = time.time()

print('데이터 걸린시간 : ', round(end1 -start1, 2), 'sec')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# ============================================================
# 데이터 증강 설정

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,     # 수평 뒤집기  (상하반전)
    vertical_flip = True,       # 수직 뒤집기  (좌우반전)
    width_shift_range = 0.1,    # 평행이동
    # height_shift_range = 0.1,   # 수직이동
    rotation_range = 5,         # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.2,           # 확대
    # shear_range = 0.7,          # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한 마디로 찌부)
    fill_mode = 'nearest',
)

# ============================================================
# 증강할 데이터 개수

augment_size = 8000

print(len(x_train_woman))
# x_train_woman = x_train_woman.reshape()
print(np.unique(y_train, return_counts=True))

# exit()

# ============================================================
# 여자 데이터에서 랜덤하게 8000개 선택

randidx = np.random.randint(
    x_train_woman.shape[0], 
    size = augment_size
)

print('randidx :', randidx)
print('randidx 개수 :', len(randidx))
print('최소 index :', np.min(randidx))
print('최대 index :', np.max(randidx))

# exit()
# print(x_train[0].shape)

# ============================================================
# 증강할 이미지와 라벨 생성

x_augmented = x_train_woman[randidx].copy()
y_augmented = y_train_woman[randidx].copy()

print(x_augmented.shape)
print(y_augmented.shape)

# exit()

# ============================================================
# 이미지 형태 확인 / reshape

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3)

print('reshape 후 x_augmented :', x_augmented.shape)

# ============================================================
# 이미지 증강

x_augmented = train_datagen.flow(
    x_augmented, 
    y_augmented,
    batch_size = augment_size,
    shuffle = False,
).next()[0]

print('증강 후 x_augmented :', x_augmented.shape)

# ============================================================
# 기존 데이터 reshape

x_train = x_train.reshape(x_train.reshape[0], 100, 100, 3)
x_test = x_test.reshape(x_test.reshape[0], 100, 100, 3)

print(x_train.shape, x_test.shape)

# ============================================================
# numpy에서 데이터 합치기

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)

print(np.unique(y_train, return_counts=True))


#2. 모델구성
model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(GlobalAveragePooling2D())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', 
              metrics=['acc'],
)

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

#★★★★★★★ mcp 세이브 파일명 만들기 시작 ★★★★★★★##
import datetime

date = datetime.datetime.now()
print(date)         # 2026-09-14 11:42:10.635267
print(type(date))   # <class 'datetime.datetime'>        # ★ calss
date = date.strftime("%m%d_%H%M")
print(date)         # 2026-09-14 11:48:27.764409
print(type(date))   # <class 'datetime.datetime'>

path = './_save/keras51/05/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'men-women', date, "-",filename])

#★★★★★★★ mcp 세이브 파일명 만들기 끝 ★★★★★★★##

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    filepath = filepath,
    verbose = 1,
)

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 300,
          batch_size = 32,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es, mcp],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras51_men-women_ --------------------")
loss = model.evaluate(x_test, y_test, 
                      verbose = 1,
                      batch_size = 8,
)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_predict = np.round(y_predict) 

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')