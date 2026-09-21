'''
2026-09-21 (월)


'''

import numpy as np
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    # horizontal_flip = True,     # 수평 뒤집기  (좌우반전)(상하반전)
    # # 이미지의 데이터를 증폭, 변환할 수 있다.
    # vertical_flip = True,       # 수직 뒤집기  (상하반전)(좌우반전)
    # width_shift_range = 0.1,    # 평행이동
    # height_shift_range = 0.1,   # 수직이동
    # rotation_range = 5,         # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.2,           # 확대
    # shear_range = 0.7,          # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한 마디로 찌부)
    # fill_mode = 'nearest',
)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
)

# 경로지정
path_data = './_data/image/horse-human/'



xy = train_datagen.flow_from_directory(
    path_data,                      # 경로
    target_size = (300, 300),
    batch_size = 1027,
    class_mode = 'binary',           # 이진분류
    color_mode = 'rgb',         # 컬러
    shuffle = True,
)
# Found 1027 images belonging to 2 classes.

# print(xy[0][0].shape)      # (1027, 300, 300, 3)
# print(xy[0][1].shape)      # (1027,)

# print(xy[0][0])           # 여기부터 에러. 이유는 160장이다 !! 배치는 10개니까

# print(type(xy))            # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy[0]))         # <class 'tuple'>   # tuple, list와 비슷한데 저장이 안 된다.
# print(type(xy[0][0]))      # <class 'numpy.ndarray'>
# print(type(xy[0][1]))      # <class 'numpy.ndarray'>

# exit()

x = xy[0][0]
y = xy[0][1]


x_train, x_test, y_train, y_test =  train_test_split(
    x, y,
    random_state = 260921,
    train_size = 0.8,
    stratify = y,
)
# print(x_train.shape, y_train.shape)   # (821, 300, 300, 3) (821,)
# print(x_test.shape, y_test.shape)     # (206, 300, 300, 3) (206,)


np_path = './_data/save_horse-human_npy/'
np.save(np_path + 'keras46_01_x_train.npy', arr = x_train) # x_train
np.save(np_path + 'keras46_01_y_train.npy', arr = y_train) # y_train
np.save(np_path + 'keras46_01_x_test.npy', arr = x_test) # x_test
np.save(np_path + 'keras46_01_y_test.npy', arr = y_test) # y_test


# exit()

#2. 모델구성

#2. 모델구성 (유닛 강화 및 Dropout 완화)
model = Sequential()
# 첫 번째 블록 (300x300)
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

# 두 번째 블록 (150x150)
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

# 세 번째 블록 (75x75)
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))

# 네 번째 블록 (37x37) - 레이어 추가하여 해상도 정보 보존력 향상
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))

# 다섯 번째 블록 (18x18) - 특징을 좁은 해상도까지 충분히 축소
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.4))

# 글로벌 풀링 적용 (압축 전 해상도를 충분히 줄여 손실 감소)
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

path = './_save/keras46/01/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'horse-human_', date, "-",filename])

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
          batch_size = 8,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es, mcp],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras46_horse-human --------------------")
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

'''
Epoch 00055: early stopping
------------------ keras46_horse-human --------------------
26/26 [==============================] - 0s 15ms/step - loss: 0.0304 - acc: 0.9903
loss :  0.03043726645410061
acc :  0.9902912378311157
accuracy_score :  0.9902912621359223
걸린 시간 :  118.75 sec

'''