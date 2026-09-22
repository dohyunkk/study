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


# 1. 데이터 세팅 및 제너레이터 정의
train_datagen = ImageDataGenerator(
     # horizontal_flip = True,     # 수평 뒤집기  (좌우반전)(상하반전)
        # # 이미지의 데이터를 증폭, 변환할 수 있다.
        # vertical_flip = True,       # 수직 뒤집기  (상하반전)(좌우반전)
        # width_shift_range = 0.1,    # 평행이동
        # height_shift_range = 0.1,   # 수직이동
        # rotation_range = 5,         # 각도조절(정해진 각도만큼 이미지 회전)
        # zoom_range = 1.2,           # 확대
        # shear_range = 0.7,          # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한 마디로 찌부)
        # fill_mode = 'nearest',
    rescale = 1./255,
)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
)

# 경로지정 (RPS 가위바위보 데이터셋)
path_train = './_data/image/rps/'

# 전체 이미지 2049장을 한 번에 담기 위해 batch_size를 2049로 지정
xy_data = train_datagen.flow_from_directory(
    path_train,                      
    target_size = (300, 300),
    batch_size = 2049,
    class_mode = 'categorical',           # ★ 다중분류 (One-Hot Encoding 형태로 y 생성됨)
    color_mode = 'rgb',         
    shuffle = True,
)
# Found 2049 images belonging to 3 classes.

# ★ xy_data에서 순수한 x(이미지)와 y(레이블) 추출
x = xy_data[0][0]
y = xy_data[0][1]

print("원본 변환 크기:")
print("x.shape:", x.shape)      # (2048, 300, 300, 3)
print("y.shape:", y.shape)      # (2048, 3) -> 3개의 클래스이므로 원핫인코딩 형태

# ★ [수정] 다중 분류 변수 분할 (stratify=y 적용으로 가위/바위/보 비율 완벽 분할)
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state = 260921,
    train_size = 0.8,
    stratify = y,
)

print("분할 후 크기:")
print("x_train.shape, y_train.shape :", x_train.shape, y_train.shape)   # (1638, 300, 300, 3) (1638, 3)
print("x_test.shape, y_test.shape   :", x_test.shape, y_test.shape)     # (410, 300, 300, 3) (410, 3)

# ★ [수정] 안전한 npy 저장 경로 및 인덱싱 제거 완료
np_path = './_data/save_rps_npy/'

np.save(np_path + 'keras46_02_x_train.npy', arr = x_train) 
np.save(np_path + 'keras46_02_y_train.npy', arr = y_train) 
np.save(np_path + 'keras46_02_x_test.npy', arr = x_test) 
np.save(np_path + 'keras46_02_y_test.npy', arr = y_test) 

print("RPS 데이터셋 .npy 저장 완료!")

# exit()


#2. 모델구성 (유닛 강화 및 Dropout 완화)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))

model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', 
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

path = './_save/keras46/02/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'rps_', date, "-",filename])

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
print("------------------ keras46_rps_ImageDataGenerator3 --------------------")
loss = model.evaluate(x_test, y_test, 
                      verbose = 1,
                      batch_size = 8,
)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_test_arg = np.argmax(y_test, axis=1)         # 원래 정답의 최대값 인덱스 추출 (1차원화)
y_predict_arg = np.argmax(y_predict, axis=1)   # 예측된 확률의 최대값 인덱스 추출 (1차원화)

acc_score = accuracy_score(y_test_arg, y_predict_arg)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')

'''


'''