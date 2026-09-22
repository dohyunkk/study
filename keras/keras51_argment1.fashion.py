'''
2026-09-22 (화)

데이터를 증폭해보자.

50-2 카피
'''
import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import fashion_mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

##### ★ 여기부터 증폭 ★ #####
datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,     # 수평 뒤집기  (좌우반전)
    # vertical_flip = True,       # 수직 뒤집기  (상하반전)
    width_shift_range = 0.1,    # 평행이동
    # height_shift_range = 0.1,   # 수직이동
    rotation_range = 30,         # 각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range = 1.1,           # 확대
    # shear_range = 0.7,          # 좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한 마디로 찌부)
    fill_mode = 'nearest',
)

augment_size = 40000 

# randidx = np.random.randint(x_train.shape[0], size = augment_size) 
# 6만개중에 4만개 랜덤뽑기 # randint 중복o
randidx = np.random.choice(x_train.shape[0], size = augment_size, replace = False) 
# 6만개중에 4만개 랜덤뽑기    # choice 중복 x
# print(randidx)
# print(randidx.shape)                       # (40000,) 벡터니까 먹혔다.
# print(len(randidx))                        # 40000 리스트는 len으로 확인해야 하는데, 벡터도 먹힌다.

print(np.min(randidx), np.max(randidx))      # 1 59998

x_augmented = x_train[randidx].copy()        # .copy() 나중에 수정하다 복사해간 글까지 수정되는 경우가 생기더라. 그러니 분리하자
y_augmented = y_train[randidx].copy()

# print(x_augmented.shape, y_augmented.shape)  # (40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 1)
    # 40000, 28, 28, 1)
# print(x_augmented.shape)                      # (40000, 28, 28, 1)

xy_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle = False,
).next()[0]

#### 변환 완료 ####
print(x_augmented.shape)                         # (40000, 28, 28, 1)

print(x_train.shape)                             # (60000, 28, 28)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)              # (100000, 28, 28, 1) (100000,)

print(np.unique(y_train, return_counts=True))

x_train = np.concatenate((x_train / 255., x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)
x_test = x_test / 255.

# onehot
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(y_train.shape)

# exit()

# [실습]
# 2.모델구성
model = Sequential()

# 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

# 
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

#
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))

#
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))

# 
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer = 'adam', 
              metrics=['acc'],
              )

start_time = time.time()

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=15, 
    restore_best_weights=True, 
    verbose=1
)

model.fit(x_train, y_train,
          epochs = 500,
          batch_size = 128,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es ],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ 51_fashion_model.evaluate --------------------")
loss = model.evaluate(x_test, y_test, verbose = 1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis = 1)#.reshape(-1, 1)
y_test = np.argmax(y_test, axis = 1)#.reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')

'''
cnn
Epoch 75: early stopping
------------------ 40_fashion_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.2131 - acc: 0.9280
loss :  0.2131367176771164
acc :  0.9279999732971191
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.928
걸린 시간 :  114.16 sec

'''