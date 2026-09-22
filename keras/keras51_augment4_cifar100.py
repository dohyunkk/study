'''
2026-09-22 (화)

!! 실습 만들어보기.
!! 목표 acc : 성능개선
'''

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# exit()
# plt.imshow(x_train[10], )
# plt.show()

# print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)

# print(np.min(x_train), np.max(x_train)) # 0 255
# print(np.min(x_test), np.max(x_test))   # 0 255
# print(np.min(y_train), np.max(y_train)) # 0 99
# print(np.min(y_test), np.max(y_test))   # 0 99

# exit()

##### ★ 여기부터 증폭 ★ #####
datagen = ImageDataGenerator(
    rescale = 1./255,           # 증강할 때 자동으로 255로 나누어 스케일링함
    horizontal_flip = True,     # 좌우반전
    width_shift_range = 0.1,    # 평행이동
    rotation_range = 30,        # 각도조절
    fill_mode = 'nearest',
)

augment_size = 40000

# 6만개 중에 4만개 랜덤 뽑기
randidx = np.random.randint(x_train.shape[0], size = augment_size) 

x_augmented = x_train[randidx].copy()        
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0], 32, 32, 3)

# ★ 중요: 변환되어 스케일링(1./255)까지 완료된 4만개 데이터를 받아옵니다.
x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle = False,
).next()[0]

#### 변환 및 데이터 병합 완료 ####
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# 원본 x_train은 스케일링이 안 되어 있으므로 255로 나누고, 변환 완료된 x_augmented와 단 한 번만 합칩니다.
x_train = np.concatenate((x_train / 255., x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)
x_test = x_test / 255.  # 테스트 데이터도 스케일링

print("최종 Train 구조:", x_train.shape, y_train.shape)   # 
print("라벨 분포:", np.unique(y_train, return_counts=True))

# 2. ★ 원-핫 인코딩 적용 ★
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

# 2. 모델구성 
model = Sequential()

# [블록 1] 입력 이미지: (32, 32, 3) -> (16, 16, 64)로 압축
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))

# [블록 2] 중수준 특성 추출 -> (8, 8, 128)로 압축
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.35))

# [블록 3] 고수준 특성 추출 -> (4, 4, 128)로 최종 압축
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.4))

# [분류기 진입]
# 4 * 4 * 128 = 2,048 노드 (과적합을 방지하는 가장 이상적인 크기)
model.add(GlobalAveragePooling2D()) 

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='softmax'))

model.summary()


# exit()

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

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 100,
          batch_size = 256,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ 51_cifar100_model.evaluate --------------------")
eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose = 1)
print('loss : ', eval_loss)
print('acc : ', eval_acc)

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis = 1)#.reshape(-1, 1)
y_test = np.argmax(y_test, axis = 1)#.reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')

'''
!! 목표 acc : 성능개선

Epoch 81: early stopping
------------------ 40_cifar100_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 1.8781 - acc: 0.5041
loss :  1.8780783414840698
acc :  0.5041000247001648
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.5041
걸린 시간 :  411.95 sec

Epoch 72: early stopping
------------------ 51_cifar100_model.evaluate --------------------
313/313 [==============================] - 1s 3ms/step - loss: 1.6761 - acc: 0.5863
loss :  1.6760517358779907
acc :  0.5863000154495239
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.5863
걸린 시간 :  747.85 sec
'''