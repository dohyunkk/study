'''
51_2 카피


'''

import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, GRU
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

##### ★ 여기부터 증폭 ★ #####
datagen = ImageDataGenerator(
    rescale = 1./255,           # 증강할 때 자동으로 255로 나누어 스케일링함
    horizontal_flip = False,     # 좌우반전
    height_shift_range=0.1,
    width_shift_range = 0.1,    # 평행이동
    rotation_range = 10,        # 각도조절
    fill_mode = 'nearest',
)

augment_size = 40000

# 6만개 중에 4만개 랜덤 뽑기
randidx = np.random.randint(x_train.shape[0], size = augment_size) 

x_augmented = x_train[randidx].copy()        
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)

# ★ 중요: 변환되어 스케일링(1./255)까지 완료된 4만개 데이터를 받아옵니다.
x_augmented = datagen.flow(
    x_augmented, y_augmented,
    batch_size = augment_size,
    shuffle = False,
).next()[0]

#### 변환 및 데이터 병합 완료 ####
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# 원본 x_train은 스케일링이 안 되어 있으므로 255로 나누고, 변환 완료된 x_augmented와 단 한 번만 합칩니다.
x_train = np.concatenate((x_train / 255., x_augmented), axis=0)
y_train = np.concatenate((y_train, y_augmented), axis=0)
x_test = x_test / 255.  # 테스트 데이터도 스케일링

print("최종 Train 구조:", x_train.shape, y_train.shape)   # (100000, 28, 28, 1) (100000,)
print("라벨 분포:", np.unique(y_train, return_counts=True))

# # 2. ★ 배웠던 방식대로 원-핫 인코딩 적용 ★
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)


# 2.모델구성
model = Sequential()
model.add(LSTM(64, input_shape=(28, 28)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))  

model.summary()

# exit()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer = 'adam', 
              metrics=['acc'],
              )

start_time = time.time()

es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=20, 
    restore_best_weights=True, 
    verbose=1
)

model.fit(x_train, y_train,
          epochs = 5000,
          batch_size = 128,
          verbose = 1,
          validation_split = 0.2,
          callbacks = [es ],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ 64_mnist_model.evaluate --------------------")
loss = model.evaluate(x_test, y_test, verbose = 1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis = 1)#.reshape(-1, 1)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')

'''
!!!!!!!!! 목표 acc : 0.995

Epoch 35: early stopping
------------------ 39_mnist_model.evaluate --------------------
313/313 [==============================] - 1s 2ms/step - loss: 0.0204 - acc: 0.9940
loss :  0.020386828109622
acc :  0.9940000176429749
313/313 [==============================] - 0s 1ms/step
accuracy_score :  0.994
걸린 시간 :  60.83 sec

!! 목표 acc : 성능개선

Epoch 00052: early stopping
------------------ 51_mnist_model.evaluate --------------------
313/313 [==============================] - 0s 2ms/step - loss: 0.0160 - acc: 0.9956
loss :  0.01604430191218853
acc :  0.9955999851226807
accuracy_score :  0.9956
걸린 시간 :  127.88 sec

Epoch 00038: early stopping
------------------ 64_mnist_model.evaluate --------------------
313/313 [==============================] - 4s 13ms/step - loss: 0.0392 - acc: 0.9893
loss :  0.03923676162958145
acc :  0.989300012588501
accuracy_score :  0.9893
걸린 시간 :  888.46 sec

Epoch 56: early stopping
------------------ 65_Reshape_model.evaluate --------------------
313/313 [==============================] - 1s 3ms/step - loss: 0.0345 - acc: 0.9879
loss :  0.034469638019800186
acc :  0.9879000186920166
313/313 [==============================] - 1s 2ms/step
accuracy_score :  0.9879
걸린 시간 :  195.03 sec

'''