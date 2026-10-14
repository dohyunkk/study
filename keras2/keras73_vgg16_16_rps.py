'''
# 동결
47 카피
'''
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import time

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import random

#! 랜덤값 고정 
random.seed(333)
np.random.seed(333)
tf.random.set_seed(333)

from tensorflow.keras.applications import VGG16
np_path = './_data/save_rps_npy/'


x_train = np.load(np_path + 'keras46_02_x_train.npy') 
y_train = np.load(np_path + 'keras46_02_y_train.npy') 
x_test = np.load(np_path + 'keras46_02_x_test.npy')
y_test = np.load(np_path + 'keras46_02_y_test.npy')

print(x_train.shape, y_train.shape) # (1638, 300, 300, 3) (1638, 3)
print(x_test.shape, y_test.shape)   # (410, 300, 300, 3) (410, 3)

# exit()

#2. 모델구성
vgg16 = VGG16(
weights='imagenet',
# include_top=True,
# input_shape=(224,224,3),
include_top=False,
input_shape=(300,300,3),
)

vgg16.trainable=False  #! 동결o
# vgg16.trainable=True    #! 동결x

# vgg16.summary()
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(512))
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



start_time = time.time()

model.fit(x_train, y_train,
          epochs = 300,
          batch_size = 16,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras73_vgg16_rps --------------------")
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
Epoch 00045: early stopping
------------------ keras47_rps_ImageDataGenerator3 --------------------
52/52 [==============================] - 1s 13ms/step - loss: 4.1146e-06 - acc: 1.0000
loss :  4.1145735849568155e-06
acc :  1.0
accuracy_score :  1.0
걸린 시간 :  194.09 sec

Epoch 00062: early stopping
------------------ keras47_rps_ImageDataGenerator3 --------------------
52/52 [==============================] - 1s 13ms/step - loss: 3.7478e-07 - acc: 1.0000
loss :  3.747774144358118e-07
acc :  1.0
accuracy_score :  1.0
걸린 시간 :  263.51 sec


Epoch 26: early stopping
------------------ keras73_vgg16_rps --------------------
52/52 [==============================] - 4s 71ms/step - loss: 7.5596e-09 - acc: 1.0000
loss :  7.559608938834117e-09
acc :  1.0
13/13 [==============================] - 6s 332ms/step
accuracy_score :  1.0
걸린 시간 :  273.46 sec
'''