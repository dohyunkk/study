'''
2026-09-21 (월)


'''
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import time

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

np_path = './_data/save_rps_npy/'


x_train = np.load(np_path + 'keras46_02_x_train.npy') 
y_train = np.load(np_path + 'keras46_02_y_train.npy') 
x_test = np.load(np_path + 'keras46_02_x_test.npy')
y_test = np.load(np_path + 'keras46_02_y_test.npy')

print(x_train.shape, y_train.shape) # (1638, 300, 300, 3) (1638, 3)
print(x_test.shape, y_test.shape)   # (410, 300, 300, 3) (410, 3)

# exit()

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
print(date)         
print(type(date))   # <class 'datetime.datetime'>        # ★ calss
date = date.strftime("%m%d_%H%M")
print(date)         
print(type(date))   # <class 'datetime.datetime'>

path = './_save/keras47/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'rps', date, "-",filename])

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
print("------------------ keras47_rps_ImageDataGenerator3 --------------------")
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

'''