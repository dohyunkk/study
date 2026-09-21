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

np_path = './_data/save_horse-human_npy/'


x_train = np.load(np_path + 'keras46_01_x_train.npy') 
y_train = np.load(np_path + 'keras46_01_y_train.npy') 
x_test = np.load(np_path + 'keras46_01_x_test.npy')
y_test = np.load(np_path + 'keras46_01_y_test.npy')

# print(x_train.shape, y_train.shape) # (821, 300, 300, 3) (821,)
# print(x_test.shape, y_test.shape)   # (206, 300, 300, 3) (206,)

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

path = './_save/keras47/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'horse', date, "-",filename])

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
print("------------------ keras47_horse-human_load --------------------")
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

Epoch 00070: early stopping
------------------ keras47_horse-human_load --------------------
26/26 [==============================] - 0s 18ms/step - loss: 0.0076 - acc: 0.9951
loss :  0.007632194086909294
acc :  0.9951456189155579
accuracy_score :  0.9951456310679612
걸린 시간 :  149.9 sec

Epoch 00073: early stopping
------------------ keras47_horse-human_load --------------------
26/26 [==============================] - 1s 19ms/step - loss: 0.0159 - acc: 0.9903
loss :  0.015862850472331047
acc :  0.9902912378311157
accuracy_score :  0.9902912621359223
걸린 시간 :  158.68 sec
'''