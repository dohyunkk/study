'''
2026-09-18 (금)

45_3 카피

할 때 마다 데이터를 변환하는게 메모리, 시간 소모가 많다.
그래서 save 파일을 load 해보자.
'''
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import time

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

np_path = './_data/save_kaggle_cat_dog_npy/'


x_train = np.load(np_path + 'keras45_03_x_train.npy') 
y_train = np.load(np_path + 'keras45_03_y_train.npy') 
x_test = np.load(np_path + 'keras45_03_x_test.npy')
y_test = np.load(np_path + 'keras45_03_y_test.npy')

print(x_train.shape, y_train.shape) # (8005, 150, 150, 3) (8005,)
print(x_test.shape, y_test.shape)   # (2023, 150, 150, 3) (2023,)

# exit()

#2. 모델구성 (유닛 강화 및 Dropout 완화)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))

model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))

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

path = './_save/keras44/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'IDG3_CatDog_', date, "-",filename])

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
print("------------------ keras45_CatDog_load --------------------")
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


'''