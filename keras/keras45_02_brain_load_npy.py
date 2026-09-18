'''
2026-09-18 (금)

45_1 카피

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

from tensorflow.python.keras.callbacks import EarlyStopping



np_path = './_data/save_brain_npy/'


x_train = np.load(np_path + 'keras45_01_x_train.npy') 
y_train = np.load(np_path + 'keras45_01_y_train.npy') 
x_test = np.load(np_path + 'keras45_01_x_test.npy')
y_test = np.load(np_path + 'keras45_01_y_test.npy')

# print(x_train.shape, y_train.shape) # (160, 300, 300, 1) (160,)
# print(x_test.shape, y_test.shape)   # (120, 300, 300, 1) (120,)

# exit()

#2. 모델구성

#2. 모델구성 (유닛 강화 및 Dropout 완화)
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(300, 300, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
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

start_time = time.time()

model.fit(x_train, y_train,
          epochs = 500,
          batch_size = 16,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras45_brain_load  --------------------")
loss = model.evaluate(x_test, y_test, verbose = 1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_predict = model.predict(x_test)

y_predict = np.round(y_predict) 

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린 시간 : ', round(end_time - start_time, 2),'sec')



'''
# 실습 
# 목표 acc : 1.0

------------------ keras44_ImageDataGenerator2 --------------------
4/4 [==============================] - 1s 104ms/step - loss: 0.0509 - acc: 0.9917
loss :  0.050853628665208817
acc :  0.9916666746139526
accuracy_score :  0.9916666666666667
걸린 시간 :  212.13 sec


'''