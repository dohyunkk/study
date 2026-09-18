'''
2026-09-18 (금)

for문은 이터레이터의 정보를 읽기 좋다.

from keras.preprocessing.image import ImageDataGenerator

44_1 카피
'''
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import time

from tensorflow.python.keras.callbacks import EarlyStopping

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
path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'


xy_train = train_datagen.flow_from_directory(
    path_train,                      # 경로
    target_size = (150, 150),
    batch_size = 160,
    class_mode = 'binary',           # 이진분류
    color_mode = 'grayscale',         # 흑백
    shuffle = True,
)
# Found 160 images belonging to 2 classes.


xy_test = test_datagen.flow_from_directory(
    path_test,                        # 경로
    target_size = (150, 150),
    batch_size = 120,
    class_mode = 'binary',            # 이진분류
    color_mode = 'grayscale',         # 흑백
    shuffle = True,
)
# Found 120 images belonging to 2 classes.

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)   # (160, 150, 150, 1) (160,)
print(x_test.shape, y_test.shape)     # (120, 150, 150, 1) (120,)


#2. 모델구성

#2. 모델구성 (유닛 강화 및 Dropout 완화)
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 1)))
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
print("------------------ keras44_brain_ImageDataGenerator2 --------------------")
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