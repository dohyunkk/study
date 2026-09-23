'''
2026-09-22 (화)

52 카피

한도에 닿으면 러닝레이트를 절반으로 낮춰서 이어가겠다.
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1, 
    factor=0.5,
)
'''

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import time

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

np_path = './_data/save_men-women_npy/'


x = np.load(np_path + 'x.npy') 
y = np.load(np_path + 'y.npy') 


# print(x_train.shape, y_train.shape) # (821, 300, 300, 3) (821,)
# print(x_test.shape, y_test.shape)   # (206, 300, 300, 3) (206,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    random_state=2414,
)

# exit()

#2. 모델구성 (유닛 강화 및 Dropout 완화)
model = Sequential()
# 첫 번째 블록 (300x300)
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(100, 100, 3)))
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
from tensorflow.keras.optimizers import Adam

# learning_rate = 0.01
learning_rate = 0.001   # 디폴트값
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='binary_crossentropy', 
              optimizer = Adam(learning_rate=learning_rate), 
              metrics=['acc'],
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(         
    monitor = 'val_loss',        
    mode = 'auto',                 
    patience = 40,                
    verbose = 1,
    restore_best_weights = True,  
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode='auto',
    patience=20,
    verbose=1, 
    factor=0.5,
)

start_time = time.time()


model.fit(x_train, y_train,
          epochs = 300,
          batch_size = 32,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[es, rlr]#, mcp],
          )

end_time = time.time()


#4. 평가, 예측
print("------------------ keras53_reducel_men-women --------------------")
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
------------------ keras47_men-women_load --------------------
849/849 [==============================] - 2s 2ms/step - loss: 0.1539 - acc: 0.9417
loss :  0.153945192694664
acc :  0.9416961073875427
accuracy_score :  0.941696113074205
걸린 시간 :  281.34 sec
 
Epoch 96: early stopping
------------------ keras52_learning_rate_men-women --------------------
849/849 [==============================] - 2s 2ms/step - loss: 0.3581 - acc: 0.8357
loss :  0.35808923840522766
acc :  0.8356890678405762
213/213 [==============================] - 1s 4ms/step
accuracy_score :  0.835394581861013
걸린 시간 :  1199.25 sec
'''