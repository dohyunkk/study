'''
# 69_1 카피

'''

import numpy as np
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T     # (100, 2)
                       # 삼성 종가     하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), 
                              # 원유가           환율
                        range(150, 250)]).transpose()      # (100, 3)
                              # 금시세
x3_datasets = np.array([range(100), range(301, 401),
                        range(77,177), range(33, 133)]).T  # (100, 4)

y = np.array(range(3001, 3101))
               # 화성의 화씨 온도

x_train1, x_test1, x_train2, x_test2, x_train3, x_test3, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y,
    test_size=0.25,
    random_state=5959,
)



# print(x_train1.shape, x_train2.shape, y_train1.shape, y_train2.shape)
# print(x_test1.shape, x_test2.shape, y_test1.shape, y_test2.shape)

# exit()
# 2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(32, activation='relu', name='han1')(input1)
dense2 = Dense(64, activation='relu', name='han2')(dense1)
dense3 = Dense(128, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

# 2-2. 모델
input21 = Input(shape=(3,))
dense21 = Dense(32, name='han21')(input21)
dense22 = Dense(64, name='han22')(dense21)
dense23 = Dense(128, name='han23')(dense22)
dense24 = Dense(32, name='han24')(dense23)
output21 = Dense(3, name='han25')(dense24)
# model2 = Model(inputs=input21, outputs=output21)

# 2-3. 모델
input31 = Input(shape=(4,))
dense31 = Dense(32, activation='relu', name='han31')(input31)
dense32 = Dense(64, activation='relu', name='han32')(dense31)
dense33 = Dense(128, activation='relu', name='han33')(dense32)
dense34 = Dense(32, activation='relu', name='han34')(dense33)
output31 = Dense(3, name='han35')(dense34)
# model2 = Model(inputs=input21, outputs=output21)

# 2-4. 모델 합치기.(앙상블)
from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21, output31])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input21, input31], outputs=last_output)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
)

model.fit([x_train1, x_train2, x_train3], y_train, 
          epochs=1000, 
          batch_size=8,
          validation_split=0.1,
          callbacks=[es],
          )


#4. 평가, 예측
result = model.evaluate([x_test1, x_test2, x_test3], y_test)
print('loss : ', result)

x1_pred = np.array([range(100, 106), range(400,406)]).T
x2_pred = np.array([range(200, 206), range(510, 516),
                    range(249, 255)]).T
x3_pred = np.array([range(100, 106), range(400, 406),
                    range(177, 183), range(133,139)]).T
y_predict = model.predict([x1_pred, x2_pred, x3_pred])
print(' 화성의 온도 예측값 : ', y_predict)
#예측값까지 완성해보시오.

'''
loss :  0.5049575567245483
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 110ms/step
 화성의 온도 예측값 :  [[3102.7375]
 [3105.5427]
 [3108.3472]
 [3111.1526]
 [3113.9575]
 [3116.7507]]
 
 '''