'''
# 69_1 카피

# y2 = np.array(range(13001, 13101))   # 비트코인 가격
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

y1 = np.array(range(3001, 3101))
               # 화성의 화씨 온도
y2 = np.array(range(13001, 13101))   # 비트코인 가격

x_train1, x_test1, x_train2, x_test2, x_train3, x_test3, y_train1, y_test1, y_train2, y_test2 = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2,
    test_size=0.25,
    random_state=5959
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

# 2-5. 분기1 깊게도 할 수 있고
last_dense1 = Dense(10, name='ld1')(merge3)
last_dense2 = Dense(10, name='ld2')(last_dense1)
last_output1 = Dense(1, name='last1')(merge3)

# 2-6. 분기2 짧게도 할 수 있다
last_output2 = Dense(1, name='last2')(merge3)


model = Model(inputs=[input1, input21, input31], outputs=[last_output1, last_output2])

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
)

model.fit([x_train1, x_train2, x_train3], 
          [y_train1, y_train2], 
          epochs=1000, 
          batch_size=8,
          validation_split=0.1,
          callbacks=[es],
          )


#4. 평가, 예측
result = model.evaluate([x_test1, x_test2, x_test3], [y_test1, y_test2])
print('loss : ', result)

x1_pred = np.array([range(100, 106), range(400,406)]).T
x2_pred = np.array([range(200, 206), range(510, 516),
                    range(249, 255)]).T
x3_pred = np.array([range(100, 106), range(400, 406),
                    range(177, 183), range(133,139)]).T
y_predict = model.predict([x1_pred, x2_pred, x3_pred])
print(' 화성의 온도 예측값 :', y_predict[0])
print(' 비트의 가격 예측값 :', y_predict[1])
#예측값까지 완성해보시오.

'''
loss :  [0.08184686303138733, 0.0028267025481909513, 0.07902015745639801]
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step
 화성의 온도 예측값 : 
 [[3099.4778]
 [3100.7922]
 [3102.1707]
 [3103.5554]
 [3104.9573]
 [3106.3713]]
 비트의 가격 예측값 : 
 [[13093.128]
 [13096.632]
 [13100.614]
 [13104.649]
 [13108.804]
 [13113.067]]
 
 '''