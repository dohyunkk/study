'''
머지(merge)가 뭐지? 합치기
모델 두개가 머지? 앙상블(ensemble)
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


y = np.array(range(3001, 3101))
               # 화성의 화씨 온도

x_train1, x_test1, x_train2, x_test2, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y,
    test_size=0.25,
    random_state=5959
)

# print(x_train1.shape, x_train2.shape, y_train1.shape, y_train2.shape)
# print(x_test1.shape, x_test2.shape, y_test1.shape, y_test2.shape)

# exit()
# 2-1. 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='han1')(input1)
dense2 = Dense(20, activation='relu', name='han2')(dense1)
dense3 = Dense(30, activation='relu', name='han3')(dense2)
output1 = Dense(5, activation='relu', name='han4')(dense3)
# model1 = Model(inputs=input1, outputs=output1)

# 2-2. 모델
input21 = Input(shape=(3,))
dense21 = Dense(50, name='han21')(input21)
dense22 = Dense(40, name='han22')(dense21)
dense23 = Dense(30, name='han23')(dense22)
dense24 = Dense(20, name='han24')(dense23)
output21 = Dense(3, name='han25')(dense24)
# model2 = Model(inputs=input21, outputs=output21)

# 2-3. 모델 합치기.(앙상블)
from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([output1, output21], name='mg1')
merge1 = Concatenate(name='mg1')([output1, output21])
merge2 = Dense(10, name='mg2')(merge1)
merge3 = Dense(5, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input21], outputs=last_output)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit([x_train1, x_train2], y_train, epochs=100, batch_size=8)


#4. 평가, 예측
result = model.evaluate([x_test1, x_test2], y_test)
print('loss : ', result)

x1_pred = np.array([range(100, 106), range(400,406)]).T
x2_pred = np.array([range(200, 206), range(510, 516),
                    range(249, 255)]).T

y_predict = model.predict([x1_pred, x2_pred])
print(' 화성의 온도 예측값 : ', y_predict)
#예측값까지 완성해보시오.

'''
loss :  0.24752317368984222
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 80ms/step
 화성의 온도 예측값 :  [[3101.542 ]
 [3102.827 ]
 [3104.1118]
 [3105.396 ]
 [3106.6807]
 [3107.965 ]]
 
 '''