'''
2026-09-21 (월)


'''
import numpy as np
import time
import datetime

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# ----------------------------------------------------
# 1. 데이터 로드 및 미니배치 스트리밍 설정 (제너레이터 자체 분할)
# ----------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2  # 8:2 비율로 내부 분할 선언
)

path_data = './_data/image/men-women/'


train_generator = train_datagen.flow_from_directory(
    path_data,                      
    target_size = (150, 150),       
    batch_size = 1000,                
    class_mode = 'binary',           
    color_mode = 'rgb',         
    shuffle = True,
    subset = 'training'             
)


test_generator = train_datagen.flow_from_directory(
    path_data,                      
    target_size = (150, 150),       
    batch_size = 1000,                
    class_mode = 'binary',           
    color_mode = 'rgb',         
    shuffle = False,                
    subset = 'validation'           
)


# ----------------------------------------------------
# 2. 정석적인 CNN 모델 구성 
# ----------------------------------------------------
model = Sequential()

# 첫 번째 블록 (150x150 -> 75x75)
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))

# 두 번째 블록 (75x75 -> 37x37)
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.2))


model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))


model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.3))


model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D()) 
model.add(Dropout(0.4))


model.add(GlobalAveragePooling2D()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) 

model.summary()


# ----------------------------------------------------
# 3. 컴파일 및 훈련
# ----------------------------------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# 조기 종료 설정
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

date = date.strftime("%m%d_%H%M")

path = './_save/keras46/03/'
filename = '{epoch:04d}-{val_loss:.4f}.keras'
filepath = "".join([path, 'men-women_', date, "-",filename])

#★★★★★★★ mcp 세이브 파일명 만들기 끝 ★★★★★★★##

# 가중치 저장 파일 체크포인트 설정
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,            # val_loss가 기존보다 떨어졌을 때만(최고 성능일 때만) 갱신 저장
    filepath=filepath,              # 위에서 완성한 경로 대입
    verbose=1,
)

start_time = time.time()

# 미니배치 학습 시작 (주머니를 넣어 통째로 학습)
model.fit(
    train_generator,
    epochs=300,
    batch_size = 128,
    verbose=2,
    validation_data=test_generator,
    callbacks=[es, mcp],            # 콜백에 es와 mcp 모두 탑재!
)

end_time = time.time()


# ----------------------------------------------------
# 4. 평가 및 검증
# ----------------------------------------------------
print("------------------ keras46_men-women 검증 결과 --------------------")
loss = model.evaluate(test_generator, verbose=1)
print('최종 검증 loss (오차) :', loss[0])
print('최종 검증 acc (정확도) :', loss[1])

# 정확도 스코어 계산
y_predict = model.predict(test_generator)
y_predict = np.round(y_predict) 
y_test_actual = test_generator.classes  # 제너레이터의 실제 정답 리스트 추출

acc_score = accuracy_score(y_test_actual, y_predict)
print('최종 accuracy_score 지표 :', acc_score)
print('전체 학습 소요 시간 :', round(end_time - start_time, 2), 'sec')


'''


'''