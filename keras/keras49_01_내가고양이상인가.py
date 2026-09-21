'''
2026-09-21 (월)

개, 고양이 가중치를 가져와서 모델 완성
데이터는 개 고양이 npy데이터 사용
내 사진도 npy 불러와서 predict만 하면 되지요.

'''

import numpy as np
from PIL import Image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator # 👈 추가

# 1. 기존에 저장했던 최적의 가중치 모델 불러오기 (.keras)
model_path = './_save/keras45/IDG3_CatDog_0921_0927-0036-0.3599.keras' 
model = load_model(model_path)

# ======================================================================
# ✨ [수정 완료] ImageDataGenerator.flow를 활용한 .npy 데이터 전처리 및 리쉐이프
# ======================================================================
# 2. 카리나 npy 이미지 수치 배열 로드 (Image.open을 쓰면 에러가 나므로 np.load 사용)
my_face_path = './' 
img_array = np.load(my_face_path + '_data/save_img/keras48_yun.npy')

# 3. ImageDataGenerator는 무조건 배치 차원이 포함된 4차원 데이터만 받습니다.
# 만약 데이터가 (200, 200, 3) 3차원이라면 (1, 200, 200, 3) 4차원으로 축 확장
if len(img_array.shape) == 3:
    img_array = np.expand_dims(img_array, axis=0)

# 4 & 5. 제너레이터를 선언하고 flow()를 통해 1./255 정규화(rescale) 및 데이터 추출
# 기존에 훈련할 때 썼던 도화지 크기(200x200)에 맞춰진 npy이므로 그대로 flow에 태웁니다.
datagen = ImageDataGenerator(rescale=1./255)
predict_generator = datagen.flow(
    x=img_array,
    y=None,
    batch_size=1,
    shuffle=False
)

img_tensor = predict_generator[0] # 👈 변환 완료된 (1, 200, 200, 3) 형태의 데이터 추출
# ======================================================================

# 6. 모델 예측 실행
prediction = model.predict(img_tensor)
result_score = prediction[0][0] # 0과 1 사이의 확률값 추출

# 7. 이진 분류 결과 해석 (ImageDataGenerator의 알파벳 순서상 0: 고양이, 1: 개)
print("\n------------------ 내 얼굴 분석 결과 ------------------")
if result_score < 0.5:
    # 0에 가까울수록 고양이상
    cat_probability = (1 - result_score) * 100
    print(f"👉 당신은 치명적인 [ 고양이상 ] 입니다!")
    print(f"신뢰도: 고양이 확률 {round(cat_probability, 2)}%")
else:
    # 1에 가까울수록 강아지상
    dog_probability = result_score * 100
    print(f"👉 당신은 귀여운 [ 강아지상 ] 입니다!")
    print(f"신뢰도: 강아지 확률 {round(dog_probability, 2)}%")
print("--------------------------------------------------------")
