'''
2026-09-21 (월)

'''
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array




# ----------------------------------------------------
# 1. 
# ----------------------------------------------------
model_path = './_save/keras47/03/men-women0921_1702-0029-0.1487.keras' 

# 테스트할 내 사진 경로
my_pic_path = './_data/image/me.png' 


# ----------------------------------------------------
# 2. 모델 불러오기 및 내 사진 이미지 전처리
# ----------------------------------------------------
# 저장된 모델 탑재
model = load_model(model_path)

# 내 사진 로드 및 학습 때와 동일한 크기(150x150)로 변환
img = load_img(my_pic_path, target_size=(100, 100))

# 이미지를 numpy 배열로 변환 (150, 150, 3)
img_array = img_to_array(img)

# 학습 때 rescale = 1./255 를 했으므로 예측할 때도 똑같이 255로 나누기
img_array = img_array / 255.0

# 모델 예측을 위해 차원 확장 (1, 150, 150, 3) -> 배치 사이즈 차원 추가
img_array = np.expand_dims(img_array, axis=0)


# ----------------------------------------------------
# 3. 예측 및 결과 출력
# ----------------------------------------------------
print("\n[정보] 모델이 사진을 분석 중입니다...")
prediction = model.predict(img_array)

# ImageDataGenerator의 클래스 인덱스 기본값: (men: 0, women: 1)
if prediction < 0.5:
    gender = "남성 (Men)"
    confidence = (1 - prediction[0][0]) * 100
else:
    gender = "여성 (Women)"
    confidence = prediction[0][0] * 100

print("=" * 50)
print(f" 최종 예측 결과 : {gender} 입니다.")
print(f" 모델의 확신도   : {confidence:.2f}%")
print("=" * 50)

'''
==================================================
 최종 예측 결과 : 남성 (Men) 입니다.
 모델의 확신도   : 99.50%
==================================================
'''