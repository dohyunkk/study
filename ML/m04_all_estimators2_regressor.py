import warnings

import numpy as np
import sklearn as sk
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

warnings.filterwarnings("ignore")

# * 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    train_size=0.8,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# * 2. 모델
all = all_estimators(type_filter="regressor")

# print("all models :", all)
# print("sk version :", sk.__version__)
# print("모델의 개수 :", len(all))
# sk version : 1.9.1
# 모델의 개수 : 55

# exit()

for name, model in all:
    # print(name)
    # print(model)
    try:
        model = model()

        # 3. 훈련
        model.fit(x_train, y_train)

        # 4. 평가
        acc = model.score(x_test, y_test)
        print(name, "의 r2 :", acc)
    except Exception as e:
        print(name, "모델 오류", e)
