import warnings

import numpy as np
import sklearn as sk
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    KFold,
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

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# * 2. 모델
all = all_estimators(type_filter="regressor")

# print("all models :", all)
# print("sk version :", sk.__version__)
# print("모델의 개수 :", len(all))
# sk version : 1.9.1
# 모델의 개수 : 55
for name, model in all:
    try:
        model = model()

        # 3. 훈련 + # 4. 평가
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print("====", name, "====")
        print("r2 :", scores, "\n평균 r2 :", round(np.mean(scores), 2))
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        r2 = r2_score(y_test, y_predict)
        print("cross_val_predict r2 :", r2)
    except:
        print(name, "모델 오류")
