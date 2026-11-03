import warnings

import numpy as np
import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import all_estimators

warnings.filterwarnings("ignore")

# * 1. 데이터
x, y = load_iris(return_X_y=True)
# print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=333,
    train_size=0.8,
    stratify=y,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# * 2. 모델
all = all_estimators(type_filter="classifier")

# print("all models :", all)
# print("sk version :", sk.__version__)
# print("모델의 개수 :", len(all))
# sk version : 1.9.1
# 모델의 개수 : 44
for name, model in all:
    try:
        model = model()

        # 3. 훈련 + # 4. 평가
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print("====", name, "====")
        print("ACC :", scores, "\n평균 ACC :", round(np.mean(scores), 2))
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predict)
        print("cross_val_predict ACC :", acc)
    except:
        print(name, "모델 오류")
