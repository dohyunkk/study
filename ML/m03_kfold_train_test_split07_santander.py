import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# * 1. 데이터
# path = './_data/kaggle_santander/'              # 상대경로
path = "c:/study/_data/kaggle_santander/"  # 절대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv", index_col=0)

x = train_csv.drop(["target"], axis=1)  # target만 빼고 전부 → 문제지
y = train_csv["target"]  # target만 → 정답지

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
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

models = [model1, model2, model3, model4]

for model in models:
    # * 3, 4 훈련, 평가
    scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring="roc_auc")
    print("==========", model.__class__.__name__, "=========")
    print("roc_auc :", scores, "\n평균 roc_auc :", round(np.mean(scores), 4))

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    acc = accuracy_score(y_test, y_predict)
    print("acc :", acc)

    y_proba = model.predict_proba(x_test)[:, 1]
    print("test roc_auc :", roc_auc_score(y_test, y_proba))

"""
========== DecisionTreeClassifier =========
roc_auc : [0.56404391 0.55178925 0.55489545 0.55729917 0.55880208] 
평균 roc_auc : 0.5574
========== RandomForestClassifier =========
roc_auc : [0.82402442 0.81585933 0.82207148 0.81834003 0.82675378] 
평균 roc_auc : 0.8214

"""
"""
========== DecisionTreeClassifier =========
roc_auc : [0.55827348 0.55477785 0.5611581  0.55275535 0.55966151] 
평균 roc_auc : 0.5573
acc : 0.834675
"""
