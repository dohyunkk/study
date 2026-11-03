import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# * 1. 데이터
path = "./_data/kaggle_bike/"  # 상대경로

train_csv = pd.read_csv(path + "train.csv", parse_dates=["datetime"])
train_csv["year"] = train_csv["datetime"].dt.year
train_csv["month"] = train_csv["datetime"].dt.month
train_csv["hour"] = train_csv["datetime"].dt.hour
train_csv["dayofweek"] = train_csv["datetime"].dt.dayofweek

test_csv = pd.read_csv(path + "test.csv", index_col=0)
# print(test_csv.shape) #(6493, 8)
submission = pd.read_csv(path + "sampleSubmission.csv", index_col=0)
# print(submission.shape) #(6493, 1)

x = train_csv.drop(["datetime", "casual", "registered", "count"], axis=1)
y = train_csv["count"]

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
model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = XGBRegressor()

models = [model1, model2, model3, model4]

for model in models:
    # * 3, 4 훈련, 평가
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print("==========", model.__class__.__name__, "=========")
    print("r2 :", scores, "\n평균 r2 :", round(np.mean(scores), 2))

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    print("test r2_score :", r2)

"""
========== DecisionTreeRegressor =========
r2 : [0.89858223 0.90754753 0.89335169 0.89933306 0.90001147] 
평균 r2 : 0.9
========== RandomForestRegressor =========
r2 : [0.94754175 0.95102253 0.95324517 0.95108406 0.95315025] 
평균 r2 : 0.95
========== GradientBoostingRegressor =========
r2 : [0.85834973 0.85037885 0.87597244 0.87748556 0.86193892] 
평균 r2 : 0.86
========== XGBRegressor =========
r2 : [0.95436919 0.95481282 0.95529848 0.95400727 0.95461923] 
평균 r2 : 0.95
"""
"""
========== DecisionTreeRegressor =========
r2 : [0.89001878 0.89168629 0.89308355 0.9096982  0.88463803] 
평균 r2 : 0.89
test r2_score : 0.8965849947725757
========== RandomForestRegressor =========
r2 : [0.93744242 0.94554416 0.95141006 0.94346774 0.94864626] 
평균 r2 : 0.95
test r2_score : 0.9474006205355535
========== GradientBoostingRegressor =========
r2 : [0.85842076 0.86109301 0.86742556 0.8562884  0.87317047] 
평균 r2 : 0.86
test r2_score : 0.8582355792671137
========== XGBRegressor =========
r2 : [0.94596457 0.9495846  0.95327801 0.94841272 0.95459801] 
평균 r2 : 0.95
test r2_score : 0.9543691873550415

"""
