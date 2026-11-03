import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

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
r2 : [0.60290952 0.60548824 0.65590183 0.6157696  0.5917572 ] 
평균 r2 : 0.61
test r2_score : 0.5876612130777672
========== RandomForestRegressor =========
r2 : [0.8019715  0.8050522  0.8106233  0.80516251 0.79916775] 
평균 r2 : 0.8
test r2_score : 0.802394829226576
========== GradientBoostingRegressor =========
r2 : [0.78296687 0.79602013 0.795555   0.78366896 0.78543238] 
평균 r2 : 0.79
test r2_score : 0.7805147880657438
========== XGBRegressor =========
r2 : [0.83647596 0.84005549 0.82732296 0.83453094 0.83432637] 
평균 r2 : 0.83
test r2_score : 0.8285390017295229
"""
