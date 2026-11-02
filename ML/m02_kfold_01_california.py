import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# * 1. 데이터
x, y = fetch_california_housing(return_X_y=True)


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
    scores = cross_val_score(model, x, y, cv=kfold)
    print("==========", model.__class__.__name__, "=========")
    print("r2 :", scores, "\n평균 r2 :", round(np.mean(scores), 2))

"""
XGBRegressor
r2 : [0.82742129 0.84140555 0.82123327 0.84284717 0.8478747 ] 
평균 r2 : 0.84

"""
