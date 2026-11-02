import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# * 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

# print(x, y)
# print(x.shape, y.shape)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# * 2. 모델
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()


# * 3, 4 훈련, 평가
scores = cross_val_score(model, x, y, cv=kfold)

print("r2 :", scores, "\n평균 r2 :", round(np.mean(scores), 2))

"""
XGBRegressor
r2 : [0.82742129 0.84140555 0.82123327 0.84284717 0.8478747 ] 
평균 r2 : 0.84

"""
