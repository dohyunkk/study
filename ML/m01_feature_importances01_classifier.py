"""
#! 분류 모델 Classifier
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# * 1. 데이터
x, y = load_iris(return_X_y=True)  # ? 이렇게 해놓으면 x, y가 분리가 된다.
print(x.shape, y.shape)  # ? (150, 4) (150,)

random_state = 123


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=random_state,
    stratify=y,
)

# * 2. 모델구성
model1 = DecisionTreeClassifier(random_state=random_state)
model2 = RandomForestClassifier(random_state=random_state)
model3 = GradientBoostingClassifier(random_state=random_state)
model4 = XGBClassifier(random_state=random_state)

models = [model1, model2, model3, model4]


for model in models:
    # * 3. 컴파일, 훈련
    model.fit(x_train, y_train)
    print("==========", model.__class__.__name__, "=========")

    # * 4. 평가 예측
    score = model.score(x_test, y_test)  # ? 이제 evaluate 대신 score
    print("acc : ", score)
    print(model.feature_importances_)


"""
DecisionTreeClassifier
random_state = 123
# ? 0.8333333333333334


random_state = 12313
========== DecisionTreeClassifier =========
acc :  0.9
========== RandomForestClassifier =========
acc :  0.9
========== GradientBoostingClassifier =========
acc :  0.9
========== XGBClassifier =========
acc :  0.9

"""
