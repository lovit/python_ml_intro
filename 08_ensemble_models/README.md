# Tree based ensemble models

## 0. Classifying Two-moon data with Random Forest

Random Forest 는 여러 개의 decision trees 로 이뤄진 ensemble model 입니다. 그렇기 때문에 decision tree 튜토리얼에서 다뤘던 동일한 데이터에 대하여 두 모델 간의 차이를 비교해봅니다. 데이터 생성 및 클래스 영역의 시각화 함수는 `ensemble_utils.py` 에 미리 만들어 뒀습니다. `07_decision_tree/decision_tree_utils.py` 과 동일한 함수가 포함되어 있지만, 편리성을 위해 복사하였습니다.

Ensemble models 은 여러 개의 weak models 의 성능을 종합하여 더 좋은 일반화 성능을 이끌어낸다고 합니다. 이를 variance and bias 관점에서 살펴보면 deep depth 로 학습하는 decision tree 는 bias 가 작은 대신 (overfitted) variance 가 매우 큽니다. Random Forest 는 bagging 을 이용하여 여러 개의 과적합 (overfitted) 된 모델들의 성능을 종합하여 variance 를 낮춰갑니다. 또한 bagging 과정에서 학습데이터의 일부만 이용하기 때문에 그 자체로 전체 데이터를 이용하는 단일 decision tree 보다 weak model 입니다. 그렇기 때문에 random forest 를 학습할 때에는 depth 를 충분히 크게 설정해야 합니다.

00_random_forest_twomoon.ipynb 에서 여러 hyper parameters 를 조절해보며 경계면의 변화를 살펴보되, 반드시 `max_depth` 에 의한 경계면의 변화를 확인해 보시기 바랍니다. `max_depth=4` 정도로는 100 개의 decision trees 를 이용하여도 two moon data 도 잘 분류하지 못합니다.

## 1. Timeseries Regression with Random Forest

Bagging 의 장점 중 하나는 특정 데이터에 대하여 base models 이 예측하는 값들이 다르다는 점입니다. 이들을 이용하면 empirical distribution 에 의한 confidence interval 을 계산할 수 있습니다. Random Forest 의 regression 의 사용법은 classification 과 크게 다르지 않습니다. 그리고 위의 classification 의 튜토리얼에서도 confidence 를 계산하는 과정이 포함되어 있지만, 01_random_forest_regression.ipynb 에서는 그 부분을 더욱 신경써서 살펴보시기 바랍니다.

## 2. XGBoost Classifier with Titanic data and XGBoost Regressor with Timeseries data

Boosting 은 bagging 과 접근법이 다릅니다. Bagging 은 low variance and low bias 를 얻기 위하여 high variance and low bias 인 여러 개의 모델을 종합하지만, boosting 은 low variance and high bias 인 모델들을 종합하여 bias 를 줄여갑니다. 이때는 모델을 under-fitting 하여야 합니다. 그렇기 때문에 XGBoost 를 이용할 때에는 개별 decision tree 의 max depth 를 작게 설정해야 합니다.

02_xgboost_classification_regression.ipynb 에서는 Titanic data 를 이용하여 classification 을 수행합니다. 하지만 데이터의 개수에 비하여 지나치게 많은 base models 를 이용하기에 학습 정확도는 계속 오르지만 일반화 성능은 오르지 않는 모습을 살펴볼 수 있습니다. 이후에 규모가 큰 데이터에 대해서도 반드시 동일한 그래프를 그려보시기 바랍니다. 또한 boosting 은 randomness 에 의존하지 않으며, `f = f1 + f2 + f3` 까지 모델을 학습한 뒤, 실제로 `f' = f1 + f2` 로만 이용할 수도 있습니다. Regression 부분에서 `predict(..., ntree_limit=10)` 처럼 `ntree_limit` 을 설정하는 부분도 함께 살펴보시기 바랍니다.

## 3. Isolation Forest for outlier detection

03_isolation_forest.ipynb 에서 isolation forest 를 이용하여 outliers 를 탐색하는 연습을 합니다. Outliers 는 문제에 따라 그 정의가 다릅니다. 이 튜토리얼에서는 밀도가 높은 지역에서 떨어져 있는 점들을 outliers 라 정의하였으며, 이 경우에는 isolation forest 의 outliers 에 대한 기준이 부합합니다. 하지만 isolation forest 를 이용할 때 고민스러운 부분은 threshold 를 어떤 값으로 정의할지 가이드가 없다는 점입니다. 이를 위해 decision function 의 score 의 분포를 살펴보시기 바랍니다. 그리고 threshold 가 지나치게 크면 정상인 데이터이지만, 군집의 가장자리에 위치하는 점들이 outliers 로 판단될 수 있음도 확인해보시기 바랍니다.


## 프로그래밍 연습

이번 실습을 통하여 다음의 항목/함수/개념을 연습 합니다.

- scikit-learn : Random Forest classification / regression, Isolation forest
- xgboost
