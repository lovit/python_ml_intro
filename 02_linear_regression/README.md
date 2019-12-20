# Linear Regression

## 0. Linear Regression Basic

연습용 데이터를 만들어보고 seaborn 을 이용하여 scatter plot 을 그려봅니다. Seaborn 의 relplot 과 scatterplot 함수의 차이를 알아봅니다. 선형회귀모델의 성능을 측정하는 MAE, MAPE, MSE, RMSE, R-square 를 직접 계산해 봅니다.

## 1. Multivariate Linear Regression

[Bike Sharing Demand dataset, Kaggle](https://www.kaggle.com/c/bike-sharing-demand) 의 데이터를 이용하여 수요를 예측하는 선형회귀 모델을 학습합니다. 이 과정에서 pandas 를 이용하여 변수 간 관계를 확인할 그림을 그리고, 필요한 변수들을 새로 만들어 추가, 학습에 필요한 변수를 선택, DataFrame 을 병합하는 작업을 수행합니다. Ridge regression 모델을 이용합니다.

## 2. Polynomial Regression

1차원의 입력변수 x 를 다항벡터로 변환한 뒤, 이를 이용하여 linear regression 을 학습합니다 (polynomial regression). 또한 least square 를 이용하는 Linear regression, l2 regularization 이 추가된 Ridge regression, 경사하강법을 기반으로 학습하는 SGDRegressor 의 학습법을 알아보고, 이들의 차이를 확인합니다. 반복되는 부분들을 함수로 만드는 연습과 positional argument, keyword argument 의 차이도 알아봅니다.

## 프로그래밍 연습

이번 실습을 통하여 다음의 항목/함수/개념을 연습 합니다.

- numpy: array, shape, reshape, concatenate, broadcast, random, operation (power, root, add)
- pandas: create data frame from dict, read csv
- pandas: select column, append variable, describe, get_dummies, concat
- python: list, dict, file open, function
- scikit-learn: fit, predict, fit_predict
- seaborn: scatterplot vs relplot
- seaborn: get_figure, save figure
