# Logistic Regression for Classification

## 0. Logistic Regression

Numerical variables 로 이뤄진 데이터를 이용하여 Categorical variable 을 예측하는 Logistic regression 을 학습합니다. 이 과정에서 Pandas DataFrame 에서 일부 columns 을 선택하여 학습데이터를 만드는 연습과, Seaborn 을 이용하여 변수값의 분포를 histogram 으로 그리는 연습을 수행합니다. 또한 scikit-learn 에서 제공하는 metrics (precision, recall, f1 score 등) 을 이용하여 모델의 학습 성능을 평가합니다.

## 1. Logistic Regression with Titanic Dataset

[Titanic dataset, Kaggle](https://www.kaggle.com/c/titanic) 을 이용하여 타이타닉 승객의 생존을 예측하는 Logistic regression 을 학습합니다. 이 과정에서 categorical variables 을 dummy variables 로 만들거나, numerical variables 을 binning 하여 categorical variables 로 만드는 연습을 합니다. 또한 Pandas 에서 NaN 값을 처리하는 방법을 연습합니다.


## 프로그래밍 연습

이번 실습을 통하여 다음의 항목/함수/개념을 연습 합니다.

- pandas: Series, to_numpy, unique, value_count, map, handling NaN
- python: mutable default argument, str split
- scikit-learn: LogisticRegression, predict_proba, classification_report, metrics
- seaborn: distplot, histogram
