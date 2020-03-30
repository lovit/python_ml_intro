# Decision Tree

## 0. Twomoon data 를 이용한 decision tree 실습

Decision Tree 도 hyper parameters 를 어떻게 설정하느냐에 따라 클래스 간 경계면의 모습 및 데이터가 존재하지 않는 공간에서의 판별 경향이 달라집니다. 그 관계성을 파악하기 위하여 decision tree 및 tree based ensembles 들은 2 차원 데이터에서 hyper parameters 혹은 데이터 분포에 따라 경계면의 모습을 확인할 필요가 있습니다. 데이터를 만들고, decision tree 에 의한 경계면을 시각적으로 표현하기 위한 함수들은 `decision_tree_utils.py` 에 미리 만들어 두었습니다. 이를 이용하여 00_decision_tree_twomoon.ipynb 에서 데이터를 만들고 판별 모델을 학습해 봅니다.

이 튜토리얼에서 그린 그림들은 `./figures` 폴더에 저장되어 있습니다.

## 프로그래밍 실습

- scikit-learn : Decision Tree
