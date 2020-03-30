# Support Vector Machine


## 0. Toydata 를 이용한 svm & svr 모델 실습

Support Vector Machine 은 support vectors 와 경계면의 모습을 살펴보는 것이 중요하기 때문에 2 차원의 인공데이터를 생성하여 이를 통해 Support Vector Classifer 와 Support Vector Regression 의 사용법을 익혀봅니다. 데이터의 생성함수와 각 클래스의 영역을 표시하는 함수는 모두 svm_utils.py 에 미리 만들어 두었습니다. 이 튜토리얼에서 모델을 학습하고 어떤 점들이 support vectors 인지 확인하며 decision score 를 계산하는 방법을 익혀봅니다. 또한 `svm_utils.py` 에 미리 만들어둔 시각화 함수들을 이용하여 SVC 에 의하여 나뉘어지는 클래스 별 공간에 대해서도 시각적으로 살펴봅니다. RBF kernel SVC 를 학습할 경우, gamma 를 조절함에 따라서 각 클래스 별 영역의 모양이 달라집니다. gamma 를 작은 수로 설정하면 경계면은 대체로 부드러운 곡선 모양을 띄며, 데이터가 존재하지 않은 영역도 높은 점수로 한 클래스에 속합니다. 반대로 gamma 가 매우 커질 경우 경계면은 대체로 복잡한 모양을 띄며, 데이터가 존재하지 않는 영역들 (두 클래스 경계의 중간 등) 은 어느 클래스에도 높은 점수를 받지 못하는 현상을 볼 수 있습니다. 이는 강의노트에서 기술한 "각 basis 마다 자신이 확신할 수 있는 구 (sphere) 형태의 영역"을 지니기 때문입니다.

이 튜토리얼에서 그린 그림들은 `./figures` 폴더에 저장되어 있습니다.

## appendix

appendix_tayler_series.ipynb 에서는 Tayler expansion 을 이용한 근사 함수를 직접 그려봅니다.

appendix_svm_multiclass_classification.ipynb 에서는 SVC 에서 multi class classification 을 해결하기 위해 이용하는 one versus one 과 one versus others 에 따른 hyper planes 의 개수의 차이를 살펴봅니다.

## 프로그래밍 실습

- bokeh : svm_utils.py 에 구현된 함수들이 조금 길지만, 지금까지 연습했던 기능만을 이용하면 튜토리얼의 그림들을 그릴 수 있습니다.
- scikit-learn : SVC, SVR
