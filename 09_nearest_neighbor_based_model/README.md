# Nearest Neighbor based Models

## 0. k-NN Classifier and Regressor

00_knn_classifier_and_regressor.ipynb 에서는 앞서 살펴본 two moon dataset 을 이용한 classification 과 time series dataset 을 이용한 regression 을 연습해 봅니다. Nearest neighbor model 에서는 distance (similarity) metric 의 정의와 k 에 의하여 클래스 간의 경계면이나 회귀직선의 모양이 달라집니다. 일반적으로 거리에 반비례하는 similarity metric 을 이용하면 k 가 아주 작은 구간을 제외하고는 robust 한 경향을 볼 수 있습니다. 이는 거리가 먼 점들은 정보력이 없는데, 그러한 점들을 많이 모아도 여전히 정보력이 적기 때문입니다. 또한 k 를 지나치게 작게 설정하면 high variance 의 과적합된 모델이 학습됩니다.

앞서 random forest 에서는 base models 간의 prediction 결과의 편차를 이용하여 confidence interval 을 정의하였습니다. 이처럼 confidence interval 은 반드시 정규 분포를 가정하지 않고도 정의할 수 있는 방법은 다양합니다. Nearest neighbor model 에서는 k 개의 neighbor points 의 prediction value 의 편차를 이용하여 confidence interval 을 정의할 수도 있습니다. 하지만 일반적으로 nearest neighbor models 에서 이 값 까지는 출력하지 않은 경우들이 있기 때문에 직접 그 값을 정의해 봅니다.

## 1. Collaborative Filtering

Collaborative filtering 은 recommender system 의 가장 기본적인 모델입니다. 한 사용자 `u` 와 취향이 비슷한 사용자들이 구매했지만, `u` 는 아직 구매하지 않은 상품을 추천하는 방식입니다. 좀 더 자세히는 한 사용자 `u` 에게 추천할 아이템 `i` 를 선택하기 위하여 `u` 와 아이템 구매 이력이 비슷한 k 명의 `v` 를 탐색합니다. 그리고 그들이 구매했던 아이템의 평점 혹은 개수로 추천할 아이템의 점수를 계산합니다. 그 중 `u` 가 사용하지 않은 아이템을 구매 가능성이 높은 아이템으로 추천합니다.

이를 위해서는 첫째, 사용자의 구매 이력을 벡터로 표현해야 합니다. 기본적인 collaborative filtering 에서는 아이템을 Boolean vector 로, 혹은 아이템에 대한 평점을 float vector 로 표현합니다. 전체 아이템의 개수 대비 구매한 아이템의 개수가 매우 작기 때문에 주로 sparse vector 로 표현됩니다. 하지만 sparse vectors 는 대부분의 값이 0 이기 때문에 cosine distance 이 1 인 (직교, orthogonal) 경우가 자주 발생합니다. 이는 실제로는 아이템 `a`, `b` 가 서로 매우 비슷한 아이템이지만, 이들의 유사성이 반영되지 않는다는 의미이기도 합니다. 이러한 점들을 해결하기 위하여 이전에는 Singular Vector Decomposition (SVD) 이나 Nonnegative Matrix Factorization (NMF) 를 이용하여 사용자의 구매 이력을 densen vector 로 표현하는 방법들이 이용되었습니다.

최근에는 neural net based embedding 방법들이 자주 이용되고 있습니다. 혹은 순차적인 구매 이력을 시계열 형식의 데이터로 취급, 다음의 아이템을 예측하는 sequential prediction models 로도 접근하기도 합니다. 이에 대해 더 관심이 있으시다면 [RecSys conferences](https://recsys.acm.org/) 의 최근의 연구들을 추가로 살펴보시기 바랍니다.

사용자의 구매 이력이 벡터로만 표현된다면, 구매 이력이 유사한 다른 사용자를 탐색한 뒤, 그들이 구매한 아이템 중 점수가 높은 아이템을 우선적으로 추천합니다. 이때, 데이터베이스에 등록된 사용자가 수백만이라면 유사한 사용자를 탐색하는데 큰 계산비용이 듭니다. 이를 해결하기 위하여 nearest neighbor search models 이 이용될 수 있습니다. 01_collaborative_filtering.ipynb 에서 Ball-Tree 를 이용하여 빠르게 유사 사용자를 탐색, 그리고 새로운 아이템을 추천하는 과정까지 연습해 봅니다.

Nearest neighbor search 분야는 최근에 다시 연구가 지속되고 있습니다. Facebook Research 의 [Faiss](https://github.com/yahoojapan/NGT) 가 최근에 제안된 방법 중에서는 가장 널리 이용되지만, 그 외에도 [NGT](https://github.com/yahoojapan/NGT) 와 같은 방법들도 존재합니다. 이에 대해 관심있으신 분들은 각각의 repository 를 들어가 보시기 바랍니다.

이 튜토리얼은 몇 가지 사전 연습이 필요합니다. appendix_collaborative_filtering_codebook.ipynb 에서는 다음을 연습합니다. (1) numpy.ndarray argsort, (2) scikit-learn pairwise distances, (3) matrix inner product (4) sparse matrix 에서 nonzero elements index 가져오기. 이들의 의미와 사용법을 확인한 뒤 본 튜토리얼을 보시기 바랍니다.



## 프로그래밍 연습

이번 실습을 통하여 다음의 항목/함수/개념을 연습 합니다.

- numpy : argsort, inner product
- scikit-learn : ball-tree, knnregressor, knnclassifier
- scipy : sparse matrix
