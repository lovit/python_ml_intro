# Clustering

## 0. clustering IRIS with kmeans and evaluation

군집화의 대표적인 알고리즘인 k-means 를 이용하여 (150, 4) 크기의 IRIS data 를 k 개의 군집으로 나눠봅니다. 군집화의 성능을 비지도기반으로 측정하는 방법 중 하나로 silhouette score 가 있지만, 이는 저차원 데이터에서만 잘 정의된 방법입니다. 하지만 다행히도 IRIS data 는 4 차원이니 이를 이용하여 k 에 따른 군집화 성능을 표현해 봅니다. Bokeh 를 이용하면 plotting 을 한 뒤, 각 그림에 여러 설명을 적거나 혹은 hover tool 로 숨김 표현을 적을 수 있습니다. 00_kmeans_iris_clustering_evaluation 에서는 k-means 학습법 외에도 bokeh 의 widgets 를 이용하여 plot 옆에 HTML 형식으로 텍스트를 기술하는 연습도 함께 수행합니다.

## 1. adding constraints to k-means

k-means 는 머신러닝 알고리즘 중에서 매우 간단하게 구현할 수 있는 것 중 하나입니다. 하지만 간단한만큼 큰 단점이 있습니다. 데이터에는 분명 outliers 가 존재할텐데, k-means 는 모든 데이터를 하나의 군집으로 할당합니다. 01_kmeans_from_scratch_dev.ipynb 에서는 k-means 를 밑바닥부터 직접 구현하면서 outliers 를 스스로 제거하기 위한 기능을 추가하는 연습을 합니다. 이를 위하여 최대한 `kmeans()` 함수의 내용을 부분별로 나눠 각각 함수로 처리해 두었습니다. 그리고 constraint 를 추가할 부분을 `reassign()` 함수에 `#TODO` 로 표시해 두었습니다. 여러 분의 생각대로 그 기능을 직접 구현해 보시기 바랍니다. 가능한 구현 코드 예시는 개인 작업중인 [ekmeans](https://github.com/lovit/ekmeans/blob/98899ef87b3c500929254cdac459a67838f28163/ekmeans/cluster.py#L542) 에 기록해 두었습니다.

물론 위의 constraint 부분을 구현하지 않는다면 우리가 알고 있는 k-means 가 됩니다. 01_kmeans_from_scratch.ipynb 에서는 위의 튜토리얼에서 개발한 k-means 코드를 `kmeans_from_scratch` 패키지로 만든 뒤, `kmeans()` 를 import 하여 활용하는 연습을 합니다.

## 2. comparison of clustering algorithms and visulization of clustering results

02_sklearn_comparison.ipynb 에서는 IRIS data 에 대하여 k-means 외의 DBSCAN, Hierarchical clustering, Gaussian Mixture Model 의 사용법에 대하여 살펴봅니다. 또한 clustering 의 결과를 시각화 하기 위하여 앞선 튜토리얼에서는 (150, 4) 크기의 벡터를 PCA 를 이용하여 (150, 2) 차원으로 변환한 뒤, 이를 clustering labels 와 함께 표현하였습니다. 이는 embedding (dimension reduction) 과정에서는 clustering labeling 의 정보가 사용되지 않았다는 의미입니다. 하지만 때로는 (150, 4) 의 구조를 잘 표현하기 보다, 어떤 데이터들이 각각 서로 다른 군집으로 구분되었는지를 표현하고 싶기도 합니다. 이를 위해서는 clustering labels 의 정보를 이용하는 embedding 방법이 필요합니다. 이전에 살펴본 UMAP 은 supervised embedding 기능을 제공합니다. 이 튜토리얼에서는 PCA 와 UMAP 을 이용한 군집화 시각화 결과를 비교해 봅니다.


## 프로그래밍 연습

이번 실습을 통하여 다음의 항목/함수/개념을 연습 합니다.

- bokeh : widgets
- scikit-learn : kmeans, dbscan, hierarchical clustering, gaussian mixture model
