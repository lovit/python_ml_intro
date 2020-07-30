# Preprocessing and feature engineering

## 0. umap

Fashion-MNIST 데이터를 이용하여 UMAP 을 학습합니다. UMAP 은 unsupervised, supervised, semi-supervised learning 모두 실습합니다. 이 과정에서 site-packages 외부에 설치된 패키지를 import 하는 방법을 연습합니다. 또한 PIL 을 이용하여 numpy.ndarray 형식인 벡터를 이미지로 표현하는 방법에 대해서 살펴봅니다.

## 1. iris umap tsne pca

Iris 데이터를 이용하여 UMAP, t-SNE, PCA 를 모두 이용하는 연습을 합니다. t-SNE 는 perplexity 조절을, UMAP 은 target_weight 를 조절하여 그 영향력을 살펴봅니다.

## 2. pandas 를 이용한 영화의 평점 분포 계산

../01_python_basic/pandas_introduction.ipynb 와 pandas_summarize_movie-rates_lecture.ipynb 를 연습합니다. 여러 개의 DataFrame 을 merge 하는 연습을 합니다. 이는 데이터베이스의 join 입니다.

## 프로그래밍 연습

이번 실습을 통하여 다음의 항목/함수/개념을 연습 합니다.

- pandas: index, merge, groupby, reset index
- python: sys.path, PIL
- scikit-learn: PCA, TSNE
