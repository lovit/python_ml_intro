# Neural Network for Non-linear Classification & Regression

## 0. image data handling

../01_python_basic/image_handling_io_resize_flatten.ipynb 에서 JPEG 와 같은 이미지 파일을 numpy.ndarray 형식으로 읽어들이고, 이를 머신러닝 알고리즘이 이용할 수 있는 크기로 resize 하거나 flatten vector 로 변경하는 방법에 대하여 연습합니다.

## 1. scikit-learn feed-forward network 를 이용한 fashion-MNIST 분류

00_fashion_mnist_ff_sklearn.ipynb 에서 scikit learn 에서 제공하는 multi-layer feed forward neural network 모델을 이용하여 fashion-MNIST 데이터를 분류하는 모델을 학습합니다. 이때 Python time package 를 이용하여 학습 및 테스트 시간을 측정하는 방법 및 zip, enumerate 와 같은 Python 의 효율적인 함수들의 사용법도 익혀봅니다. 또한 scikit-learn 의 많은 알고리즘에서 지원하는 warm_start (partial_fit) 기능도 알아봅니다.

## 2. Keras Convolutional Neural Network 를 이용한 fashion-MNIST 분류

01_fashion_mnist_cnn_keras.ipynb 에서 tensorflow 를 backend 로 이용하는 Keras 를 이용하여 간단한 convolutional neural network 모델을 만들어봅니다. 앞서 연습한 feed forward neural network 도 재구현하여 두 모델의 성능을 비교해 봅니다. 이 과정에서 Sequential 을 이용하여 다양한 레이어로 딥러닝 모델을 구성하는 방법을 연습합니다. 특히 CNN 모델을 구성하는 방법에 대하여 살펴봅니다. Initializer, optimizer, loss 를 정의하여 모델을 학습하는 방법도 살펴봅니다. 그리고 모델에 test data 가 입력되었을 때 각 레이어에서의 hidden output vectors 를 가져오는 방법, 학습된 모델을 저장하고 불러들이는 방법에 대해서도 살펴봅니다.


## 프로그래밍 연습

이번 실습을 통하여 다음의 항목/함수/개념을 연습 합니다.

- bokeh : output_notebook, figure, scatter, image, gridplot
- keras : Sequential, Conv2D, Dense, Flatten, MaxPooling2D, Model.summary(), fit, history, predict, getting hidden vectors, save/load
- numpy : flatten
- python : enumerate, zip, PIL, time
- scikit-learn : MLPClassifier, warm_start
