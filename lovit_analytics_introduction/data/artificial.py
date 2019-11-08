import numpy as np

def make_linear_regression_data(n_data=100, a=1.0, b=1.0, noise=1.0, x_range=(-10.0, 10.0)):
    """
    :param int n_data: 생성할 데이터 개수
    :param float a: y = ax + b 에서의 선형회귀계수 a
    :param float b: y = ax + b 에서의 y 절편 b
    :param float noise: e = y - (ax + b) 에서의 e 의 분포 범위
    :param tuple x_range: (float, float) 형식의 x 값 범위
    :return x, y: n_data 개수의 1차원 데이터

    Usage

        >>> x, y = make_linear_regression()
        >>> x, y = make_linear_regression(
            n_data=100, a=1.0, b=1.0, noise=1.0, x_range=(-10.0, 10.0))
    """

    assert (len(x_range) == 2) and (x_range[0] < x_range[1])

    x_scale = x_range[1] - x_range[0]
    x = np.random.random_sample(n_data) * x_scale + x_range[0]
    residual = (np.random.random_sample(n_data) - 0.5) * noise
    y = a * x + b + residual
    return x, y
