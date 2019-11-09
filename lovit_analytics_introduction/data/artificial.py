import numpy as np

def make_linear_regression_data(n_data=100, a=1.0, b=1.0,
    noise=1.0, x_range=(-10.0, 10.0), random_seed=None):
    """
    It generates artificial data for linear regression

    :param int n_data: Number of generated data
    :param float a: Regression coefficient a in 'y = ax + b'
    :param float b: Interpret coefficient b in 'y = ax + b'
    :param float noise: Range of residual, e = y - (ax + b)
    :param tuple x_range: size = (float, float)
    :param int_or_None random_seed: If not None, fix random seed

    :returns: x, y
        - x : numpy.ndarray, shape = (n_data,)
        - y : numpy.ndarray, shape = (n_data,)

    Usage

        >>> x, y = make_linear_regression_data()
        >>> x, y = make_linear_regression_data(
            n_data=100, a=1.0, b=1.0, noise=1.0, x_range=(-10.0, 10.0))
    """

    assert (len(x_range) == 2) and (x_range[0] < x_range[1])

    if isinstance(random_seed, int):
        np.random.seed(random_seed)

    x_scale = x_range[1] - x_range[0]
    x = np.random.random_sample(n_data) * x_scale + x_range[0]
    residual = (np.random.random_sample(n_data) - 0.5) * noise
    y = a * x + b + residual
    return x, y
