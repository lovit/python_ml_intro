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
        - y_true : numpy.ndarray, shape = (n_data,)

    Usage

        >>> x, y, _ = make_linear_regression_data()
        >>> x, y, y_true = make_linear_regression_data(
            n_data=100, a=1.0, b=1.0, noise=1.0, x_range=(-10.0, 10.0))
    """

    assert (len(x_range) == 2) and (x_range[0] < x_range[1])

    if isinstance(random_seed, int):
        np.random.seed(random_seed)

    x_scale = x_range[1] - x_range[0]
    x = np.random.random_sample(n_data) * x_scale + x_range[0]
    residual = (np.random.random_sample(n_data) - 0.5) * noise
    y = a * x + b + residual
    y_true = a * x + b
    return x, y, y_true

def make_polynomial_regression_data(n_data=100, degree=2, coefficients=None,
    coefficient_scale=3.0, noise=0.1, x_range=(-1.0, 1.0), random_seed=None):
    """
    It generates artificial data for linear regression

    :param int n_data: Number of generated data
    :param int degree: Degree of polynomial
    :param list_or_None coefficients: Coefficients bi such that y = b0 + sum_{i=1 to degree} bi x x^i
    :param float coefficient_scale: Range of coefficients bi.
        Default is 1.0
    :param float noise: Range of residual, e = y - f(x)
    :param tuple x_range: size = (float, float)
    :param int_or_None random_seed: If not None, fix random seed

    :returns: x, y
        - x : numpy.ndarray, shape = (n_data,)
        - y : numpy.ndarray, shape = (n_data,)
        - y_true : numpy.ndarray, shape = (n_data,)

    Usage

        >>> x, y, _ = make_linear_regression()
        >>> x, y, y_true = make_linear_regression(
            n_data=100, a=1.0, b=1.0, noise=1.0, x_range=(-10.0, 10.0))
    """

    if (not isinstance(degree, int)) or (degree < 0):
        raise ValueError(f'degree must be nonnegative integer, however input is {degree}')

    if isinstance(random_seed, int):
        np.random.seed(random_seed)

    if coefficients is None:
        coefficients = coefficient_scale * (np.random.random_sample(degree + 1) - 0.5)

    len_coef = len(coefficients)
    if len_coef != degree + 1:
        raise ValueError('The length of coefficients must be degree'\
            f'However, length is {len_coef} with degree = {degree}')

    x_scale = x_range[1] - x_range[0]
    x = np.random.random_sample(n_data) * x_scale + x_range[0]

    y_true = np.zeros(n_data)
    for p, coef in enumerate(coefficients):
        y_true = y_true + coef * np.power(x, p)
    residual = (np.random.random_sample(n_data) - 0.5) * noise
    y = y_true + residual
    return x, y, y_true

def generate_svc_data(n_data=50, p_noise=0.0):
    """
    Arguments
    ---------
    n_data : int
        The number of generated points
    p : float
        Proportion of noise data

    Returns
    -------
    X : numpy.ndarray
        Shape = (n_data, 2)
        The random seed is fixed with 0
    label : numpy.ndarray
        Shape = (n_data,)

    Usage
    -----
    To generate toy data

        >>> X, label = generate_svc_data(n_data=50, p_noise=0.0)

    Remove three points to make linear separable dataset

        >>> subindices = np.array([i for i in range(50) if not i in [8, 13, 14]])
        >>> X_ = X[subindices]
        >>> label_ = label[subindices]

    Train linear SVM

        >>> from sklearn.svm import SVC
        >>> svm = SVC(C=10.0, kernel='linear')
        >>> svm.fit(X_, label_)

    Train RBF kernel SVM

        >>> svm = SVC(C=10.0, kernel='rbf')
        >>> svm.fit(X, label)

    Draw scatter plot overlapped with activation map

        >>> from bokeh.plotting import show

        >>> sv = np.zeros(X.shape[0], dtype=np.int)
        >>> sv[svm.support_] = 1
        >>> decision = svm.decision_function(X)

        >>> p = draw_activate_image(svm, X, resolution=500, decision=True)
        >>> p = scatterplot_2class(X, label, sv, decision, p=p, size=10)
        >>> show(p)
    """
    np.random.seed(0)
    X = np.random.random_sample((n_data*2, 2))
    label = np.zeros(n_data*2, dtype=np.int)

    # set label
    dist = np.linalg.norm(X - np.array([0.4, 0.9]), axis=1)
    label[(0.35 < dist) & (dist <= 0.4)] = -1
    label[(0.4 < dist) & (dist < 0.9)] = 1
    label[(0.9 <= dist)] = -1

    if (n_data * p_noise) > 0:
        n_noise = int(label.shape[0] * p_noise)
        indices = np.random.permutation(n_noise)[:n_noise]
        label[indices] = np.random.randint(0, 2, n_noise)

    indices = np.where(label >= 0)[0]
    X = X[indices][:n_data]
    label = label[indices][:n_data]

    return X, label

def generate_svr_data(n_data=200, n_repeats=5):
    """
    Arguments
    ---------
    n_data : int
        The number of generated points
    n_repeats : int
        The size of x = n_data * n_repeats

    Returns
    -------
    Random seed is fixed with 0

    x_line : numpy.ndarray
        Shape = (n_data,)
        Array of [0, 1, 2, ... n_data-1]
    x : numpy.ndarray
        Shape = (n_data * n_repeats,)
        Array of [0, 1, 2, ... n_data-1, 0, 1, 2, ... n_data-1, ...]
    y_line : numpy.ndarray
        Shape = (n_data,)
        Timeseries type column vector
    y : numpy.ndarray
        Shape = (n_data * n_repeats,)
        y_base += noise

    Usage
    -----
    """

    np.random.seed(0)
    x_line = np.arange(n_data)
    x = np.concatenate([x_line for _ in range(n_repeats)])
    y_line = np.random.randn(n_data).cumsum()
    y = np.concatenate([y_line + np.random.randn(n_data) for _ in range(n_repeats)])

    return x_line, x, y_line, y
