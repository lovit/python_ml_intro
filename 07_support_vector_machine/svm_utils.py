import numpy as np
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.palettes import PuBu9, OrRd9
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool, BoxZoomTool, WheelZoomTool, ResetTool, SaveTool, PanTool


def initialize_figure(height, width, title, X, margin, decision):
    """
    Arguments
    ---------
    height : int
        Figure height
    width : int
        Figure width
    title : str or None
        Figure title
    X : numpy.ndarray
        Data to be plotted. It will be used to set `x_range` and `y_range`
        Shape of X = (n_data, 2)
    margin : float
        Padding size. The range of figures is
        `x_range` = (x_min - margin, x_max + margin)
        `y_range` = (y_min - margin, y_max + margin)
    decision : Boolean or None
        If not None or True, hover tool shows decision value of each point

    Returns
    -------
    p : bokeh.plotting.Figure
        Initialized figure.
        Hovertool looks up objects of which name is `scatter`
    """
    tooltips = [
        ("data index", "$index point"),
        ("label", "@label"),
        ("(x, y)", "($x, $y)"),
        ('Support Vector', '@sv')
    ]

    if (decision is not None) or (decision == True):
        tooltips.append(('decision value', '@decision'))

    tools = [
        HoverTool(names=["scatter"]),
        PanTool(),
        BoxZoomTool(),
        WheelZoomTool(),
        ResetTool(),
        SaveTool()
    ]

    x_range, y_range = check_range(X, margin)

    p = figure(height=height, width=width, title=title, tooltips=tooltips,
        tools=tools, x_range=x_range, y_range=y_range)

    return p

def check_range(X, margin=0):
    """
    Arguments
    ---------
    X : numpy.ndarray
        Data to be plotted. It will be used to set `x_range` and `y_range`
        Shape of X = (n_data, 2)
    margin : float
        Padding size. The range of figures is
        `x_range` = (x_min - margin, x_max + margin)
        `y_range` = (y_min - margin, y_max + margin)

    Returns
    -------
    x_range : tuple of float
        (x_min - margin, x_max + margin)
    y_range : tuple of float
        (y_min - margin, y_max + margin)
    """
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    x_range = (x_min - margin, x_max + margin)
    y_range = (y_min - margin, y_max + margin)
    return x_range, y_range

def scatterplot_2class(X, label, sv=None, decision=None, size=5,
    height=600, width=600, title=None, p=None, margin=0.05):
    """
    Arguments
    ---------
    X : numpy.ndarray
        Data to be plotted. It will be used to set `x_range` and `y_range`
        Shape of X = (n_data, 2)
    label : numpy.ndarray
        Shape of label = (n_data,)
    sv : numpy.ndarray or None
        Boolean array. 1 means that corresponding point is support vector
        Shape of sv = (n_data,)
    decision : numpy.ndarray or None
        The decision value or coefficient of dual variable of each point
        Shape of decision = (n_data,)
    size : int
        The size of support vector
        The size of other points is `size` - 3
        Therefore, set `size` with larger than 4
    height : int
        Figure height
    width : int
        Figure width
    title : str or None
        Figure title
    p : bokeh.plotting.Figure or None
        If `p` is None, it initializes new figure.
        Hovertool looks up objects of which name is `scatter`
    margin : float
        Padding size. The range of figures is
        `x_range` = (x_min - margin, x_max + margin)
        `y_range` = (y_min - margin, y_max + margin)

    Returns
    -------
    p : bokeh.plotting.Figure
        Figure of scatterplot
    """

    if p is None:
        p = initialize_figure(height, width, title, X, margin, decision)

    if sv is None:
        sv = np.zeros(X.shape[0], dtype=np.int)

    # prepare alpha, color & size
    alpha = [0.9 if svi else 0.3 for svi in sv]
    colormap = ['#2b83ba', '#d7191c']
    color = [colormap[l] for l in label]
    size = [size if svi else size-3 for svi in sv]

    # prepare source
    x, y = X[:,0], X[:,1]
    data = {
        'x': x, 'y': y, 'sv': sv, 'label': label,
        'color': color, 'alpha': alpha, 'size':size
    }
    if (decision is not None) and (hasattr(decision, '__len__')):
        data['decision'] = decision
    source = ColumnDataSource(data)

    # plotting
    p.scatter(x='x', y='y', fill_color='color', line_color=None,
              fill_alpha='alpha', size='size', source=source, name='scatter')

    return p

def append_circles(X, label, radius, p, alpha=0.1):
    """
    Arguments
    ---------
    X : numpy.ndarray
        Data to be plotted. It will be used to set `x_range` and `y_range`
        Shape of X = (n_data, 2)
    label : numpy.ndarray
        Shape of label = (n_data,)
    radius : float
        Radius of each circle
    p : bokeh.plotting.Figure
        Initialized figure.
    alpha : float
        Transparency of circle

    Returns
    -------
    p : bokeh.plotting.Figure
        Circles appended figure
    """

    colormap = ['#2b83ba', '#d7191c']
    color = [colormap[l] for l in label]
    x, y = X[:,0], X[:,1]
    source = ColumnDataSource({'x': x, 'y': y, 'color':color})
    p.circle(x='x', y='y', fill_color='color', line_color=None,
             radius=radius, alpha=alpha, source=source)

    return p

def draw_activate_image(model, X, resolution=100, height=600, width=600,
    title=None, p=None, margin=0.05, alpha=0.2, decision=False):
    """
    Arguments
    ---------
    model : sklearn.svm.BaseSVC
        Trained support vector classifier model
    X : numpy.ndarray
        Data to be plotted. It will be used to set `x_range` and `y_range`
        Shape of X = (n_data, 2)
    resolution : int
        Number of points located on one axis.
    height : int
        Figure height
    width : int
        Figure width
    title : str or None
        Figure title
    p : bokeh.plotting.Figure or None
        If `p` is None, it initializes new figure.
        Hovertool looks up objects of which name is `scatter`
    margin : float
        Padding size. The range of figures is
        `x_range` = (x_min - margin, x_max + margin)
        `y_range` = (y_min - margin, y_max + margin)
    alpha : float
        Transparency of image
    decision : Boolean
        It is used to initialize figure. If True, the hovertool shows
        decision value or dual coefficient.

    Returns
    -------
    p : bokeh.plotting.Figure
        Class activation map appended figure.
        If the predicted value stands for class `0`, the region fills with blue
        If the predicted value stands for class `1`, the region fills with red
    """

    if p is None:
        p = initialize_figure(height, width, title, X, margin, decision)

    # make grid
    (x_min, x_max), (y_min, y_max) = check_range(X, margin)
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    X_test = np.array([(x, y) for y in ys for x in xs])

    # predict
    label = model.decision_function(X_test)
    label[np.where(label < -1)[0]] = -1
    label[np.where(label > 1)[0]] = 1

    # as image
    label = 255 / 2 * (label + 1)
    label = np.array(label, dtype=np.uint8)
    img = label.reshape(resolution, -1)

    palette = PuBu9[:2] + PuBu9[-3:]
    palette += reversed(OrRd9[:2] + OrRd9[-3:])

    p.image(image=[img], x=x_min, y=y_min, dw=x_max-x_min, dh=y_max-y_min,
            palette=palette, alpha=alpha)

    return p

def append_rbf_radial_basis(rbf_svm, X, label, gamma, p,
    threshold=0.05, circle_alpha=0.3):
    """
    Arguments
    ---------
    rbf_svm : sklearn.svm.BaseSVC
        Trained support vector classifier model with `kernel='rbf'`
    X : numpy.ndarray
        Data to be plotted. It will be used to set `x_range` and `y_range`
        Shape of X = (n_data, 2)
    label : numpy.ndarray
        Shape = (n_data,)
    gamma : float
        Gamma of RBF kernel, K(x,y) = exp(-gamma(|x - y|_2^2)
    p : bokeh.plotting.Figure
        Initialized figure
    threshold : float
        Threshold of K(x,y). The smaller `threshold` leads the larger rbf basis circle.
    circle_alpha : float
        0 < `circle_alpha` < 1
        Transparency of circle.

    Returns
    -------
    p : bokeh.plotting.Figure
        RBF basis circles appended figure
    """

    radius = -np.log(threshold) / gamma
    X_sv = X[rbf_svm.support_]
    label_sv = label[rbf_svm.support_]
    p = append_circles(X_sv, label_sv, radius, p, alpha=circle_alpha)
    return p

def scatterplot_timeseries(x, y, y_line=None, height=400, width=800, title=None,
    p=None, margin=2.0, point_color='grey', line_color='#2b83ba', size=2):
    """
    Arguments
    ---------
    x : numpy.ndarray
        Data to be plotted. Shape of x = (n_data,)
    y : numpy.ndarray
        Data to be plotted. Shape of y = (n_data,)
    y_line : numpy.ndarray
        Data of true line. Shape of y_line = (n_line_data,)
        n_line_data < n_data
    resolution : int
        Number of points located on one axis.
    height : int
        Figure height
    width : int
        Figure width
    title : str or None
        Figure title
    p : bokeh.plotting.Figure or None
        If `p` is None, it initializes new figure.
        Hovertool looks up objects of which name is `scatter`
    margin : float
        Padding size. The range of figures is
        `x_range` = (x_min - margin, x_max + margin)
        `y_range` = (y_min - margin, y_max + margin)
    point_color : str
        Color code or name of (`x`, `y`)
    line_color : str
        Color code or name of (`x`, `y_line`)
    size : int
        Size of points (`x`, `y`)

    Returns
    -------
    p : bokeh.plotting.Figure
        The overlapped figures that consists with scatter
        plot of (`x`, `y`) and line plot of (`x`, `y_line`)
    """
    if p is None:
        X = np.vstack([x, y]).T
        p = initialize_figure(height, width, title, X, margin, True)

    p.scatter(x=x, y=y, size=size, alpha=0.5, color=point_color)
    if y_line is not None:
        p.line(x=x[:y_line.shape[0]], y=y_line, line_width=1, color=line_color, alpha=0.5)

    return p

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

def net_parameter_compression_ratio(model, X):
    """
    Arguments
    ---------
    model : sklearn.neural_net.MLPRegressor
        Trained neural network model
    X : numpy.ndarray
        Data to be compute ratio between size of data and parameters.

    Returns
    -------
    times : float
        The proportion of size between trained data and model parameters

    Usage
    -----
    Train neural network model

        >>> from sklearn.neural_network import MLPRegressor

        >>> mlp = MLPRegressor(hidden_layer_sizes=(200, 5), solver='adam', activation='tanh', max_iter=1000)
        >>> mlp.fit(X, y)

    To compute efficiency of neural net

        >>> times = net_parameter_compression_ratio(mlp, X)
        >>> print(f'size of parameter / data = x {times:.4}')
    """

    times = 0
    for h in model.coefs_:
        times += h.shape[0] * h.shape[1]
    times /= (X.shape[0] * X.shape[1])
    return times
