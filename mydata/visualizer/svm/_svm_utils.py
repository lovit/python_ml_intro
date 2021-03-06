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
