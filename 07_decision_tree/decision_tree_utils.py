import numpy as np
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.palettes import PuBu9, OrRd9
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool, BoxZoomTool, WheelZoomTool, ResetTool, SaveTool, PanTool


def initialize_figure(height, width, title, X, margin, score):
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
    score : Boolean or None
        If not None or True, hover tool shows prediction score

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
    ]

    if (score is not None) or (score == True):
        tooltips.append(('prediction score', '@score'))

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

def scatterplot_2class(X, label, score=None, size=5,
    height=600, width=600, title=None, p=None, margin=0.05, colormap=None):

    """
    Arguments
    ---------
    X : numpy.ndarray
        Data to be plotted. It will be used to set `x_range` and `y_range`
        Shape of X = (n_data, 2)
    label : numpy.ndarray
        Shape of label = (n_data,)
    score : numpy.ndarray or None
        The prediction score of each point
        Shape of score = (n_data,)
    size : int
        The size of points
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
    colormap : list of str or None
        Predefined colormap

    Returns
    -------
    p : bokeh.plotting.Figure
        Figure of scatterplot
    """

    if p is None:
        p = initialize_figure(height, width, title, X, margin, score)

    # prepare alpha, color & size
    alpha = [0.9 for l in label]
    if colormap is None:
        colormap = ['#2b83ba', '#d7191c']
    color = [colormap[l] for l in label]
    size = [size for l in label]

    # prepare source
    x, y = X[:,0], X[:,1]
    data = {'x': x, 'y': y, 'label': label, 'color': color, 'alpha': alpha, 'size':size}
    if (score is not None) and (hasattr(score, '__len__')):
        data['score'] = score
    source = ColumnDataSource(data)

    # plotting
    p.scatter(x='x', y='y', fill_color='color', line_color=None,
              fill_alpha='alpha', size='size', source=source, name='scatter')

    return p

def draw_activate_image(model, X, score_type='prediction', resolution=100, height=600, width=600,
    title=None, p=None, margin=0.05, alpha=0.2, use_score=False):
    
    """
    Arguments
    ---------
    model : sklearn classifier
        Trained classifiers
    X : numpy.ndarray
        Data to be plotted. It will be used to set `x_range` and `y_range`
        Shape of X = (n_data, 2)
    score : str
        Score type.

            'prediction': Prediction score, ranged in [-1, 1]
            'var': p(1-p)
        
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
    use_score : Boolean, None or numpy.ndarray
        Predefined score. The range is [-1, 1] and the color of value 0 is 'white'
        It is used to initialize figure. 
        If True, the hovertool shows prediction score.

    Returns
    -------
    p : bokeh.plotting.Figure
        Class activation map appended figure.
        If the predicted value stands for class `0`, the region fills with blue
        If the predicted value stands for class `1`, the region fills with red
    """

    if p is None:
        p = initialize_figure(height, width, title, X, margin, use_score)

    # make grid
    (x_min, x_max), (y_min, y_max) = check_range(X, margin)
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    X_test = np.array([(x, y) for y in ys for x in xs])

    # predict
    # label in {0, 1}
    prob = model.predict_proba(X_test)
    if score_type =='prediction':
        score = prob[:,1] - prob[:,0]
        score[np.where(score < -1)[0]] = -1
        score[np.where(score > 1)[0]] = 1
        score = 255 / 2 * (score + 1)
    elif score_type == 'var':
        max_score = prob.max(axis=1)
        score = 255 * max_score * (1 - max_score) * 4
    else:
        raise ValueError('score_type must be one of ["prediction", "var"]')

    # as image
    score = np.array(score, dtype=np.uint8)
    img = score.reshape(resolution, -1)

    palette = PuBu9[:2] + PuBu9[-3:]
    palette += reversed(OrRd9[:2] + OrRd9[-3:])

    p.image(image=[img], x=x_min, y=y_min, dw=x_max-x_min, dh=y_max-y_min,
            palette=palette, alpha=alpha)

    return p