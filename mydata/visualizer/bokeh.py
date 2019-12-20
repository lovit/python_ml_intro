import bokeh
import numpy as np

from bokeh.plotting import figure
from bokeh.palettes import Set1, turbo


def prepare_colors(labels, colors):
    """
    :param list labels: Data corresponding label list
        If None, it considers all data belong to same category
        numpy.ndarray is available
        It assumes that labels begin 0, and the number of labels is labels.max() + 1
    :param list colors: Data corresponding colors
        If None, it automatically set colors
        numpy.ndarray is available

    :returns: n_labels, unique_labels, colors
        - n_labels : Number of labels. It assumes that label index begins from 0
        - unique_labels : List of unique labels
        - colors : List of color code, length is n_data

    Usage

        >>> labels = [0, 0, 1, 1, 2, 5]
        >>> colors = None
        >>> n_labels, unique_labels, colors = prepare_colors(labels, colors)
        >>> print(n_labels)
        6
    """
    # find unique lables and check num lables
    if labels is not None:
        unique_labels = np.unique(labels)
    else:
        unique_labels = np.zeros(1, dtype=np.int)
    n_labels = unique_labels.max() + 1

    # check inserted colors
    if colors is not None:
        if isinstance(colors, str):
            colors = [colors] * n_labels
        if len(colors) < n_labels:
            raise ValueError(f'There exists {n_labels}.'\
                             ' However, the length of colors is too short')
        return n_labels, unique_labels, colors

    # prepare colors
    if n_labels <= 9:
        colors = Set1[9][:n_labels]
    elif n_labels > 256:
        raise ValueError(f'There exists {n_labels}, too many labels')
    else:
        colors = turbo(n_labels)

    return n_labels, unique_labels, colors

def prepare_plot_data(n_labels, legends, markers):
    """
    :param int n_labels: Number of labels
    :param list_or_None legends: List of legends
        If None, it will be transformed to list of None
    :param list_or_None markers: List of markers
        If None, it will be transformed to list of 'circle'

    :returns: legends, markers
        - legends : transformed list of legends
        - markers : transformed list of markers

    Usage

        >>> n_labels = 3
        >>> legends, markers = None, None
        >>> legends, markers = prepare_plot_data(n_labels, legends, markers)
        >>> print(legends)
        [None, None, None]
        >>> print(markers) # default markers
        ['circle', 'circle', 'circle']
    """
    if legends is None:
        legends = [None] * n_labels
    if markers is None:
        markers = ['circle'] * n_labels

    def length_checker(list):
        if len(list) < n_labels:
            list_instance = locals()['list']
            raise ValueError(
                f'Number of labels is {n_labels}, however too short list: {list_instance}')

    for list in [legends, markers]:
        length_checker(list)
    return legends, markers

def scatterplot(data, labels=None, colors=None, p=None, title=None,
    size=5, alpha=0.85, legends=None, markers=None,
    background_fill_color='#fefefe', grid_line_color=None):
    """
    :param numpy.ndarray data: 2D input data
        shape = (n data, 2)
    :param list labels: Data corresponding label list
        If None, it considers all data belong to same category
        numpy.ndarray is available
        It assumes that labels begin 0, and the number of labels is labels.max() + 1
    :param list colors: Data corresponding colors
        If None, it automatically set colors
        numpy.ndarray is available
    :param bokeh.plotting.figure.Figure p: Bokeh figure
        If None, it generates new figure.
        Else, it overlay scatter plots
    :param str title: Title of new generated figure
    :param int size: Size of points
    :param float alpha: Transparency of points
        Range is (0, 1]
    :param list legends: List of legends corresponding label index
    :param list markers: List of markers corresponding label index
    :param str background_fill_color: Color code for background
    :param str_or_None grid_line_color: Grid line color
        If None, it set grid line color as same with background
        Suggested combination is following

            >>> background_fill_color='#fefefe'
            >>> grid_line_color='white'

        or

            >>> background_fill_color='#050505'
            >>> grid_line_color='white'

    :returns: p
        - p : bokeh.plotting.figure.Figure

    Usage

        Generating sample data

        >>> import numpy as np
        >>> from bokeh.plotting import show
        >>> data = np.random.random_sample((10, 2))

        Plotting without label

        >>> p = scatterplot(data)
        >>> show(p)

        Plotting with label and background / grid line color setting

        >>> p = scatterplot(data, labels=0, background_fill_color='#010101', grid_line_color='white')
        >>> show(p)

        Overlay plots

        >>> data = np.random.random_sample((10, 2)) + 5
        >>> p = scatterplot(data, labels=1, p=p)
        >>> show(p)
    """

    assert 0 < alpha <= 1

    if isinstance(data, list) or isinstance(data, tuple):
        if (len(data) != 2):
            raise ValueError('"data" must be 2D vectors or tuple of two column vector')
        len_x, len_y = data[0].shape[0], data[1].shape[0]
        if len_x != len_y:
            raise ValueError(f'Two column vector have to be same length, but ({len_x, len_y})')
        data = np.vstack(data).transpose()
    n_data, dim = data.shape
    if dim != 2:
        raise ValueError(f'data must be (n_data, 2) shape, but {data.shape}')
    if isinstance(labels, int):
        labels = [labels] * n_data

    n_labels, unique_labels, colors = prepare_colors(labels, colors)
    legends, markers = prepare_plot_data(n_labels, legends, markers)
    if labels is None:
        labels = np.zeros(n_data, dtype=np.int)

    # generate new figure
    if p is None:
        title = '' if title is None else title
        p = figure(title=title, background_fill_color=background_fill_color)
        if grid_line_color is None:
            grid_line_color = background_fill_color
        p.grid.grid_line_color=grid_line_color

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        x = data[indices,0]
        y = data[indices,1]
        color = colors[label]
        legend = legends[label]
        marker = markers[label]
        if legend is None:
            p.scatter(x=x, y=y, marker=marker, size=size, line_color="white",
                fill_color=color, alpha=alpha)
        else:
            p.scatter(x=x, y=y, marker=marker, size=size, line_color="white",
                fill_color=color, alpha=alpha, legend_label=legend)

    return p

def overlay_regression_line(x, model_or_y, p, n_steps=2, margin=0.025,
    legend=None, line_dash=(4,4), line_color='orange', line_width=2):

    """
    :param numpy.ndarray x: x value or range of x
        If x stands for x-range, the length of x must be 2
        If x is instance of numpy.ndarray, the x must be column vector
    :param model_or_numpy.ndarray model_or_y: predicative model or y value
        All functions are possilble to use if it works like

            y = model(x)

        Or numpy.ndarray column vector

    :param bokeh.plotting.figure.Figure p: Figure to overlay line
    :param int n_steps: The number of points in x
        If works only when x is range
    :param float margin: x_ramge margin
    :param str legend: Line legend
    :param tuple line_dash: bokeh.core.properties.DashPattern
    :param str line_color: Color code
    :param int line_width: Width of regression line

    :returns: p
        Bokeh figure which ovelayed regerssion line

    Usgae
    -----
        Draw base scatter plot

        >>> p = scatterplot((x, y), colors='#323232')

        With dummy model

        >>> model = lambda x: 1.0 * x + 1.0
        >>> p = overlay_regression_line(x, model, p, n_steps=5, legend='test')
        >>> show(p)
    """

    if p is None:
        raise ValueError('p must be Bokeh Figure')
    if isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        elif len(x.shape) > 2:
            raise ValueError(f'x must be vector not tensor, however the shape of input is {x.shape}')
        if x.shape[1] > 1:
            raise ValueError(f'x must be 1D data, however the shape of input is {x.shape}')
        x_ = x.copy()
        sorting_indices = x_.argsort(axis=0).reshape(-1)
        x_ = x_[sorting_indices]
    elif len(x) != 2:
        raise ValueError(f'x must be numpy.ndarray column vector or range, however the length of x is {len(x)}')
    else:
        x_min, x_max = x
        x_min, x_max = x_min - margin, x_max + margin
        x_ = np.linspace(x_min, x_max, n_steps).reshape(-1,1)

    if isinstance(model_or_y, np.ndarray):
        y_pred = model_or_y.copy()
        if not isinstance(x, np.ndarray):
            raise ValueError(f'x should be numpy.ndarray when y is numpy.ndarray instance')
        y_pred = y_pred[sorting_indices]
    else:
        # (n_data, 1) -> (n_data, 1)
        y_pred = model_or_y(x_)

    # as column vector
    x_ = x_.reshape(-1)
    y_pred = y_pred.reshape(-1)

    if legend is None:
        p.line(x_, y_pred, line_dash=line_dash, line_color=line_color, line_width=line_width)
    else:
        p.line(x_, y_pred, line_dash=line_dash, line_color=line_color, line_width=line_width, legend_label=legend)
    return p
