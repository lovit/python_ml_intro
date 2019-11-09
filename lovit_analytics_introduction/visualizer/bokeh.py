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
