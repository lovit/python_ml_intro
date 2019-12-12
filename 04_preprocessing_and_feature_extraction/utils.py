from bokeh.palettes import Set1
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
import numpy as np

marker_list = 'circle square triangle asterisk circle_x square_x '\
    'inverted_triangle x circle_cross square_cross diamond cross'.split()

def scatterplot_iris(iris_df, labels=None, title=None, height=400, width=400,
    x='petal_length', y='petal_width', alpha=1.0, show_inline=True):

    # prepare x, y, labels
    if labels is None:
        labels = iris_df['target']
    if isinstance(x, str):
        x = iris_df[x]
        y = iris_df[y]
    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha = [alpha] * len(iris_df)

    data = iris_df.copy()
    data['label'] = labels
    data['x'] = x
    data['y'] = y

    # set color & marker
    n_labels = np.unique(labels).shape[0]
    palette = Set1[max(3, n_labels)]
    data['color'] = [palette[l] if l >= 0 else 'lightgrey' for l in labels]
    data['marker'] = [marker_list[t] for t in iris_df['target']]
    data['alpha'] = alpha

    # as ColumnDataSource
    source = ColumnDataSource(data)

    # prepare hover tool
    tooltips = [
        ('index', '$index'),
        ('(sepal length, sepal width)', '(@sepal_length, @sepal_width)'),
        ('(petal length, petal width)', '(@petal_length, @petal_width)'),
        ('cluster label', '@label'),
        ('true label', '@target')
    ]

    # draw figure
    p = figure(title=title, height=height, width=width, tooltips=tooltips)
    # p.background_fill_color='#202020'
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.scatter(x='x', y='y', size=7, line_color='white', alpha='alpha',
        color='color', marker='marker', source=source, legend_field='label')

    if show_inline:
        show(p)
    return p
