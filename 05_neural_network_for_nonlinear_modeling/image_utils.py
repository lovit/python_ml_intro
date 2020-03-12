import math
import numpy as np
from bokeh.plotting import show, figure
from bokeh.layouts import gridplot


def show_image_grid(images, labels=None, n_cols=4, rot90=0):

    if labels is None:
        labels = ['Unknown'] * len(images)

    figs = []

    for img, l in zip(images, labels):
        p = figure(width=150, height=150, title=f'Label {l}')
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
        img = np.rot90(img, k=rot90)
        p.image(image=[img], x=0, dw=1, y=0, dh=1)
        figs.append(p)

    # 소수점 올림
    n_rows = math.ceil(len(figs) / n_cols)
    rows = []
    for i in range(n_rows):
        b = i * n_cols
        e = (i+1) * n_cols
        rows.append(figs[b:e])

    # input = list of list
    gp = gridplot(rows)
    show(gp)

    return rows