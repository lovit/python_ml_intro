import ipywidgets as widgets

from bokeh.plotting import show
from ipywidgets import interact


def bokeh_image_slider(images):
    """
    It shows interactive IPython notebook cell of image slider

    :param list images: List of Bokeh.plotting.Figure
        Bokeh figure list
    """

    def index_slider(index):
        image = images[index]
        show(image)

    slider = widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0)
    interact(index_slider, index=slider)
