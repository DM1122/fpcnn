"""Plotlib testing."""
# stdlib
import math

# external
import numpy as np
import pandas as pd
import pytest

# project
from fpcnn.libs import plotlib, printlib


@pytest.mark.plot
def test_plot_series():

    x = np.linspace(
        start=-math.pi,
        stop=math.pi,
        num=64,
        endpoint=True,
        retstep=False,
        dtype=None,
        axis=0,
    )
    y = np.sin(x)

    df = pd.DataFrame(data=y, index=x, columns=["data"], dtype=None, copy=False)
    plotlib.plot_series(
        df=df,
        title="Test",
        traces=["data"],
        indicies=None,
        x_errors=None,
        y_errors=None,
        title_x="x",
        title_y="y",
        size=None,
        draw_mode="lines+markers",
        vlines=None,
        hlines=None,
        dark_mode=True,
    )


@pytest.mark.plot
def test_plot_dist():

    x1 = np.random.normal(loc=0.0, scale=1.0, size=128)
    x2 = np.random.normal(loc=1, scale=0.75, size=128)

    data = np.stack((x1, x2), axis=1, out=None)
    printlib.print_ndarray_stats(array=data, title="Data")

    df = pd.DataFrame(
        data=data, index=None, columns=["x1", "x2"], dtype=None, copy=False
    )
    plotlib.plot_dist(
        df=df,
        title="Test",
        traces=["x1", "x2"],
        nbins=50,
        x_errors=None,
        y_errors=None,
        title_x="x",
        title_y="y",
        size=None,
        vlines=None,
        hlines=None,
        dark_mode=True,
    )
