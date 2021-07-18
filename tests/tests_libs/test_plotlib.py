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
    """Test for series plot."""

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
    """Test for distribution plot."""
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


@pytest.mark.plot
def test_plot_context():
    """Test for context plot."""
    offsets = [
        # first shell
        (-1, 0, 0),
        (-1, -1, 0),
        (0, -1, 0),
        (1, -1, 0),
        # second shell
        (-2, 0, 0),
        (-2, -1, 0),
        (-2, -2, 0),
        (-1, -2, 0),
        (0, -2, 0),
        (1, -2, 0),
        (2, -2, 0),
        (2, -1, 0),
        # third shell
        (-3, 0, 0),
        (-3, -1, 0),
        (-3, -2, 0),
        (-3, -3, 0),
        (-2, -3, 0),
        (-1, -3, 0),
        (0, -3, 0),
        (1, -3, 0),
        (2, -3, 0),
        (3, -3, 0),
        (3, -2, 0),
        (3, -1, 0),
        # fourth shell
        (-4, 0, 0),
        (-4, -1, 0),
        (-4, -2, 0),
        (-4, -3, 0),
        (-4, -4, 0),
        (-3, -4, 0),
        (-2, -4, 0),
        (-1, -4, 0),
        (0, -4, 0),
        (1, -4, 0),
        (2, -4, 0),
        (3, -4, 0),
        (4, -4, 0),
        (4, -3, 0),
        (4, -2, 0),
        (4, -1, 0),
    ]

    plotlib.plot_context(offsets=offsets, shape=(11, 11))
