"""Plotting utilities."""

# external
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spectral


def plot_datacube(data, band=0, title="Datacube"):
    """Plot datacube band.

    Args:
        data (ndarray): 3D data array
        band (int): band selection
        title (str): plot title
    """
    spectral.imshow(
        data=data,
        bands=(band,),
        classes=None,
        source=None,
        colors=None,
        figsize=None,
        fignum=None,
        title=title,
    )


def plot_traces(traces, names=["trace"], title="Traces", xtitle="x", ytitle="y", dark=False):
    """Plot one or more sequences.

    Args:
        traces (list{ndarray}): List of ndarrays containing sequences
        names (list{str}): List of strings with trace names
        title (str, optional): Plot title. Defaults to "Traces".
        xtitle (str, optional): X-axis title. Defaults to "x".
        ytitle (str, optional): Y-axis title Defaults to "y".
        dark (bool, optional): Dark mode. Defaults to False.
    """
    fig = go.Figure()

    for i in range(len(traces)):
        trace = traces[i].flatten() if traces[i].ndim != 1 else traces[i]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(trace))), y=trace, mode="lines+markers", name=names[i]
            )
        )

    template = "plotly" if not dark else "plotly_dark"
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template=template,
    )
    fig.show()


def plot_distribution(data, title="Distribution", dark=False):
    """Plot numerical distribution of array data.

    Args:
        data (ndarray): data
        title (str, optional): Plot title. Defaults to "Distribution".
        dark (bool, optional): Plot dark mode. Defaults to False.
    """
    assert type(data) == np.ndarray, f"Array is not numpy array, (data:{type(data)})"

    data_seq = data.flatten() if data.ndim != 1 else data

    template = "plotly" if not dark else "plotly_dark"
    df = pd.DataFrame(
        data=data_seq, index=None, columns=["data"], dtype=None, copy=False
    )
    fig = px.histogram(
        df,
        x="data",
        marginal="violin",
        hover_data=df.columns,
        nbins=100,
        title=title,
        template=template,
    )
    fig.show()

def plot_band(data, band, title):
    """Draw a specific band of a datcube.

    Args:
        data (ndarray): datacube
        band (int): band selection
        title (str): title
    """
    spectral.imshow(data=data, bands=(band,), classes=None, source=None, colors=None, figsize=None, fignum=None, title=title)
