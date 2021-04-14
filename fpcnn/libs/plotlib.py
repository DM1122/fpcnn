"""Plotting utilities."""

# external
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spectral


def plot_traces(
    df,
    title,
    traces,
    indicies=None,
    x_errors=None,
    y_errors=None,
    title_x="x",
    title_y="y",
    size=None,
    draw_mode="lines+markers",
    vlines=None,
    hlines=None,
    dark_mode=False,
):
    """Plots traces.

    Args:
        df (pandas.core.DataFrame): Pandas dataframe
        title (str): Plot title
        traces (list[str]): Column names of traces to graph
        indicies (list[str], optional): Column names of indicies for trace to use. Defaults to None.
        x_errors (str, optional): Column name of x-errors. Defaults to None.
        y_errors ([type], optional): Column name of y-errors. Defaults to None.
        title_x (str, optional): X-axis title. Defaults to "x".
        title_y (str, optional): Y-axis title. Defaults to "y".
        size (tuple, optional): Width and height of plot in pixels. Defaults to None.
        draw_mode (str, optional): Whether to render lines or markers or both. Defaults to "lines+markers".
        vlines (list[int], optional): List of ordinates for vertical lines. Defaults to None.
        hlines (list[int], optional): List of ordinates for horizontal lines. Defaults to None.
        dark_mode (bool, optional): Dark mode. Defaults to False.
    """
    fig = go.Figure()

    template = "plotly" if not dark_mode else "plotly_dark"

    assert (
        type(df) == pd.core.frame.DataFrame
    ), f"Array is not pandas dataframe (df:{type(df)})"

    # draw traces
    for i in range(len(traces)):
        name = traces[i]
        y = df[traces[i]]
        index = (
            df[indicies[i]] if indicies is not None else df.index
        )  # cannot have one trace use df.index and others not. WIP
        x_error = df[x_errors[i]] if x_errors is not None else None
        y_error = df[y_errors[i]] if y_errors is not None else None

        # error bars
        x_error = dict(
            type="data",
            symmetric=True,
            array=x_error,
            # color="black",
            # thickness=1.5,
            # width=3
        )

        y_error = dict(
            type="data",
            symmetric=True,
            array=y_error,
            # color="black",
            # thickness=1.5,
            # width=3
        )

        fig.add_trace(
            go.Scatter(
                arg=None,
                cliponaxis=None,
                connectgaps=False,
                customdata=None,
                customdatasrc=None,
                dx=None,
                dy=None,
                error_x=x_error,
                error_y=y_error,
                fill=None,
                fillcolor=None,
                groupnorm=None,
                hoverinfo=None,
                hoverinfosrc=None,
                hoverlabel=None,
                hoveron=None,
                hovertemplate=None,
                hovertemplatesrc=None,
                hovertext=None,
                hovertextsrc=None,
                ids=None,
                idssrc=None,
                legendgroup=None,
                line=None,
                marker=None,
                meta=None,
                metasrc=None,
                mode=draw_mode,
                name=name,
                opacity=None,
                orientation=None,
                r=None,
                rsrc=None,
                selected=None,
                selectedpoints=None,
                showlegend=True,
                stackgaps=None,
                stackgroup=None,
                stream=None,
                t=None,
                text=None,
                textfont=None,
                textposition=None,
                textpositionsrc=None,
                textsrc=None,
                texttemplate=None,
                texttemplatesrc=None,
                tsrc=None,
                uid=None,
                uirevision=None,
                unselected=None,
                visible=None,
                x=index,
                x0=None,
                xaxis=None,
                xcalendar=None,
                xsrc=None,
                y=y,
                y0=None,
                yaxis=None,
                ycalendar=None,
                ysrc=None,
            )
        )

    # draw vertical lines
    if vlines is not None:
        for vline in vlines:
            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=vline,
                y0=0,
                x1=vline,
                y1=1,
                line=dict(color="black", width=2, dash="dash"),
            )

    # draw horizontal lines
    if hlines is not None:
        for hline in hlines:
            fig.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                y0=hline,
                x1=1,
                y1=hline,
                line=dict(color="black", width=2, dash="dash"),
            )

    # config plot
    w = size[0] if size is not None else None
    h = size[1] if size is not None else None

    fig.update_layout(
        title=title,
        width=w,
        height=h,
        xaxis_title=title_x,
        yaxis_title=title_y,
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
