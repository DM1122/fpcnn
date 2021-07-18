"""Plotting utilities."""

# stdlib
import os

# external
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import spectral
from skopt import plots as skplot

matplotlib.use("TkAgg")


def plot_series(
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
    """Plot series.

    Args:
        df (pandas.core.DataFrame): Pandas dataframe
        title (str): Plot title
        traces (list[str]): Column names of traces to graph
        indicies (list[str], optional): Column names of indicies for trace to use.
            Defaults to None.
        x_errors (str, optional): Column name of x-errors. Defaults to None.
        y_errors ([type], optional): Column name of y-errors. Defaults to None.
        title_x (str, optional): X-axis title. Defaults to "x".
        title_y (str, optional): Y-axis title. Defaults to "y".
        size (tuple, optional): Width and height of plot in pixels. Defaults to None.
        draw_mode (str, optional): Whether to render lines or markers or both.
            Defaults to "lines+markers".
        vlines (list[int], optional): List of ordinates for vertical lines.
            Defaults to None.
        hlines (list[int], optional): List of ordinates for horizontal lines.
            Defaults to None.
        dark_mode (bool, optional): Dark mode. Defaults to False.
    """
    fig = go.Figure()

    template = "plotly" if not dark_mode else "plotly_dark"

    # draw traces
    for i, trace in enumerate(traces):
        name = trace
        y = df[trace]
        index = (
            df[indicies[i]] if indicies is not None else df.index
        )  # cannot have one trace use df.index and others not. WIP
        x_error = df[x_errors[i]] if x_errors is not None else None
        y_error = df[y_errors[i]] if y_errors is not None else None

        # error bars
        x_error = {
            "type": "data",
            "symmetric": True,
            "array": x_error,
            # "color":"black",
            # "thickness":1.5,
            # "width":3
        }

        y_error = {
            "type": "data",
            "symmetric": True,
            "array": y_error,
            # "color":"black",
            # "thickness":1.5,
            # "width":3
        }

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
                line={"color": "black", "width": 2, "dash": "dash"},
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
                line={"color": "black", "width": 2, "dash": "dash"},
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


def plot_dist(
    df,
    title,
    traces,
    nbins=None,
    x_errors=None,
    y_errors=None,
    title_x="x",
    title_y="y",
    size=None,
    vlines=None,
    hlines=None,
    dark_mode=False,
):
    """Plot distribution.

    Args:
        df (pandas.core.DataFrame): Pandas dataframe.
        title (str): Plot title.
        traces (list[str]): Column names of traces to graph.
        nbins (int): Number of bins for histogram along x-axis.
        x_errors (str, optional): Column name of x-errors. Defaults to None.
        y_errors ([type], optional): Column name of y-errors. Defaults to None.
        title_x (str, optional): X-axis title. Defaults to "x".
        title_y (str, optional): Y-axis title. Defaults to "y".
        size (tuple, optional): Width and height of plot in pixels. Defaults to None.
        draw_mode (str, optional): Whether to render lines or markers or both.
            Defaults to "lines+markers".
        vlines (list[int], optional): List of ordinates for vertical lines.
            Defaults to None.
        hlines (list[int], optional): List of ordinates for horizontal lines.
            Defaults to None.
        dark_mode (bool, optional): Dark mode. Defaults to False.
    """
    fig = go.Figure()
    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=False,
        start_cell="top-left",
        print_grid=False,
        horizontal_spacing=None,
        vertical_spacing=0.025,
        subplot_titles=None,
        column_widths=None,
        row_heights=[0.2, 0.8],
        specs=None,
        insets=None,
        column_titles=None,
        row_titles=None,
        x_title=title_x,
        y_title=title_y,
        figure=fig,
    )

    template = "plotly" if not dark_mode else "plotly_dark"
    opacity = 1.0 if len(traces) == 1 else 0.75

    # draw traces
    for i, trace in enumerate(traces):
        name = trace
        x = df[trace]
        x_error = df[x_errors[i]] if x_errors is not None else None
        y_error = df[y_errors[i]] if y_errors is not None else None

        # error bars
        x_error = {
            "type": "data",
            "symmetric": True,
            "array": x_error,
            # "color":"black",
            # "thickness":1.5,
            # "width":3
        }

        y_error = {
            "type": "data",
            "symmetric": True,
            "array": y_error,
            # "color":"black",
            # "thickness":1.5,
            # "width":3
        }

        fig.add_trace(
            go.Histogram(
                arg=None,
                alignmentgroup=None,
                bingroup=0,
                cumulative=None,
                customdata=None,
                customdatasrc=None,
                error_x=x_error,
                error_y=y_error,
                histfunc="count",
                histnorm=None,
                hoverinfo=None,
                hoverinfosrc=None,
                hoverlabel=None,
                hovertemplate=None,
                hovertemplatesrc=None,
                hovertext=None,
                hovertextsrc=None,
                ids=None,
                idssrc=None,
                legendgroup=i,
                marker=None,
                marker_color=px.colors.qualitative.Plotly[i],
                meta=None,
                metasrc=None,
                name=name,
                nbinsx=nbins,
                nbinsy=None,
                offsetgroup=None,
                opacity=opacity,
                orientation="v",
                selected=None,
                selectedpoints=None,
                showlegend=True,
                stream=None,
                text=None,
                textsrc=None,
                uid=None,
                uirevision=None,
                unselected=None,
                visible=None,
                x=x,
                xaxis=None,
                xbins=None,
                xcalendar=None,
                xsrc=None,
                y=None,
                yaxis=None,
                ybins=None,
                ycalendar=None,
                ysrc=None,
            ),
            row=2,
            col=1,
        )

        # draw marginals
        fig.add_trace(
            go.Box(
                alignmentgroup=None,
                boxmean=None,
                boxpoints="all",
                customdata=None,
                customdatasrc=None,
                dx=None,
                dy=None,
                fillcolor=None,
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
                jitter=0,
                legendgroup=i,
                line=None,
                line_color=px.colors.qualitative.Plotly[i],
                lowerfence=None,
                lowerfencesrc=None,
                marker=None,
                marker_symbol="line-ns-open",
                mean=None,
                meansrc=None,
                median=None,
                mediansrc=None,
                meta=None,
                metasrc=None,
                name=name,
                notched=True,
                notchspan=None,
                notchspansrc=None,
                notchwidth=None,
                offsetgroup=None,
                opacity=None,
                orientation=None,
                pointpos=None,
                q1=None,
                q1src=None,
                q3=None,
                q3src=None,
                quartilemethod=None,
                sd=None,
                sdsrc=None,
                selected=None,
                selectedpoints=None,
                showlegend=False,
                stream=None,
                text=None,
                textsrc=None,
                uid=None,
                uirevision=None,
                unselected=None,
                upperfence=None,
                upperfencesrc=None,
                visible=None,
                whiskerwidth=None,
                width=None,
                x=x,
                x0=None,
                xaxis=None,
                xcalendar=None,
                xperiod=None,
                xperiod0=None,
                xperiodalignment=None,
                xsrc=None,
                y=None,
                y0=None,
                yaxis=None,
                ycalendar=None,
                yperiod=None,
                yperiod0=None,
                yperiodalignment=None,
                ysrc=None,
            ),
            row=1,
            col=1,
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
                line={"color": "black", "width": 2, "dash": "dash"},
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
                line={"color": "black", "width": 2, "dash": "dash"},
            )

    # config plot
    w = size[0] if size is not None else None
    h = size[1] if size is not None else None

    fig.update_layout(
        title=title,
        width=w,
        height=h,
        # xaxis_title=title_x,
        # yaxis_title=title_y,
        template=template,
        barmode="overlay",
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

    # x=x,
    # marker_symbol="line-ns-open",
    # marker_color=None,
    # boxpoints="all",
    # jitter=0,
    # fillcolor=px.colors.qualitative.Plotly[i],
    # line_color=px.colors.qualitative.Plotly[i],
    # hoveron="points",
    # legendgroup=i,
    # showlegend=False,
    # name=name,


# region skopt
def plot_skopt_convergence(opt_res, path):
    """Plots the convergence plot from the skopt package.

    Args:
        opt_res (scipy.optimize.OptimizeResult): Optimization result object.
        path (str): Directory at which to save plot.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    fig = plt.figure()
    skplot.plot_convergence(opt_res, ax=None, true_minumum=None, yscale=None)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path + "convergence")
    fig.show()


def plot_skopt_evaluations(opt_res, path):
    """Plots the evaluations plot from the skopt package.

    Args:
        opt_res (scipy.optimize.OptimizeResult): Optimization result object.
        path (str): Directory at which to save plot.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    skplot.plot_evaluations(result=opt_res, bins=32, dimensions=None, plot_dims=None)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path + "evaluations")
    fig.show()


def plot_skopt_objective(opt_res, path):
    """Plots the objective plot from the skopt package.

    Args:
        opt_res (scipy.optimize.OptimizeResult): Optimization result object.
        path (str): Directory at which to save plot.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    skplot.plot_objective(
        result=opt_res,
        levels=64,
        n_points=64,
        n_samples=256,
        size=2,
        zscale="linear",
        dimensions=None,
        sample_source="random",
        minimum="result",
        n_minimum_search=None,
        plot_dims=None,
        show_points=True,
        cmap="viridis_r",
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path + "objective")
    fig.show()


def plot_skopt_regret(opt_res, path):
    """Plots the regret plot from the skopt package.

    Args:
        opt_res (scipy.optimize.OptimizeResult): Optimization result object.
        path (str): Directory at which to save plot.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    fig = plt.figure()
    skplot.plot_regret(opt_res, ax=None, true_minumum=None, yscale=None)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path + "regret")
    fig.show()


# endregion


def plot_context(offsets, shape):
    """Plots context selection for visualization.

    Args:
        offsets (list): List of offsets to visualize.
        shape (tuple): Shape of context window.
    """
    assert shape[0] & 1 and shape[1] & 1, f"Shape is not odd {shape}"

    matrix = np.zeros(
        shape=(shape[1], shape[0]), dtype=np.uint8, order="C"
    )  # matplotlib.matshow() defines their matricies as (y, x) (i think..?)

    center = (shape[0] // 2, shape[1] // 2)
    matrix[center[1]][center[0]] = 1
    for offset in offsets:
        matrix[center[1] + offset[1]][center[0] + offset[0]] = 2

    plt.matshow(A=matrix, fignum=None)
    plt.grid(b=False, which="major", axis="both")
    plt.show()
