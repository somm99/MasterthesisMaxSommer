"""
Plotly figures for the interactive parts of the Streamlit frontend.

Every function here is pure: it takes the artefacts produced by
`core_tangles.build_tangle_model` (embedding, scores, feature effects, ...) and
returns a `plotly.graph_objects.Figure`. Nothing is displayed or cached; the
caller (`app.py`) hands the figure to `st.plotly_chart`.

All functions degrade gracefully on empty or missing input by returning an empty
figure instead of raising, so the UI can render before a selection is made.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def fig_tsne_membership(embedding: np.ndarray, scores: np.ndarray, customdata, membership: np.ndarray, tangle_index, hover_columns: list[str] | None = None, ) -> go.Figure:
    """
    Scatter plot of the embedding for a single tangle, in one of two modes.

    Without a `membership` mask the points are colored continuously by their
    normalized tangle score (Viridis, fixed 0..1 scale). With a mask, the plot
    switches to a binary view: members are gold, everything else light grey.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding (e.g. t-SNE), shape (n_points, 2).
    scores : np.ndarray
        Normalized scores of all tangles, shape (n_points, n_tangles).
    customdata : np.ndarray | None
        Per-point hover payload from `core_tangles.build_customdata`, or None.
    membership : np.ndarray | None
        Boolean mask selecting the points inside the tangle. None selects the
        continuous score view.
    tangle_index : int
        Column of `scores` to visualize.
    hover_columns : list[str] | None
        Column names matching the columns of `customdata`, used as hover labels.

    Returns
    -------
    go.Figure
        The scatter figure.
    """

    if membership is None:
        title = "Tangle Score"
        color_values = scores[:, tangle_index]

        fig = go.Figure(
            data=go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color=color_values,
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    colorbar=dict(title=f"Tangle {tangle_index} score"),
                    showscale=True,
                ),
                customdata=customdata,
            )
        )

    else:
        title = "Tangle Membership"
        colors = np.where(membership, "gold", "lightgray")
        fig = go.Figure(
            data=go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color=colors,
                    line=dict(width=0),
                ),
                customdata=customdata,
            )
        )

    # hover template: one line per requested column, plus the raw score
    if customdata is not None:
        lines = [
            f"{col}: %{{customdata[{i}]}}"
            for i, col in enumerate(hover_columns)
        ]
        lines += ["<br>"]
        lines += [f"score[t={tangle_index}]: %{{marker.color:.3f}}"]

        hovertemplate = (
                "<br>".join(lines)
                + "<extra></extra>"
        )
    else:
        lines = ["<br>"]
        lines += [f"score[t={tangle_index}]: %{{marker.color:.3f}}"]
        hovertemplate = (
                "<br>".join(lines)
                + "<br>"
                + "x %{x:.3f}<br>y: %{y:.3f}"
                + "<extra></extra>"
        )

    fig.update_traces(hovertemplate=hovertemplate)

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        margin=dict(l=40, r=40, t=40, b=40),
        title=title,
    )

    return fig

def fig_tangle_feature_heatmap(effects: pd.DataFrame, max_features: int = 20,) -> go.Figure:
    """
    Heatmap of the per-tangle feature effects (tangles x features).

    Only the `max_features` most discriminative features are shown, ranked by
    their variance across tangles: features that look the same in every tangle
    carry no information for telling tangles apart. The diverging colorscale is
    centered at 0, i.e. at the global feature mean.

    Parameters
    ----------
    effects : pd.DataFrame
        z-scores from `core_tangles.compute_tangle_feature_effects`,
        shape (n_tangles, n_features).
    max_features : int
        Upper bound on the number of feature columns to display.

    Returns
    -------
    go.Figure
        The heatmap, or an empty figure if no effects are available.
    """

    if effects is None or effects.empty:
        return go.Figure()

    # pick the most informative features: highest variance across tangles
    var = effects.var(axis=0)
    top_features = var.sort_values(ascending=False).head(max_features).index
    data = effects[top_features]

    fig = go.Figure(
        data=go.Heatmap(
            z=data.values,
            x=list(data.columns),
            y=list(data.index),
            colorscale="RdBu_r",
            zmid=0.0,  # 0 = global feature mean
            colorbar=dict(title="z-score vs global"),
        )
    )
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Tangle",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

def fig_parallel_coordinates(df_numeric: pd.DataFrame, membership: np.ndarray, selected_features: list[str], score_values: np.ndarray | None = None,) -> go.Figure:
    """
    Parallel-coordinates plot of the points inside one tangle.

    Each selected feature becomes one axis, scaled to the min/max of the
    selected subset, and every member point becomes one polyline. Lines are
    colored by tangle score when scores are supplied. Only numeric features are
    supported.

    Parameters
    ----------
    df_numeric : pd.DataFrame
        Numeric subset of the data, row-aligned with `membership`.
    membership : np.ndarray
        Boolean mask selecting the points to draw.
    selected_features : list[str]
        Feature columns to use as axes, in the given order.
    score_values : np.ndarray | None
        Scores of the active tangle for all points (not just members); used to
        color the lines. None colors every line uniformly.

    Returns
    -------
    go.Figure
        The parallel-coordinates figure, or an empty figure if no features are
        selected or no point matches `membership`.
    """

    if not selected_features:
        return go.Figure()

    df_plot = df_numeric.loc[membership, selected_features].copy()

    if df_plot.empty:
        return go.Figure()

    dimensions = []
    for col in selected_features:
        col_values = df_plot[col].to_numpy(dtype=float)

        dimensions.append(
            dict(
                range=[float(np.nanmin(col_values)), float(np.nanmax(col_values))],
                label=col,
                values=col_values,
            )
        )

    if score_values is not None:
        color_values = score_values[membership]
    else:
        color_values = np.ones(len(df_plot))

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=color_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="score"),
            ),
            dimensions=dimensions,
        )
    )

    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

def fig_tsne_compare_tangles(
    embedding: np.ndarray,
    scores_all: np.ndarray,
    selected_tangles: list[int],
    threshold: float = 0.3,
    customdata=None,
    hover_columns: list[str] | None = None,
    n_cols: int = 2,
) -> go.Figure:
    """
    Small-multiples comparison of several tangles in the same embedding.

    Every subplot shows the identical point cloud but highlights only the points
    of one tangle (score >= `threshold`); non-members stay faded. All subplots
    share the same padded axis ranges so the panels are visually comparable.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding, shape (n_points, 2).
    scores_all : np.ndarray
        Normalized scores of all tangles, shape (n_points, n_tangles).
    selected_tangles : list[int]
        Column indices of `scores_all` to show, one subplot each.
    threshold : float
        Minimum score for a point to count as a member of its tangle.
    customdata : np.ndarray | None
        Per-point hover payload, or None.
    hover_columns : list[str] | None
        Column names matching the columns of `customdata`.
    n_cols : int
        Number of subplot columns; the row count follows from it.

    Returns
    -------
    go.Figure
        The subplot grid, or an empty figure if no tangle is selected.
    """
    if not selected_tangles:
        return go.Figure()

    n = len(selected_tangles)
    n_rows = int(np.ceil(n / n_cols))

    subplot_titles = [f"Tangle {t}" for t in selected_tangles]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    x_all = embedding[:, 0]
    y_all = embedding[:, 1]

    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())

    dx = x_max - x_min
    dy = y_max - y_min
    pad_x = 0.05 * dx if dx > 0 else 1.0
    pad_y = 0.05 * dy if dy > 0 else 1.0

    x_range = [x_min - pad_x, x_max + pad_x]
    y_range = [y_min - pad_y, y_max + pad_y]

    for i, t_idx in enumerate(selected_tangles):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # score and membership of this tangle
        score_values = scores_all[:, t_idx]
        membership = score_values >= threshold

        colors = np.where(membership, "gold", "rgba(180,180,180,0.25)")
        sizes = np.where(membership, 8, 5)

        if customdata is not None and hover_columns is not None:
            lines = [
                f"{col_name}: %{{customdata[{j}]}}"
                for j, col_name in enumerate(hover_columns)
            ]
            hovertemplate = (
                f"tangle: {t_idx}<br>"
                + "<br>".join(lines) + "<br>"
                + "<br>score: %{text:.3f}<extra></extra>"
            )
        else:
            hovertemplate = (
                f"tangle: {t_idx}"
                "<br>score: %{text:.3f}"
                "<br>"
                "<br>x: %{x:.3f}, y: %{y:.3f}<extra></extra>"
            )

        fig.add_trace(
            go.Scatter(
                x=x_all,
                y=y_all,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(width=0),
                ),
                customdata=customdata,
                text=score_values,
                hovertemplate=hovertemplate,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(range=x_range, row=row, col=col)
        fig.update_yaxes(range=y_range, row=row, col=col)

    fig.update_layout(
        height=max(350 * n_rows, 400),
        margin=dict(l=40, r=40, t=60, b=40),
        title="TSNE comparison of multiple tangles",
    )
    return fig

def fig_agreement_scan_comparison(results_long: pd.DataFrame) -> go.Figure:
    """
    Line plot comparing several agreement scans: agreement vs. number of tangles.

    One line per saved run, so different order functions or separation settings
    can be compared over the same agreement range.

    Parameters
    ----------
    results_long : pd.DataFrame
        Long-format table with the columns `agreement`, `n_tangles` and
        `run_label` (one label per saved scan).

    Returns
    -------
    go.Figure
        The comparison plot, or an empty figure if no scan has been saved.
    """
    if results_long is None or results_long.empty:
        return go.Figure()

    fig = px.line(
        results_long,
        x="agreement",
        y="n_tangles",
        color="run_label",
        markers=True,
    )

    fig.update_traces(
        hovertemplate=(
            "run: %{fullData.name}<br>"
            "agreement: %{x}<br>"
            "maximal tangles: %{y}<extra></extra>"
        )
    )

    fig.update_layout(
        title="Agreement scan comparison",
        xaxis_title="Agreement parameter",
        yaxis_title="Number of maximal tangles",
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title="Saved runs",
    )
    return fig

def fig_tsne_rule_combination(
    embedding: np.ndarray,
    membership: np.ndarray,
    customdata,
    hover_columns: list[str] | None,
    title: str,
) -> go.Figure:
    """
    Embedding plot highlighting the intersection of a set of selected rules.

    The mask is computed by the caller (`app.py` intersects the corresponding
    columns of S); this function only renders it, drawing the points that
    satisfy every selected rule in gold and the rest faded.

    Parameters
    ----------
    embedding : np.ndarray
        2D embedding, shape (n_points, 2).
    membership : np.ndarray
        Boolean mask of the points satisfying all selected rules.
    customdata : np.ndarray | None
        Per-point hover payload, or None.
    hover_columns : list[str] | None
        Column names matching the columns of `customdata`.
    title : str
        Figure title, typically naming the active tangle.

    Returns
    -------
    go.Figure
        The scatter figure, or an empty figure if embedding or mask is missing.
    """
    if embedding is None or membership is None:
        return go.Figure()

    x = embedding[:, 0]
    y = embedding[:, 1]

    colors = np.where(membership, "gold", "rgba(180,180,180,0.25)")
    sizes = np.where(membership, 9, 5)

    if customdata is not None and hover_columns is not None:
        lines = [
            f"{col_name}: %{{customdata[{j}]}}"
            for j, col_name in enumerate(hover_columns)
        ]
        hovertemplate = "<br>".join(lines) + "<extra></extra>"
    else:
        hovertemplate = "x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"

    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=0),
            ),
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        margin=dict(l=40, r=40, t=60, b=40),
        title=title,
    )

    return fig