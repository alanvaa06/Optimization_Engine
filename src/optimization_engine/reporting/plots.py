"""Plotting helpers (Plotly).

Plotly is preferred over Matplotlib here because the same figure objects
render natively in Streamlit and in Jupyter without extra setup.

A note on the frontier chart: it takes a :class:`FrontierResult` rather than
a bare summary frame. Passing the frame alone is how the "max Sharpe" star
ends up on the wrong point — the caller drops NaN rows for plotting, then
highlights a position computed against the undropped frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#: Muted, colour-blind-safe qualitative palette used across the app.
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]

_LAYOUT = dict(
    template="plotly_white",
    colorway=PALETTE,
    margin=dict(l=60, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def plot_efficient_frontier(
    frontier,
    highlight_index: int | None = None,
    title: str = "Efficient Frontier",
    risk_free_rate: float | None = None,
    current_portfolio: tuple[float, float, str] | None = None,
    show_dominated: bool = False,
) -> go.Figure:
    """Draw a frontier with its anchor portfolios and, optionally, the CAL.

    Args:
        frontier: A ``FrontierResult``. A bare summary DataFrame is still
            accepted for backwards compatibility, but then the anchors and
            the dominated branch cannot be drawn.
        highlight_index: Position within the plotted frame to star. Defaults
            to the highest-Sharpe point.
        title: Chart title.
        risk_free_rate: When given, draws the capital allocation line from
            ``rf`` through the tangency portfolio — the set of risk/return
            combinations reachable by mixing it with cash.
        current_portfolio: ``(risk, return, label)`` for the allocation
            actually chosen, so the analyst can see where it sits relative to
            the frontier. ``risk`` must be measured on the same axis the
            frontier uses — pass ``None`` rather than a volatility when the
            frontier is drawn against CVaR.
        show_dominated: Also draw the inefficient branch below the
            minimum-variance point, dashed and greyed.
    """
    is_result = hasattr(frontier, "plot_frame")
    if is_result:
        df = frontier.plot_frame(efficient_only=True)
        summary = frontier.summary
    else:
        summary = frontier
        df = summary.dropna(
            subset=["expected_volatility", "expected_return"]
        ).reset_index(drop=True)

    # Plot the risk measure the frontier was actually traced against. Drawing
    # a mean-CVaR frontier on a volatility axis shows a curve that was never
    # optimized, and the shapes differ wherever returns are skewed.
    risk_measure = getattr(frontier, "risk_measure", "volatility") if is_result else "volatility"
    if risk_measure == "CVaR" and "cvar" in df.columns and df["cvar"].notna().any():
        risk_col = "cvar"
        risk_label = "Conditional VaR (annualized)"
        # The minimum-variance and tangency anchors, and the capital
        # allocation line, are defined in volatility space. They have no
        # position on a CVaR axis, so they are not drawn there.
        show_anchors = False
    else:
        risk_col = "expected_volatility"
        risk_label = "Volatility (annualized)"
        show_anchors = True

    fig = go.Figure()

    if show_dominated and is_result and "is_efficient" in summary.columns:
        dominated = summary[
            (summary["status"] == "ok") & (~summary["is_efficient"])
        ].dropna(subset=["expected_volatility", "expected_return"])
        if not dominated.empty:
            fig.add_trace(
                go.Scatter(
                    x=dominated[risk_col],
                    y=dominated["expected_return"],
                    mode="markers+lines",
                    line=dict(dash="dot", color="#BAB0AC"),
                    marker=dict(size=6, color="#BAB0AC"),
                    name="Dominated (below min-variance)",
                    hovertemplate=(
                        "Vol: %{x:.2%}<br>Return: %{y:.2%}<br>"
                        "<i>Inefficient: more return is available at this "
                        "risk</i><extra></extra>"
                    ),
                )
            )

    if not df.empty:
        fig.add_trace(
            go.Scatter(
                x=df[risk_col],
                y=df["expected_return"],
                mode="markers+lines",
                line=dict(color="#4C78A8", width=2),
                marker=dict(
                    size=8,
                    color=df["sharpe_ratio"],
                    colorscale="Viridis",
                    colorbar=dict(title="Sharpe", thickness=12),
                    showscale=True,
                ),
                name="Efficient frontier",
                hovertemplate=(
                    f"{risk_label.split(' (')[0]}: %{{x:.2%}}<br>"
                    "Return: %{y:.2%}<br>"
                    "Sharpe: %{marker.color:.3f}<extra></extra>"
                ),
            )
        )

        has_exact_tangency = is_result and show_anchors and frontier.tangency is not None
        if highlight_index is None and not has_exact_tangency and "sharpe_ratio" in df.columns:
            valid = df["sharpe_ratio"].dropna()
            highlight_index = int(valid.idxmax()) if len(valid) else None
        if highlight_index is not None and 0 <= highlight_index < len(df):
            row = df.iloc[highlight_index]
            fig.add_trace(
                go.Scatter(
                    x=[row[risk_col]],
                    y=[row["expected_return"]],
                    mode="markers",
                    marker=dict(size=16, color="#E45756", symbol="star"),
                    name="Max Sharpe on frontier",
                    hovertemplate=(
                        "Max Sharpe<br>Vol: %{x:.2%}<br>Return: %{y:.2%}"
                        "<extra></extra>"
                    ),
                )
            )

    if is_result and show_anchors:
        for anchor, symbol, color in (
            (frontier.min_variance, "diamond", "#54A24B"),
            (frontier.tangency, "circle", "#F58518"),
        ):
            if anchor is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[anchor["expected_volatility"]],
                    y=[anchor["expected_return"]],
                    mode="markers",
                    marker=dict(size=13, color=color, symbol=symbol,
                                line=dict(width=1.5, color="white")),
                    name=str(anchor["label"]),
                    hovertemplate=(
                        f"{anchor['label']}<br>Vol: %{{x:.2%}}<br>"
                        "Return: %{y:.2%}<extra></extra>"
                    ),
                )
            )

        if risk_free_rate is not None and frontier.tangency is not None:
            tan_vol = float(frontier.tangency["expected_volatility"])
            tan_ret = float(frontier.tangency["expected_return"])
            if tan_vol > 0:
                x_max = float(
                    max(
                        df["expected_volatility"].max() if not df.empty else tan_vol,
                        tan_vol,
                    )
                    * 1.15
                )
                slope = (tan_ret - risk_free_rate) / tan_vol
                fig.add_trace(
                    go.Scatter(
                        x=[0.0, x_max],
                        y=[risk_free_rate, risk_free_rate + slope * x_max],
                        mode="lines",
                        line=dict(dash="dash", color="#F58518", width=1.5),
                        name=f"Capital allocation line (slope {slope:.2f})",
                        hovertemplate=(
                            "Vol: %{x:.2%}<br>Return: %{y:.2%}<br>"
                            "<i>Mix of cash and the tangency portfolio</i>"
                            "<extra></extra>"
                        ),
                    )
                )

    if current_portfolio is not None:
        vol, ret, label = current_portfolio
        fig.add_trace(
            go.Scatter(
                x=[vol], y=[ret],
                mode="markers+text",
                marker=dict(size=15, color="#B279A2", symbol="x-thin",
                            line=dict(width=3, color="#B279A2")),
                text=[label], textposition="bottom center",
                name=label,
                hovertemplate=(
                    f"{label}<br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=risk_label,
        yaxis_title="Expected Return (annualized)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", y=-0.18, x=0),
        **_LAYOUT,
    )
    return fig


def plot_portfolio_composition(
    weights: pd.DataFrame, title: str = "Portfolio Composition", as_percent: bool = True
) -> go.Figure:
    """Stacked weights, one bar per column of ``weights``."""
    df = weights.T.copy()
    if as_percent:
        df = df * 100.0
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(
            go.Bar(
                name=str(col),
                x=df.index.astype(str),
                y=df[col],
                marker_color=PALETTE[i % len(PALETTE)],
                hovertemplate=f"{col}<br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="",
        yaxis_title="Weight (%)" if as_percent else "Weight",
        legend=dict(orientation="v", x=1.02, y=1),
        **_LAYOUT,
    )
    return fig


def plot_weights_bar(
    weights: pd.Series, title: str = "Weights", bounds: pd.DataFrame | None = None
) -> go.Figure:
    """Horizontal weight bars, optionally showing where each bound binds.

    Seeing which positions are pinned to a cap is the fastest way to tell
    whether the constraints or the optimizer produced the answer.
    """
    w = weights.sort_values()
    colors = [PALETTE[0] if v >= 0 else PALETTE[3] for v in w.values]
    at_bound: list[str] = []
    if bounds is not None:
        for asset, value in w.items():
            lo = float(bounds.loc[asset, "Min Weight"])
            hi = float(bounds.loc[asset, "Max Weight"])
            if abs(value - hi) < 1e-4:
                at_bound.append("at max")
            elif abs(value - lo) < 1e-4 and lo > 0:
                at_bound.append("at min")
            else:
                at_bound.append("")
    fig = go.Figure(
        go.Bar(
            x=w.values * 100,
            y=[str(i) for i in w.index],
            orientation="h",
            marker_color=colors,
            text=[
                f"{v:.1%}" + (f"  ({b})" if b else "")
                for v, b in zip(w.values, at_bound or [""] * len(w))
            ],
            textposition="auto",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=title, xaxis_title="Weight (%)", yaxis_title="", showlegend=False,
        height=max(320, 26 * len(w)), **_LAYOUT,
    )
    return fig


def plot_risk_contributions(rc: pd.DataFrame, title: str = "Risk Contributions") -> go.Figure:
    """Grouped bars of risk shares (or any per-asset frame)."""
    df = rc * 100.0
    fig = px.bar(df, barmode="group", title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(yaxis_title="% of risk", xaxis_title="Asset", **_LAYOUT)
    return fig


def plot_weight_vs_risk(
    decomposition: pd.DataFrame, title: str = "Capital vs. risk"
) -> go.Figure:
    """Capital weight beside risk share for each asset.

    The gap between the two bars is the whole argument for risk parity: a
    position can be small in capital and large in risk.
    """
    df = decomposition.sort_values("share_of_risk", ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Capital weight", x=[str(i) for i in df.index],
            y=df["weight"] * 100, marker_color=PALETTE[0],
            hovertemplate="%{x}<br>Capital: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Risk share", x=[str(i) for i in df.index],
            y=df["share_of_risk"] * 100, marker_color=PALETTE[1],
            hovertemplate="%{x}<br>Risk: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group", title=title, yaxis_title="% of portfolio",
        xaxis_title="", legend=dict(orientation="h", y=-0.25), **_LAYOUT,
    )
    return fig


def plot_wealth_index(returns: pd.DataFrame, title: str = "Wealth Index") -> go.Figure:
    """Cumulative growth of 1 unit."""
    wealth = (1 + returns).cumprod()
    fig = px.line(wealth, title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(
        yaxis_title="Wealth (start = 1)", xaxis_title="", **_LAYOUT
    )
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>")
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Correlation matrix with values annotated for small universes."""
    fig = px.imshow(
        corr.values,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title,
        aspect="auto",
        text_auto=".2f" if len(corr) <= 15 else False,
    )
    fig.update_layout(**_LAYOUT)
    fig.update_traces(
        hovertemplate="%{y} vs %{x}<br>ρ = %{z:.3f}<extra></extra>"
    )
    return fig


def plot_drawdown(returns: pd.Series | pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    """Underwater chart."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    fig = px.area(dd, title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(
        yaxis_title="Drawdown", yaxis_tickformat=".0%", xaxis_title="", **_LAYOUT
    )
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>")
    return fig


def plot_rolling_metrics(
    rolling: pd.DataFrame, title: str = "Rolling performance"
) -> go.Figure:
    """Rolling return, volatility, Sharpe and drawdown, in stacked panels.

    A full-sample Sharpe cannot distinguish a strategy that worked throughout
    from one that earned everything in a single quarter. This can.

    The panels are stacked rather than overlaid on twin axes: a Sharpe series
    and an annualized-return series happen to occupy similar numeric ranges,
    so on one plot they trace nearly the same path and read as a duplicate
    line rather than two different measurements.
    """
    from plotly.subplots import make_subplots

    panels = [
        ("rolling_return", "Return", ".1%", ".0%"),
        ("rolling_volatility", "Volatility", ".1%", ".0%"),
        ("rolling_sharpe", "Sharpe", ".2f", ".1f"),
        ("rolling_drawdown", "Drawdown", ".1%", ".0%"),
    ]
    present = [p for p in panels if p[0] in rolling.columns]
    if not present:
        return go.Figure()

    fig = make_subplots(
        rows=len(present), cols=1, shared_xaxes=True, vertical_spacing=0.045,
        subplot_titles=[p[1] for p in present],
    )
    for row, (col, label, hover_fmt, tick_fmt) in enumerate(present, start=1):
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling[col],
                name=label,
                line=dict(color=PALETTE[(row - 1) % len(PALETTE)], width=1.6),
                fill="tozeroy" if col == "rolling_drawdown" else None,
                hovertemplate=(
                    f"{label}<br>%{{x|%Y-%m-%d}}: %{{y:{hover_fmt}}}<extra></extra>"
                ),
            ),
            row=row, col=1,
        )
        fig.update_yaxes(tickformat=tick_fmt, row=row, col=1)
        if col == "rolling_sharpe":
            fig.add_hline(y=0, line_width=1, line_color="#BAB0AC", row=row, col=1)

    fig.update_layout(
        title=title,
        showlegend=False,
        height=170 * len(present) + 90,
        **_LAYOUT,
    )
    fig.update_annotations(font_size=12)
    return fig


def plot_walk_forward_comparison(
    in_sample: pd.Series,
    out_of_sample: pd.Series,
    title: str = "In-sample vs. out-of-sample",
) -> go.Figure:
    """Fitted and walk-forward wealth curves on one axis.

    The distance between the lines is how much of the backtest was hindsight.
    """
    fig = go.Figure()
    for name, series, color, dash in (
        ("In-sample (fitted)", in_sample, PALETTE[0], None),
        ("Out-of-sample (walk-forward)", out_of_sample, PALETTE[3], "dash"),
    ):
        wealth = (1 + series).cumprod()
        fig.add_trace(
            go.Scatter(
                x=wealth.index, y=wealth.values, name=name,
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title, yaxis_title="Wealth (start = 1)", xaxis_title="",
        legend=dict(orientation="h", y=-0.2), **_LAYOUT,
    )
    return fig


def plot_weight_evolution(
    weights: pd.DataFrame, title: str = "Weights through time"
) -> go.Figure:
    """Stacked area of held weights over a backtest."""
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        fig.add_trace(
            go.Scatter(
                x=weights.index, y=weights[col] * 100,
                name=str(col), mode="lines", stackgroup="one",
                line=dict(width=0.5, color=PALETTE[i % len(PALETTE)]),
                hovertemplate=f"{col}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        title=title, yaxis_title="Weight (%)", xaxis_title="",
        legend=dict(orientation="v", x=1.02, y=1), **_LAYOUT,
    )
    return fig


def plot_return_distribution(
    returns: pd.Series,
    var: float | None = None,
    cvar: float | None = None,
    title: str = "Return distribution",
) -> go.Figure:
    """Histogram of periodic returns with the VaR and CVaR thresholds marked.

    Seeing where the tail cut falls relative to the actual histogram is what
    makes a CVaR number mean something.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=returns.values, nbinsx=60, marker_color=PALETTE[0],
            name="Returns", opacity=0.85,
            hovertemplate="%{x:.2%}<br>count %{y}<extra></extra>",
        )
    )
    for value, label, color in (
        (var, "VaR", PALETTE[1]),
        (cvar, "CVaR", PALETTE[3]),
    ):
        if value is None or not np.isfinite(value):
            continue
        fig.add_vline(
            x=-abs(value), line_dash="dash", line_color=color,
            annotation_text=f"{label} {abs(value):.2%}",
            annotation_position="top left",
        )
    fig.update_layout(
        title=title, xaxis_title="Periodic return", yaxis_title="Frequency",
        xaxis_tickformat=".1%", showlegend=False, **_LAYOUT,
    )
    return fig


def plot_frontier_uncertainty(
    uncertainty,
    title: str = "How much can you trust this frontier?",
) -> go.Figure:
    """The efficient frontier as a confidence band rather than a single line.

    The point estimate is drawn on top of the quantile band from resampled
    histories. Where the band is wide, two portfolios that look different on
    the point-estimate curve are not distinguishable given the data.
    """
    q = uncertainty.quantiles
    x = list(q.index)
    fig = go.Figure()

    for lo, hi, opacity, label in (
        ("q05", "q95", 0.14, "5th-95th percentile"),
        ("q25", "q75", 0.24, "25th-75th percentile"),
    ):
        if lo not in q.columns or hi not in q.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=list(q[hi]) + list(q[lo])[::-1],
                fill="toself",
                fillcolor=f"rgba(76, 120, 168, {opacity})",
                line=dict(width=0),
                name=label,
                hoverinfo="skip",
            )
        )

    if "q50" in q.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=q["q50"], mode="lines",
                line=dict(color="#4C78A8", width=2, dash="dot"),
                name="Median across draws",
                hovertemplate="Vol: %{x:.2%}<br>Median return: %{y:.2%}<extra></extra>",
            )
        )

    point = uncertainty.point_estimate.plot_frame()
    if not point.empty and x:
        # Clip the point estimate to the band's range. The band spans only the
        # risk levels *every* draw could reach, so drawing the full
        # point-estimate curve beside it makes the band look truncated when it
        # is simply the region where a comparison is defined.
        lo, hi = min(x), max(x)
        clipped = point[
            (point["expected_volatility"] >= lo - 1e-12)
            & (point["expected_volatility"] <= hi + 1e-12)
        ]
        if clipped.empty:
            clipped = point
        fig.add_trace(
            go.Scatter(
                x=clipped["expected_volatility"], y=clipped["expected_return"],
                mode="lines+markers",
                line=dict(color="#E45756", width=2.5),
                marker=dict(size=6),
                name="Point estimate (observed sample)",
                hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Volatility (annualized)",
        yaxis_title="Expected Return (annualized)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", y=-0.2),
        **_LAYOUT,
    )
    return fig


def plot_weight_dispersion(
    dispersion: pd.Series, title: str = "Which weights the data cannot pin down"
) -> go.Figure:
    """Standard deviation of each asset's weight across resampled frontiers.

    A large bar means the optimizer's conviction in that position comes from
    the particular sample it was handed, not from the asset.
    """
    d = dispersion.sort_values()
    fig = go.Figure(
        go.Bar(
            x=d.values * 100,
            y=[str(i) for i in d.index],
            orientation="h",
            marker_color=PALETTE[1],
            hovertemplate="%{y}: ±%{x:.2f}pp across draws<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Weight standard deviation across resampled histories (pp)",
        yaxis_title="", showlegend=False,
        height=max(300, 24 * len(d)), **_LAYOUT,
    )
    return fig
