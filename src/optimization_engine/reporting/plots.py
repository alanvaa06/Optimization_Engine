"""Plotting helpers (Plotly).

Plotly is preferred over Matplotlib here because the same figure objects
render natively in Streamlit and in Jupyter without extra setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_efficient_frontier(
    summary: pd.DataFrame,
    highlight_index: int | None = None,
    title: str = "Efficient Frontier",
) -> go.Figure:
    df = summary.dropna(subset=["expected_volatility", "expected_return"]).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["expected_volatility"],
            y=df["expected_return"],
            mode="markers+lines",
            marker=dict(
                size=8,
                color=df["sharpe_ratio"],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
                showscale=True,
            ),
            name="Frontier",
            hovertemplate=(
                "Vol: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                "Sharpe: %{marker.color:.3f}<extra></extra>"
            ),
        )
    )
    if highlight_index is not None and 0 <= highlight_index < len(df):
        row = df.iloc[highlight_index]
        fig.add_trace(
            go.Scatter(
                x=[row["expected_volatility"]],
                y=[row["expected_return"]],
                mode="markers",
                marker=dict(size=18, color="red", symbol="star"),
                name="Max Sharpe",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Volatility (annualized)",
        yaxis_title="Expected Return (annualized)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_portfolio_composition(
    weights: pd.DataFrame, title: str = "Portfolio Composition", as_percent: bool = True
) -> go.Figure:
    df = weights.T.copy()
    if as_percent:
        df = df * 100.0
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Bar(name=col, x=df.index.astype(str), y=df[col])
        )
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Target",
        yaxis_title="Weight (%)" if as_percent else "Weight",
        template="plotly_white",
        legend=dict(orientation="v", x=1.02, y=1),
    )
    return fig


def plot_risk_contributions(rc: pd.DataFrame, title: str = "Risk Contributions") -> go.Figure:
    df = rc * 100.0
    fig = px.bar(df, barmode="group", title=title, template="plotly_white")
    fig.update_layout(yaxis_title="% of variance", xaxis_title="Asset")
    return fig


def plot_wealth_index(returns: pd.DataFrame, title: str = "Wealth Index") -> go.Figure:
    wealth = (1 + returns).cumprod()
    fig = px.line(wealth, title=title, template="plotly_white")
    fig.update_layout(yaxis_title="Wealth (start = 1)", xaxis_title="Date")
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    fig = px.imshow(
        corr.values,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title,
        aspect="auto",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_drawdown(returns: pd.Series | pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    fig = px.area(dd, title=title, template="plotly_white")
    fig.update_layout(yaxis_title="Drawdown", yaxis_tickformat=".0%")
    return fig
