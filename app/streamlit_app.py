"""Streamlit UI for the optimization engine.

Run with::

    streamlit run app/streamlit_app.py

The app is intentionally driven by the same `EngineConfig` machinery as
the CLI, so anything that works in the UI also works headless.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make src/ importable when running ``streamlit run app/streamlit_app.py``
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.analytics.performance import summary_stats  # noqa: E402
from optimization_engine.analytics.relative import summary_relative  # noqa: E402
from optimization_engine.analytics.risk import risk_contribution  # noqa: E402
from optimization_engine.config import EngineConfig, OptimizerSpec  # noqa: E402
from optimization_engine.data.covariance import covariance_matrix  # noqa: E402
from optimization_engine.data.loader import (  # noqa: E402
    load_prices,
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.engine import run_engine  # noqa: E402
from optimization_engine.optimizers.factory import available_optimizers  # noqa: E402
from optimization_engine.reporting.exporters import write_excel_report  # noqa: E402
from optimization_engine.reporting.plots import (  # noqa: E402
    plot_correlation_heatmap,
    plot_drawdown,
    plot_efficient_frontier,
    plot_portfolio_composition,
    plot_risk_contributions,
    plot_wealth_index,
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Optimization Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card { background: #f7f9fc; padding: 0.75rem 1rem; border-radius: 8px;
                   border: 1px solid #e6ebf1; }
    .small-muted { color: #6b7280; font-size: 0.85rem; }
    .section-title { margin-top: 1.5rem; margin-bottom: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Multi-Asset Optimization Engine")
st.caption(
    "Mean-variance · Risk parity · HRP · Black-Litterman · CVaR · Max diversification"
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_sample(n_periods: int) -> pd.DataFrame:
    return sample_dataset(n_periods=n_periods)


def _load_uploaded(file: io.BytesIO, sheet: str) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file, sheet_name=sheet, index_col=0, parse_dates=True)
    if name.endswith(".csv"):
        return pd.read_csv(file, index_col=0, parse_dates=True)
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    raise ValueError(f"Unsupported file: {file.name}")


def _editable_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return returns


# ---------------------------------------------------------------------------
# Sidebar — data + optimizer config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1 · Data")
    data_source = st.radio(
        "Source",
        options=["Sample", "Upload file"],
        index=0,
        horizontal=True,
    )

    if data_source == "Sample":
        years = st.slider("Years of history", 2, 15, 8)
        prices = _load_sample(years * 252)
    else:
        uploaded = st.file_uploader(
            "Price file (Excel/CSV/Parquet)",
            type=["xlsx", "xls", "xlsm", "csv", "parquet"],
        )
        sheet = st.text_input("Sheet name (Excel)", value="Precios")
        if uploaded is None:
            st.info("Upload a file to continue, or switch to Sample data.")
            st.stop()
        prices = _load_uploaded(uploaded, sheet)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().dropna(how="all")

    st.success(f"Loaded {prices.shape[0]} rows × {prices.shape[1]} assets")

    selected_assets = st.multiselect(
        "Universe (assets to include)",
        options=list(prices.columns),
        default=list(prices.columns),
    )
    if not selected_assets:
        st.warning("Select at least one asset.")
        st.stop()
    prices = prices[selected_assets]

    st.divider()
    st.header("2 · Optimizer")
    optimizer_name = st.selectbox(
        "Method",
        options=available_optimizers(),
        index=available_optimizers().index("mean_variance"),
        help="Choose the optimization technique.",
    )
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0, max_value=0.30,
        value=0.04, step=0.005, format="%.4f",
    )
    periods_per_year = st.number_input(
        "Periods per year", min_value=1, max_value=365, value=252
    )
    cov_method = st.selectbox(
        "Covariance estimator",
        options=["ledoit_wolf", "sample", "oas", "ewma", "semi", "shrink"],
        index=0,
    )
    ewma_lambda = (
        st.slider("EWMA λ", 0.80, 0.999, 0.94, 0.005)
        if cov_method == "ewma"
        else 0.94
    )

    target_return: float | None = None
    target_volatility: float | None = None
    risk_aversion = 1.0
    cvar_alpha = 0.05
    risk_budget: dict[str, float] | None = None

    if optimizer_name == "mean_variance":
        mode = st.radio(
            "Mode", ["Target return", "Target volatility", "Utility"], horizontal=True
        )
        if mode == "Target return":
            target_return = st.number_input(
                "Target return (annual)", value=0.07, step=0.005, format="%.4f"
            )
        elif mode == "Target volatility":
            target_volatility = st.number_input(
                "Target volatility (annual)", value=0.10, step=0.005, format="%.4f"
            )
        else:
            risk_aversion = st.slider("Risk aversion λ", 0.1, 20.0, 2.5)
    elif optimizer_name == "cvar":
        cvar_alpha = st.slider(
            "CVaR tail prob α", 0.01, 0.20, 0.05, 0.01,
            help="0.05 ⇒ 95% CVaR.",
        )
        target_return = st.number_input(
            "Target return (optional)", value=0.0, step=0.005, format="%.4f"
        )
        target_return = None if target_return == 0.0 else target_return

    st.divider()
    st.header("3 · Frontier")
    build_frontier = st.checkbox("Build efficient frontier", value=True)
    n_frontier_points = st.slider("Frontier points", 5, 100, 25)


# ---------------------------------------------------------------------------
# Returns + tabs
# ---------------------------------------------------------------------------

returns = prices_to_returns(prices)

st.markdown("---")
tab_overview, tab_assets, tab_constraints, tab_optimize, tab_report = st.tabs(
    [
        "🌐 Overview",
        "📊 Assets",
        "⚙️ Expected returns & constraints",
        "🚀 Optimize",
        "📤 Report",
    ]
)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets", returns.shape[1])
    c2.metric("Periods", returns.shape[0])
    c3.metric("Start", str(returns.index.min().date()))
    c4.metric("End", str(returns.index.max().date()))

    st.subheader("Cumulative returns")
    st.plotly_chart(plot_wealth_index(returns), use_container_width=True)

    st.subheader("Correlation heatmap")
    st.plotly_chart(plot_correlation_heatmap(returns.corr()), use_container_width=True)


# ---------------------------------------------------------------------------
# Assets — per-asset stats
# ---------------------------------------------------------------------------

with tab_assets:
    st.subheader("Per-asset summary statistics")
    stats = summary_stats(
        returns, periods_per_year=periods_per_year, riskfree_rate=risk_free_rate
    )
    formatter = {
        "Annualized Return": "{:.2%}",
        "Annualized Vol": "{:.2%}",
        "Skewness": "{:.3f}",
        "Kurtosis": "{:.3f}",
        "Cornish-Fisher VaR(5%)": "{:.2%}",
        "Historic CVaR(5%)": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Sortino Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}",
    }
    st.dataframe(stats.style.format(formatter), use_container_width=True)

    st.subheader("Drawdown")
    sel = st.multiselect(
        "Series to plot", options=list(returns.columns), default=list(returns.columns[:3])
    )
    if sel:
        st.plotly_chart(plot_drawdown(returns[sel]), use_container_width=True)


# ---------------------------------------------------------------------------
# Expected returns + constraints — fully editable in-table
# ---------------------------------------------------------------------------

with tab_constraints:
    st.subheader("Expected returns and constraints")
    st.caption(
        "Edit any cell. Min/Max are weight bounds; Group is used for "
        "asset-class constraints set below."
    )

    historical_mu = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1

    if "config_table" not in st.session_state or set(st.session_state.config_table.index) != set(returns.columns):
        st.session_state.config_table = pd.DataFrame(
            {
                "Expected Return": historical_mu.round(4),
                "Min Weight": 0.0,
                "Max Weight": 1.0,
                "Group": "Other",
            },
            index=returns.columns,
        )

    edited = st.data_editor(
        st.session_state.config_table,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Expected Return": st.column_config.NumberColumn(format="%.4f"),
            "Min Weight": st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.01, format="%.2f"),
            "Max Weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.5, step=0.01, format="%.2f"),
            "Group": st.column_config.TextColumn(),
        },
    )
    st.session_state.config_table = edited

    st.markdown("**Group constraints**")
    unique_groups = sorted(edited["Group"].dropna().unique().tolist())
    if unique_groups:
        gb_default = pd.DataFrame(
            {"Min Weight": 0.0, "Max Weight": 1.0}, index=unique_groups
        )
        if "group_bounds" not in st.session_state or list(st.session_state.group_bounds.index) != unique_groups:
            st.session_state.group_bounds = gb_default
        st.session_state.group_bounds = st.data_editor(
            st.session_state.group_bounds,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Min Weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.5, step=0.01, format="%.2f"),
                "Max Weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.5, step=0.01, format="%.2f"),
            },
        )

    if optimizer_name == "risk_parity":
        st.markdown("**Risk budgets** (each share of total variance, sums to 1)")
        if "risk_budget" not in st.session_state or set(st.session_state.risk_budget.index) != set(returns.columns):
            st.session_state.risk_budget = pd.DataFrame(
                {"Risk Budget": 1.0 / len(returns.columns)}, index=returns.columns
            )
        st.session_state.risk_budget = st.data_editor(
            st.session_state.risk_budget,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Risk Budget": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.3f"),
            },
        )
        risk_budget = st.session_state.risk_budget["Risk Budget"].to_dict()
    elif optimizer_name == "black_litterman":
        st.markdown("**Black-Litterman views** (asset → annual expected return)")
        if "bl_views" not in st.session_state or set(st.session_state.bl_views.index) != set(returns.columns):
            st.session_state.bl_views = pd.DataFrame(
                {"View": np.nan, "Confidence (variance)": np.nan},
                index=returns.columns,
            )
        st.session_state.bl_views = st.data_editor(
            st.session_state.bl_views,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "View": st.column_config.NumberColumn(format="%.4f"),
                "Confidence (variance)": st.column_config.NumberColumn(format="%.6f"),
            },
        )


def _build_config() -> EngineConfig:
    table = st.session_state.config_table
    bounds = {
        a: [float(table.loc[a, "Min Weight"]), float(table.loc[a, "Max Weight"])]
        for a in returns.columns
    }
    groups = {a: str(table.loc[a, "Group"]) for a in returns.columns}
    group_bounds: dict[str, list[float]] = {}
    if "group_bounds" in st.session_state:
        for g, row in st.session_state.group_bounds.iterrows():
            group_bounds[str(g)] = [float(row["Min Weight"]), float(row["Max Weight"])]

    expected_returns = {a: float(table.loc[a, "Expected Return"]) for a in returns.columns}

    spec = OptimizerSpec(
        name=optimizer_name,
        target_return=target_return,
        target_volatility=target_volatility,
        risk_free_rate=float(risk_free_rate),
        risk_aversion=float(risk_aversion),
        cvar_alpha=float(cvar_alpha),
    )

    if optimizer_name == "risk_parity" and "risk_budget" in st.session_state:
        spec.risk_budget = st.session_state.risk_budget["Risk Budget"].to_dict()

    if optimizer_name == "black_litterman" and "bl_views" in st.session_state:
        v = st.session_state.bl_views
        spec.bl_views = {
            a: float(v.loc[a, "View"])
            for a in v.index
            if pd.notna(v.loc[a, "View"])
        }
        spec.bl_view_confidences = {
            a: float(v.loc[a, "Confidence (variance)"])
            for a in v.index
            if pd.notna(v.loc[a, "Confidence (variance)"])
        }

    return EngineConfig(
        expected_returns=expected_returns,
        bounds=bounds,
        groups=groups,
        group_bounds=group_bounds,
        periods_per_year=int(periods_per_year),
        covariance_method=cov_method,
        ewma_lambda=float(ewma_lambda),
        optimizer=spec,
    )


# ---------------------------------------------------------------------------
# Optimize
# ---------------------------------------------------------------------------

with tab_optimize:
    st.subheader("Run optimization")
    if st.button("Optimize portfolio", type="primary"):
        config = _build_config()
        with st.spinner("Solving…"):
            try:
                run = run_engine(
                    returns,
                    config,
                    build_frontier=build_frontier,
                    n_frontier_points=n_frontier_points,
                )
            except Exception as exc:
                st.error(f"Optimization failed: {exc}")
                st.stop()
        st.session_state["last_run"] = run
        st.success("Done.")

    run = st.session_state.get("last_run")
    if run is None:
        st.info("Configure the optimizer in the sidebar and the constraints tab, then click **Optimize portfolio**.")
        st.stop()

    weights = run.result.weights
    nonzero = weights[weights.abs() > 1e-4].sort_values(ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Return", f"{run.result.expected_return:.2%}")
    c2.metric("Expected Volatility", f"{run.result.expected_volatility:.2%}")
    c3.metric("Sharpe Ratio", f"{run.result.sharpe_ratio:.2f}")
    c4.metric("Active Positions", f"{int((weights.abs() > 1e-4).sum())}")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Weights**")
        st.dataframe(
            nonzero.to_frame("Weight").style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
        )

    with right:
        rc = run.risk_contributions().rename("Risk Contribution")
        rc_nz = rc[rc.abs() > 1e-4].sort_values(ascending=False)
        st.markdown("**Risk contributions** (% of variance)")
        st.dataframe(
            rc_nz.to_frame().style.format({"Risk Contribution": "{:.2%}"}),
            use_container_width=True,
        )

    if run.frontier is not None:
        st.markdown("### Efficient Frontier")
        st.plotly_chart(
            plot_efficient_frontier(run.frontier.summary, run.frontier.max_sharpe_index),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_portfolio_composition(run.frontier.weights, "Weights along frontier"),
                use_container_width=True,
            )
        with c2:
            if run.frontier.group_weights is not None and not run.frontier.group_weights.empty:
                st.plotly_chart(
                    plot_portfolio_composition(
                        run.frontier.group_weights, "Group weights along frontier"
                    ),
                    use_container_width=True,
                )

    bt = run.backtest_returns()
    st.markdown("### Backtest")
    st.plotly_chart(plot_wealth_index(bt, "Portfolio wealth (1 = inception)"), use_container_width=True)
    st.plotly_chart(plot_drawdown(bt["portfolio"], "Portfolio drawdown"), use_container_width=True)

    abs_summary = run.absolute_summary(riskfree_rate=risk_free_rate)
    st.markdown("**Backtest summary**")
    st.dataframe(
        abs_summary.style.format(
            {
                "Annualized Return": "{:.2%}",
                "Annualized Vol": "{:.2%}",
                "Skewness": "{:.3f}",
                "Kurtosis": "{:.3f}",
                "Cornish-Fisher VaR(5%)": "{:.2%}",
                "Historic CVaR(5%)": "{:.2%}",
                "Sharpe Ratio": "{:.3f}",
                "Sortino Ratio": "{:.3f}",
                "Max Drawdown": "{:.2%}",
            }
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

with tab_report:
    run = st.session_state.get("last_run")
    if run is None:
        st.info("Run an optimization first.")
        st.stop()

    sheets: dict[str, pd.DataFrame] = {
        "weights": run.result.weights.to_frame("weight"),
        "summary": pd.DataFrame(
            [
                {
                    "expected_return": run.result.expected_return,
                    "expected_volatility": run.result.expected_volatility,
                    "sharpe_ratio": run.result.sharpe_ratio,
                }
            ]
        ),
        "risk_contributions": run.risk_contributions().to_frame("share_of_variance"),
        "expected_returns": run.expected_returns.to_frame("annualized"),
        "cov_matrix": run.cov_matrix,
        "absolute_summary": run.absolute_summary(riskfree_rate=risk_free_rate),
    }
    if run.frontier is not None:
        sheets["frontier_summary"] = run.frontier.summary
        sheets["frontier_weights"] = run.frontier.weights
        if run.frontier.group_weights is not None and not run.frontier.group_weights.empty:
            sheets["frontier_groups"] = run.frontier.group_weights

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=True)
    buf.seek(0)
    st.download_button(
        label="📥 Download Excel report",
        data=buf,
        file_name="optimization_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("**Config used (YAML)**")
    import yaml as _yaml
    st.code(_yaml.safe_dump(run.config.to_dict(), sort_keys=False), language="yaml")
