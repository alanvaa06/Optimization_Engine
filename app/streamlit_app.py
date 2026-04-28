"""Streamlit UI for the optimization engine.

Run with::

    streamlit run app/streamlit_app.py

The app is intentionally driven by the same `EngineConfig` machinery as
the CLI, so anything that works in the UI also works headless.
"""

from __future__ import annotations

import io
import json
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
from optimization_engine.data.fx import (  # noqa: E402
    FXError,
    convert_prices_to_base,
    fetch_fx_to_base,
    supported_currencies,
)
from optimization_engine.data.yahoo import (  # noqa: E402
    YahooFinanceError,
    load_prices_yahoo,
)
from optimization_engine.engine import apply_fx_conversion, run_engine  # noqa: E402
from optimization_engine.optimizers.factory import available_optimizers  # noqa: E402
from optimization_engine.reporting.exporters import write_excel_report  # noqa: E402
from optimization_engine.scenarios import (  # noqa: E402
    NOTES_MAX_LEN,
    Scenario,
    config_signature,
    delete_scenario as _delete_scenario,
    dump_scenarios_yaml,
    load_scenarios_yaml,
    now_iso,
    rename_scenario as _rename_scenario,
    scenario_signature,
)
from optimization_engine.reporting.plots import (  # noqa: E402
    plot_correlation_heatmap,
    plot_drawdown,
    plot_efficient_frontier,
    plot_portfolio_composition,
    plot_risk_contributions,
    plot_wealth_index,
)
from optimization_engine.ui_state import (  # noqa: E402
    derive_widget_state,
    yahoo_cache_key,
    yahoo_prices_for_rerun,
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

# ---------------------------------------------------------------------------
# Scenario state (initialized once per session) + pending Load handler
# ---------------------------------------------------------------------------

if "scenarios" not in st.session_state:
    st.session_state.scenarios: dict[str, Scenario] = {}
if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = None
if "last_run_by_scenario" not in st.session_state:
    st.session_state.last_run_by_scenario = {}
if "scenario_load_warning" not in st.session_state:
    st.session_state.scenario_load_warning = None


def _seed_table_from_config(cfg) -> None:
    """Replace the editable assets / groups / FX tables with values from cfg."""
    assets = list(cfg.expected_returns.keys())
    st.session_state.config_table = pd.DataFrame(
        {
            "Expected Return": pd.Series(cfg.expected_returns).reindex(assets),
            "Min Weight": pd.Series(
                {a: float(cfg.bounds.get(a, [0.0, 1.0])[0]) for a in assets}
            ),
            "Max Weight": pd.Series(
                {a: float(cfg.bounds.get(a, [0.0, 1.0])[1]) for a in assets}
            ),
            "Group": pd.Series({a: str(cfg.groups.get(a, "Other")) for a in assets}),
            "Currency": pd.Series(
                {a: str(cfg.currencies.get(a, cfg.base_currency)) for a in assets}
            ),
        },
        index=assets,
    )
    if cfg.group_bounds:
        st.session_state.group_bounds = pd.DataFrame(
            {
                "Min Weight": [float(v[0]) for v in cfg.group_bounds.values()],
                "Max Weight": [float(v[1]) for v in cfg.group_bounds.values()],
            },
            index=list(cfg.group_bounds.keys()),
        )
    st.session_state.asset_currency = {
        a: str(cfg.currencies.get(a, cfg.base_currency)) for a in assets
    }
    if getattr(cfg.optimizer, "risk_budget", None):
        st.session_state.risk_budget = pd.DataFrame(
            {"Risk Budget": pd.Series(cfg.optimizer.risk_budget)}
        )
    if getattr(cfg.optimizer, "bl_views", None):
        idx = sorted(
            set(cfg.optimizer.bl_views) | set(cfg.optimizer.bl_view_confidences or {})
        )
        st.session_state.bl_views = pd.DataFrame(
            {
                "View": pd.Series(cfg.optimizer.bl_views).reindex(idx),
                "Confidence (variance)": pd.Series(
                    cfg.optimizer.bl_view_confidences or {}
                ).reindex(idx),
            },
            index=idx,
        )


_pending = st.session_state.pop("pending_scenario_load", None)
if _pending and _pending in st.session_state.scenarios:
    _scn = st.session_state.scenarios[_pending]
    _cfg = _scn.config
    # Seed sidebar-widget keys *before* the widgets render this run.
    st.session_state["optimizer_name"] = _cfg.optimizer.name
    st.session_state["cov_method"] = _cfg.covariance_method
    st.session_state["ewma_lambda"] = float(_cfg.ewma_lambda)
    st.session_state["base_currency"] = _cfg.base_currency
    st.session_state["risk_free_rate"] = float(_cfg.optimizer.risk_free_rate)
    st.session_state["risk_aversion"] = float(_cfg.optimizer.risk_aversion)
    st.session_state["cvar_alpha"] = float(_cfg.optimizer.cvar_alpha)
    if _cfg.optimizer.target_return is not None:
        st.session_state["mv_mode"] = "Target return"
        st.session_state["target_return"] = float(_cfg.optimizer.target_return)
    elif _cfg.optimizer.target_volatility is not None:
        st.session_state["mv_mode"] = "Target volatility"
        st.session_state["target_volatility"] = float(_cfg.optimizer.target_volatility)
    else:
        st.session_state["mv_mode"] = "Utility"
    _seed_table_from_config(_cfg)
    st.session_state.active_scenario = _pending


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


@st.cache_data(show_spinner=True, ttl=60 * 60)
def _load_yahoo_cached(
    tickers: tuple[str, ...],
    period: str,
    start: str | None,
    end: str | None,
    interval: str,
) -> pd.DataFrame:
    if start:
        return load_prices_yahoo(list(tickers), start=start, end=end or None, interval=interval)
    return load_prices_yahoo(list(tickers), period=period, interval=interval)


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
        options=["Sample", "Upload file", "Yahoo Finance"],
        index=0,
        horizontal=True,
    )

    if data_source == "Sample":
        years = st.slider("Years of history", 2, 15, 8)
        prices = _load_sample(years * 252)
    elif data_source == "Upload file":
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
    else:
        st.markdown(
            "Pull adjusted prices directly from Yahoo Finance. "
            "Tickers are validated locally before any network call."
        )
        yahoo_tickers = st.text_input(
            "Tickers (comma- or space-separated)",
            value="SPY, QQQ, EFA, EEM, AGG, TLT, IEF, GLD, DBC, VNQ",
        )
        yahoo_period = st.selectbox(
            "Period",
            options=["1y", "2y", "5y", "10y", "max", "Custom range"],
            index=2,
        )
        yahoo_start: str | None = None
        yahoo_end: str | None = None
        if yahoo_period == "Custom range":
            today = pd.Timestamp.today().normalize()
            default_start = (today - pd.DateOffset(years=5)).date()
            yahoo_start = str(st.date_input("Start", value=default_start))
            yahoo_end = str(st.date_input("End", value=today.date()))
            yahoo_period = "5y"  # ignored when start is set
        yahoo_interval = st.selectbox(
            "Interval", options=["1d", "1wk", "1mo"], index=0
        )

        fetch_clicked = st.button("Fetch from Yahoo", type="primary")

        tickers_tuple = tuple(t for t in yahoo_tickers.replace(",", " ").split() if t)
        cache_key = yahoo_cache_key(
            tickers_tuple,
            yahoo_period,
            yahoo_start,
            yahoo_end,
            yahoo_interval,
        )

        try:
            prices = yahoo_prices_for_rerun(
                fetch_clicked=fetch_clicked,
                cache_key=cache_key,
                state=st.session_state,
                fetch_prices=lambda: _load_yahoo_cached(
                    tickers_tuple,
                    period=yahoo_period,
                    start=yahoo_start,
                    end=yahoo_end,
                    interval=yahoo_interval,
                ),
            )
        except YahooFinanceError as exc:
            st.error(f"Yahoo Finance error: {exc}")
            st.stop()
        except Exception as exc:  # network / library issues
            st.error(f"Could not load Yahoo prices: {exc}")
            st.stop()
        if prices is None:
            st.info("Set tickers and click **Fetch from Yahoo** to download prices.")
            st.stop()

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
    st.header("2 · Currency")
    currency_options = supported_currencies()
    base_currency = st.selectbox(
        "Base currency",
        options=currency_options,
        index=currency_options.index("USD"),
        key="base_currency",
        help=(
            "All asset prices are converted into this currency before "
            "computing returns. FX rates come from FRED."
        ),
    )

    st.divider()
    st.header("3 · Optimizer")
    optimizer_name = st.selectbox(
        "Method",
        options=available_optimizers(),
        index=available_optimizers().index("mean_variance"),
        key="optimizer_name",
        help="Choose the optimization technique.",
    )
    ws = derive_widget_state(optimizer_name)
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0, max_value=0.30,
        value=0.04, step=0.005, format="%.4f",
        key="risk_free_rate",
        disabled=not ws["risk_free_rate"]["enabled"],
        help=ws["risk_free_rate"]["tooltip"],
    )
    periods_per_year = st.number_input(
        "Periods per year", min_value=1, max_value=365, value=252,
        key="periods_per_year",
    )
    cov_method = st.selectbox(
        "Covariance estimator",
        options=["ledoit_wolf", "sample", "oas", "ewma", "semi", "shrink"],
        index=0,
        key="cov_method",
        disabled=not ws["cov_method"]["enabled"],
        help=ws["cov_method"]["tooltip"],
    )
    ewma_lambda = (
        st.slider(
            "EWMA λ", 0.80, 0.999, 0.94, 0.005,
            key="ewma_lambda",
            disabled=not ws["ewma_lambda"]["enabled"],
            help=ws["ewma_lambda"]["tooltip"],
        )
        if cov_method == "ewma" and ws["cov_method"]["enabled"]
        else 0.94
    )

    target_return: float | None = None
    target_volatility: float | None = None
    risk_aversion = 1.0
    cvar_alpha = 0.05
    risk_budget: dict[str, float] | None = None

    # The Mode radio is for methods that genuinely offer >1 mode
    # (mean_variance and Black-Litterman). CVaR has only target_return and
    # is handled in its own block below.
    show_mode_radio = (
        optimizer_name != "cvar"
        and (ws["target_return"]["enabled"] or ws["target_volatility"]["enabled"])
        and ws["risk_aversion"]["enabled"]
    )
    if show_mode_radio:
        modes = []
        if ws["target_return"]["enabled"]:
            modes.append("Target return")
        if ws["target_volatility"]["enabled"]:
            modes.append("Target volatility")
        if ws["risk_aversion"]["enabled"]:
            modes.append("Utility")
        mode = st.radio(
            "Mode", modes, horizontal=True, key="mv_mode",
        )
        if mode == "Target return":
            target_return = st.number_input(
                "Target return (annual)", value=0.07, step=0.005, format="%.4f",
                key="target_return",
            )
        elif mode == "Target volatility":
            target_volatility = st.number_input(
                "Target volatility (annual)", value=0.10, step=0.005, format="%.4f",
                key="target_volatility",
            )
        else:
            risk_aversion = st.slider("Risk aversion λ", 0.1, 20.0, 2.5, key="risk_aversion")
    if optimizer_name == "cvar":
        cvar_alpha = st.slider(
            "CVaR tail prob α", 0.01, 0.20, 0.05, 0.01,
            key="cvar_alpha",
            help="0.05 ⇒ 95% CVaR.",
        )
        target_return = st.number_input(
            "Target return (optional)", value=0.0, step=0.005, format="%.4f",
            key="target_return_cvar",
        )
        target_return = None if target_return == 0.0 else target_return

    st.divider()
    st.header("4 · Frontier")
    build_frontier = st.checkbox(
        "Build efficient frontier",
        value=True,
        disabled=not ws["frontier"]["enabled"],
        help=ws["frontier"]["tooltip"],
    )
    n_frontier_points = st.slider(
        "Frontier points", 5, 100, 25,
        disabled=not ws["frontier"]["enabled"],
    )
    if not ws["frontier"]["enabled"]:
        build_frontier = False


# ---------------------------------------------------------------------------
# Currency conversion (apply once, before returns)
# ---------------------------------------------------------------------------

if "asset_currency" not in st.session_state or set(st.session_state.asset_currency) != set(prices.columns):
    st.session_state.asset_currency = {a: base_currency for a in prices.columns}

unique_currencies = {st.session_state.asset_currency.get(a, base_currency) for a in prices.columns}
if unique_currencies != {base_currency}:
    try:
        prices = convert_prices_to_base(
            prices,
            asset_currency=st.session_state.asset_currency,
            base=base_currency,
        )
        with st.sidebar:
            st.caption(
                f"FX-converted {len(prices.columns)} series → {base_currency} via FRED."
            )
    except FXError as exc:
        st.sidebar.error(f"FX conversion failed: {exc}")
        st.stop()


# ---------------------------------------------------------------------------
# Returns + tabs
# ---------------------------------------------------------------------------

returns = prices_to_returns(prices)

st.markdown("---")
(
    tab_overview,
    tab_assets,
    tab_constraints,
    tab_compare,
    tab_whatif,
    tab_optimize,
    tab_report,
) = st.tabs(
    [
        "🌐 Overview",
        "📊 Assets",
        "⚙️ Expected returns & constraints",
        "🆚 Compare",
        "🎚️ What-if",
        "🚀 Optimize",
        "📤 Report",
    ]
)


if st.session_state.scenario_load_warning:
    st.warning(st.session_state.scenario_load_warning)
    st.session_state.scenario_load_warning = None


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
                "Currency": [
                    st.session_state.asset_currency.get(a, base_currency) for a in returns.columns
                ],
            },
            index=returns.columns,
        )
    elif "Currency" not in st.session_state.config_table.columns:
        st.session_state.config_table["Currency"] = [
            st.session_state.asset_currency.get(a, base_currency) for a in returns.columns
        ]

    edited = st.data_editor(
        st.session_state.config_table,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Expected Return": st.column_config.NumberColumn(format="%.4f"),
            "Min Weight": st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.01, format="%.2f"),
            "Max Weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.5, step=0.01, format="%.2f"),
            "Group": st.column_config.TextColumn(),
            "Currency": st.column_config.SelectboxColumn(
                "Currency",
                options=supported_currencies(),
                help="ISO code of the currency the price series is quoted in.",
            ),
        },
    )
    st.session_state.config_table = edited
    st.session_state.asset_currency = {
        a: str(edited.loc[a, "Currency"]) for a in returns.columns
    }
    st.caption(
        f"Currencies set per asset; non-{base_currency} series are converted "
        "via FRED FX rates the next time prices are loaded."
    )

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
        currencies=dict(st.session_state.asset_currency),
        base_currency=base_currency,
        periods_per_year=int(periods_per_year),
        covariance_method=cov_method,
        ewma_lambda=float(ewma_lambda),
        optimizer=spec,
    )


# ---------------------------------------------------------------------------
# Cached scenario solver (shared by Compare & What-if)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=32)
def _solve_scenario_cached(signature: str, returns_df: pd.DataFrame):
    """Solve a config given its JSON signature; cached on (signature, returns)."""
    cfg = EngineConfig.from_dict(json.loads(signature))
    return run_engine(returns_df, cfg, build_frontier=False)


def _summarize_run(name: str, run) -> dict:
    w = run.result.weights.sort_values(ascending=False)
    top3 = ", ".join(f"{a}: {v:.1%}" for a, v in w.head(3).items())
    return {
        "Scenario": name,
        "Optimizer": run.config.optimizer.name,
        "Expected Return": run.result.expected_return,
        "Expected Vol": run.result.expected_volatility,
        "Sharpe": run.result.sharpe_ratio,
        "Active": int((w.abs() > 1e-4).sum()),
        "Top 3 holdings": top3,
    }


def _scenario_returns_subset(scn: Scenario, full_returns: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in full_returns.columns if c in scn.config.expected_returns]
    if not keep:
        raise ValueError("Scenario universe disjoint from loaded data.")
    return full_returns[keep]


# ---------------------------------------------------------------------------
# 🆚 Compare
# ---------------------------------------------------------------------------

with tab_compare:
    if not st.session_state.scenarios:
        st.info("Save at least one scenario from the **📚 Scenarios** sidebar block to compare.")
    else:
        names_all = sorted(st.session_state.scenarios.keys())
        default_sel = (
            [st.session_state.active_scenario]
            if st.session_state.active_scenario in names_all
            else names_all[:1]
        )
        chosen = st.multiselect(
            "Scenarios to compare",
            options=names_all,
            default=default_sel,
            max_selections=5,
            help="Up to 5 scenarios at once.",
        )
        if not chosen:
            st.info("Pick one or more scenarios.")
        else:
            runs: dict[str, "object"] = {}
            for n in chosen:
                scn = st.session_state.scenarios[n]
                try:
                    sub = _scenario_returns_subset(scn, returns)
                    runs[n] = _solve_scenario_cached(scenario_signature(scn), sub)
                    coverage = (
                        len([c for c in returns.columns if c in scn.config.expected_returns]),
                        len(scn.config.expected_returns),
                    )
                    if coverage[0] != coverage[1]:
                        st.caption(
                            f"_{n}: covers {coverage[0]}/{coverage[1]} of its assets in the loaded data._"
                        )
                except Exception as exc:
                    st.error(f"{n}: {exc}")

            if runs:
                summary_rows = [_summarize_run(n, r) for n, r in runs.items()]
                summary_df = pd.DataFrame(summary_rows).set_index("Scenario")
                st.dataframe(
                    summary_df.style.format(
                        {
                            "Expected Return": "{:.2%}",
                            "Expected Vol": "{:.2%}",
                            "Sharpe": "{:.3f}",
                        }
                    ),
                    use_container_width=True,
                )

                weights_df = pd.DataFrame(
                    {n: r.result.weights for n, r in runs.items()}
                ).fillna(0.0)
                st.plotly_chart(
                    plot_portfolio_composition(weights_df, title="Weights by scenario"),
                    use_container_width=True,
                )

                grouped = plot_risk_contributions(weights_df)
                grouped.update_layout(title="Per-asset weights")
                st.plotly_chart(grouped, use_container_width=True)

                bt_frames = {}
                for n, r in runs.items():
                    bt = r.backtest_returns()["portfolio"]
                    bt_frames[n] = bt
                bt_df = pd.concat(bt_frames, axis=1)
                st.plotly_chart(
                    plot_wealth_index(bt_df, "Backtest comparison"),
                    use_container_width=True,
                )


# ---------------------------------------------------------------------------
# 🎚️ What-if
# ---------------------------------------------------------------------------

with tab_whatif:
    if not st.session_state.scenarios:
        st.info("Save a scenario first; What-if needs an anchor.")
    else:
        names_all = sorted(st.session_state.scenarios.keys())
        default_idx = (
            names_all.index(st.session_state.active_scenario)
            if st.session_state.active_scenario in names_all
            else 0
        )
        anchor_name = st.selectbox(
            "Anchor scenario",
            options=names_all,
            index=default_idx,
            key="whatif_anchor",
        )

        # Reset overrides on anchor change.
        if st.session_state.get("whatif_last_anchor") != anchor_name:
            st.session_state.whatif_overrides = {}
            st.session_state.whatif_extra = {}
            st.session_state.whatif_run = None
            st.session_state.whatif_error = None
            st.session_state.whatif_last_anchor = anchor_name

        anchor_scn = st.session_state.scenarios[anchor_name]
        anchor_cfg = anchor_scn.config
        anchor_assets = list(anchor_cfg.expected_returns.keys())
        n_assets = len(anchor_assets)
        is_slow = anchor_cfg.optimizer.name == "cvar" or n_assets > 25

        if is_slow:
            st.info(
                f"Live re-solve disabled (optimizer='{anchor_cfg.optimizer.name}', "
                f"{n_assets} assets > 25). Drag sliders, then press **Recompute**."
            )

        st.markdown("**Per-asset weight bounds**")
        cols = st.columns(2)
        overrides: dict[str, tuple[float, float]] = dict(st.session_state.whatif_overrides)
        for i, a in enumerate(anchor_assets):
            lo0, hi0 = anchor_cfg.bounds.get(a, [0.0, 1.0])
            current = overrides.get(a, (float(lo0), float(hi0)))
            with cols[i % 2]:
                lo, hi = st.slider(
                    f"{a}",
                    min_value=-1.0, max_value=1.5, step=0.01,
                    value=(float(current[0]), float(current[1])),
                    key=f"whatif_bnd_{anchor_name}_{a}",
                )
            overrides[a] = (lo, hi)
        st.session_state.whatif_overrides = overrides

        st.markdown("**Optimizer extras**")
        extras: dict[str, float] = dict(st.session_state.whatif_extra)
        opt_name = anchor_cfg.optimizer.name
        if opt_name == "mean_variance":
            mode_choices = ["Target return", "Target volatility", "Utility"]
            if anchor_cfg.optimizer.target_return is not None:
                default_mode = "Target return"
            elif anchor_cfg.optimizer.target_volatility is not None:
                default_mode = "Target volatility"
            else:
                default_mode = "Utility"
            wf_mode = st.radio(
                "Mode", mode_choices,
                index=mode_choices.index(default_mode),
                horizontal=True,
                key="whatif_mv_mode",
            )
            if wf_mode == "Target return":
                tr = st.number_input(
                    "Target return (annual)",
                    value=float(anchor_cfg.optimizer.target_return or 0.07),
                    step=0.005, format="%.4f",
                    key="whatif_target_return",
                )
                extras = {"target_return": float(tr), "target_volatility": None}
            elif wf_mode == "Target volatility":
                tv = st.number_input(
                    "Target volatility (annual)",
                    value=float(anchor_cfg.optimizer.target_volatility or 0.10),
                    step=0.005, format="%.4f",
                    key="whatif_target_vol",
                )
                extras = {"target_return": None, "target_volatility": float(tv)}
            else:
                ra = st.slider(
                    "Risk aversion λ", 0.1, 20.0,
                    float(anchor_cfg.optimizer.risk_aversion or 2.5),
                    key="whatif_risk_aversion",
                )
                extras = {
                    "target_return": None,
                    "target_volatility": None,
                    "risk_aversion": float(ra),
                }
        elif opt_name == "cvar":
            ca = st.slider(
                "CVaR tail prob α", 0.01, 0.20,
                float(anchor_cfg.optimizer.cvar_alpha or 0.05), 0.01,
                key="whatif_cvar_alpha",
            )
            extras = {"cvar_alpha": float(ca)}
        st.session_state.whatif_extra = extras

        # Build the live config from anchor + overrides + extras.
        def _live_config():
            cfg_dict = anchor_cfg.to_dict()
            cfg_dict["bounds"] = {
                a: list(st.session_state.whatif_overrides.get(a, anchor_cfg.bounds.get(a, [0.0, 1.0])))
                for a in anchor_assets
            }
            opt = dict(cfg_dict["optimizer"])
            for k, v in st.session_state.whatif_extra.items():
                opt[k] = v
            cfg_dict["optimizer"] = opt
            return EngineConfig.from_dict(cfg_dict)

        if is_slow:
            should_solve = st.button("Recompute", type="primary", key="whatif_recompute")
        else:
            should_solve = True

        if should_solve:
            try:
                live_cfg = _live_config()
                sub = returns[[c for c in returns.columns if c in anchor_assets]]
                st.session_state.whatif_run = _solve_scenario_cached(
                    config_signature(live_cfg), sub
                )
                st.session_state.whatif_error = None
            except Exception as exc:
                st.session_state.whatif_run = None
                st.session_state.whatif_error = str(exc)

        wf_run = st.session_state.whatif_run
        if st.session_state.whatif_error:
            st.error(f"Solver: {st.session_state.whatif_error}")
            if st.button("Reset to anchor", key="whatif_reset"):
                st.session_state.whatif_overrides = {}
                st.session_state.whatif_extra = {}
                st.session_state.whatif_error = None
                st.rerun()
        elif wf_run is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected Return", f"{wf_run.result.expected_return:.2%}")
            c2.metric("Expected Vol", f"{wf_run.result.expected_volatility:.2%}")
            c3.metric("Sharpe", f"{wf_run.result.sharpe_ratio:.2f}")
            c4.metric("Active", f"{int((wf_run.result.weights.abs() > 1e-4).sum())}")

            try:
                anchor_run = _solve_scenario_cached(
                    scenario_signature(anchor_scn),
                    _scenario_returns_subset(anchor_scn, returns),
                )
                weights_df = pd.DataFrame(
                    {"Anchor": anchor_run.result.weights, "What-if": wf_run.result.weights}
                ).fillna(0.0)
            except Exception:
                weights_df = pd.DataFrame({"What-if": wf_run.result.weights}).fillna(0.0)
            st.plotly_chart(
                plot_portfolio_composition(weights_df, title="Anchor vs What-if"),
                use_container_width=True,
            )

            if st.button("Save these as new scenario", key="whatif_save"):
                st.session_state["whatif_save_pending"] = True

            if st.session_state.get("whatif_save_pending"):
                new_name = st.text_input(
                    "Scenario name",
                    key="whatif_save_name",
                    placeholder="e.g. Anchor + tighter equities",
                )
                cA, cB = st.columns(2)
                if cA.button("Save", key="whatif_save_confirm", type="primary"):
                    if not new_name:
                        st.error("Name is required.")
                    elif new_name in st.session_state.scenarios:
                        st.error(f"Scenario {new_name!r} already exists.")
                    else:
                        live_cfg = _live_config()
                        st.session_state.scenarios[new_name] = Scenario(
                            name=new_name,
                            config=live_cfg,
                            notes=f"Forked from {anchor_name}",
                            created_at=now_iso(),
                            updated_at=now_iso(),
                        )
                        st.session_state.active_scenario = new_name
                        st.session_state["whatif_save_pending"] = False
                        st.success(f"Saved as {new_name!r}.")
                        st.rerun()
                if cB.button("Cancel", key="whatif_save_cancel"):
                    st.session_state["whatif_save_pending"] = False
                    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — 📚 Scenarios block (rendered after _build_config exists)
# ---------------------------------------------------------------------------

with st.sidebar.expander("📚 Scenarios", expanded=False):
    names_in_state = sorted(st.session_state.scenarios.keys())
    default_idx = 0
    sel_options = ["—"] + names_in_state
    if st.session_state.active_scenario in names_in_state:
        default_idx = sel_options.index(st.session_state.active_scenario)
    selected = st.selectbox(
        "Active scenario",
        options=sel_options,
        index=default_idx,
        key="scn_select",
    )
    has_selection = selected != "—"

    new_name = st.text_input("Name", key="scn_new_name").strip()
    notes = st.text_area(
        "Notes (optional)",
        key="scn_notes",
        height=70,
        max_chars=NOTES_MAX_LEN,
    )
    cA, cB = st.columns(2)
    save_clicked = cA.button("Save", use_container_width=True, key="scn_save")
    update_clicked = cB.button(
        "Update", disabled=not has_selection, use_container_width=True, key="scn_update"
    )
    cC, cD = st.columns(2)
    load_clicked = cC.button(
        "Load", disabled=not has_selection, use_container_width=True, key="scn_load"
    )
    delete_clicked = cD.button(
        "Delete", disabled=not has_selection, use_container_width=True, key="scn_delete"
    )

    rename_to = st.text_input(
        "Rename to",
        key="scn_rename_to",
        disabled=not has_selection,
    ).strip()
    rename_clicked = st.button(
        "Rename",
        disabled=(not has_selection) or (not rename_to),
        key="scn_rename",
    )

    st.divider()
    st.download_button(
        "⬇ Download all (YAML)",
        data=dump_scenarios_yaml(st.session_state.scenarios) if st.session_state.scenarios else "",
        file_name="scenarios.yaml",
        mime="text/yaml",
        disabled=not st.session_state.scenarios,
    )
    upl = st.file_uploader(
        "⬆ Upload scenarios YAML", type=["yaml", "yml"], key="scn_upload"
    )
    merge_mode = st.radio(
        "On name collision",
        options=["Skip", "Overwrite", "Suffix"],
        horizontal=True,
        key="scn_merge_mode",
    )

    # Handlers
    if save_clicked:
        if not new_name:
            st.error("Name is required.")
        elif new_name in st.session_state.scenarios:
            st.error(f"Scenario {new_name!r} already exists. Use Update or pick another name.")
        else:
            cfg_now = _build_config()
            ts = now_iso()
            st.session_state.scenarios[new_name] = Scenario(
                name=new_name, config=cfg_now, notes=notes,
                created_at=ts, updated_at=ts,
            )
            st.session_state.active_scenario = new_name
            st.success(f"Saved scenario {new_name!r}.")
            st.rerun()

    if update_clicked and has_selection:
        cfg_now = _build_config()
        prev = st.session_state.scenarios[selected]
        st.session_state.scenarios[selected] = Scenario(
            name=selected,
            config=cfg_now,
            notes=notes or prev.notes,
            created_at=prev.created_at or now_iso(),
            updated_at=now_iso(),
        )
        st.session_state.last_run_by_scenario.pop(selected, None)
        st.success(f"Updated {selected!r}.")
        st.rerun()

    if load_clicked and has_selection:
        target_cfg = st.session_state.scenarios[selected].config
        target_assets = set(target_cfg.expected_returns)
        loaded_assets = set(returns.columns)
        missing = sorted(target_assets - loaded_assets)
        if missing:
            # Drop missing assets from the scenario before applying (Decision #1).
            kept = [a for a in target_cfg.expected_returns if a in loaded_assets]
            if not kept:
                st.error(
                    f"Cannot load {selected!r}: none of its assets are in the loaded data."
                )
            else:
                trimmed_dict = target_cfg.to_dict()
                trimmed_dict["expected_returns"] = {
                    a: trimmed_dict["expected_returns"][a] for a in kept
                }
                trimmed_dict["bounds"] = {
                    a: trimmed_dict["bounds"].get(a, [0.0, 1.0]) for a in kept
                }
                trimmed_dict["groups"] = {
                    a: trimmed_dict["groups"].get(a, "Other") for a in kept
                }
                trimmed_dict["currencies"] = {
                    a: trimmed_dict["currencies"].get(a, trimmed_dict["base_currency"])
                    for a in kept
                }
                # Stash a temporary trimmed scenario keyed by name → reload normally.
                trimmed_scn = Scenario(
                    name=selected,
                    config=EngineConfig.from_dict(trimmed_dict),
                    notes=st.session_state.scenarios[selected].notes,
                    created_at=st.session_state.scenarios[selected].created_at,
                    updated_at=st.session_state.scenarios[selected].updated_at,
                )
                st.session_state.scenarios[selected] = trimmed_scn
                st.session_state.scenario_load_warning = (
                    f"Loaded {selected!r}; dropped {len(missing)} missing asset(s): "
                    + ", ".join(missing)
                )
                st.session_state.pending_scenario_load = selected
                st.rerun()
        else:
            st.session_state.pending_scenario_load = selected
            st.rerun()

    if delete_clicked and has_selection:
        st.session_state.scenarios = _delete_scenario(
            st.session_state.scenarios, selected
        )
        st.session_state.last_run_by_scenario.pop(selected, None)
        if st.session_state.active_scenario == selected:
            st.session_state.active_scenario = None
        st.success(f"Deleted {selected!r}.")
        st.rerun()

    if rename_clicked and has_selection and rename_to:
        try:
            st.session_state.scenarios = _rename_scenario(
                st.session_state.scenarios, selected, rename_to
            )
            old_run = st.session_state.last_run_by_scenario.pop(selected, None)
            if old_run is not None:
                st.session_state.last_run_by_scenario[rename_to] = old_run
            if st.session_state.active_scenario == selected:
                st.session_state.active_scenario = rename_to
            st.success(f"Renamed to {rename_to!r}.")
            st.rerun()
        except (KeyError, ValueError) as exc:
            st.error(str(exc))

    if upl is not None:
        try:
            text = upl.read().decode("utf-8")
            incoming = load_scenarios_yaml(text)
            applied = 0
            for k, v in incoming.items():
                if k in st.session_state.scenarios:
                    if merge_mode == "Skip":
                        continue
                    if merge_mode == "Overwrite":
                        st.session_state.scenarios[k] = v
                        st.session_state.last_run_by_scenario.pop(k, None)
                        applied += 1
                    else:  # Suffix
                        suffix = 2
                        candidate = f"{k} ({suffix})"
                        while candidate in st.session_state.scenarios:
                            suffix += 1
                            candidate = f"{k} ({suffix})"
                        new_v = Scenario(
                            name=candidate,
                            config=v.config,
                            notes=v.notes,
                            created_at=v.created_at,
                            updated_at=v.updated_at,
                        )
                        st.session_state.scenarios[candidate] = new_v
                        applied += 1
                else:
                    st.session_state.scenarios[k] = v
                    applied += 1
            st.success(f"Uploaded {applied} scenario(s).")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")


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
