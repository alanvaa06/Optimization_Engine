"""Streamlit UI for the optimization engine.

Run with::

    streamlit run app/streamlit_app.py

The app is driven by the same `EngineConfig` machinery as the CLI, so
anything that works here also works headless.

The flow is deliberately linear — load data, check it, choose a method, set
assumptions, state constraints, solve, then interrogate the result — and each
step surfaces what could make the next one wrong: data-quality problems before
estimation, constraint feasibility before solving, and the gap between the
in-sample and walk-forward track records before anyone believes the backtest.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make src/ and this directory importable when running
# ``streamlit run app/streamlit_app.py``.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
for path in (str(SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from components import (  # noqa: E402
    format_table,
    metric_row,
    num,
    pct,
    render_assumptions,
    render_compliance,
    render_covariance_diagnostics,
    render_data_quality,
    render_feasibility,
    render_frontier_health,
    render_method_card,
    render_portfolio_diagnostics,
    render_projection_distance,
)

from optimization_engine.analytics.performance import rolling_metrics, summary_stats  # noqa: E402
from optimization_engine.analytics.relative import (  # noqa: E402
    active_share,
    summary_relative,
)
from optimization_engine.analytics.risk import drawdown_table  # noqa: E402
from optimization_engine.config import EngineConfig, OptimizerSpec  # noqa: E402
from optimization_engine.data.covariance import (  # noqa: E402
    COVARIANCE_DESCRIPTIONS,
    EXPECTED_RETURN_DESCRIPTIONS,
    covariance_matrix,
)
from optimization_engine.data.fx import (  # noqa: E402
    FXError,
    convert_prices_to_base,
    supported_currencies,
)
from optimization_engine.data.loader import (  # noqa: E402
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.data.quality import align_panel, analyze_prices  # noqa: E402
from optimization_engine.data.yahoo import (  # noqa: E402
    YahooFinanceError,
    load_prices_yahoo,
)
from optimization_engine.engine import run_engine  # noqa: E402
from optimization_engine.optimizers.factory import (  # noqa: E402
    available_optimizers,
    constraints_from_config,
    effective_expected_returns,
)
from optimization_engine.optimizers.feasibility import analyze_feasibility  # noqa: E402
from optimization_engine.optimizers.requirements import requirements_for  # noqa: E402
from optimization_engine.reporting.exporters import run_sheets  # noqa: E402
from optimization_engine.reporting.plots import (  # noqa: E402
    plot_correlation_heatmap,
    plot_drawdown,
    plot_efficient_frontier,
    plot_portfolio_composition,
    plot_return_distribution,
    plot_risk_contributions,
    plot_rolling_metrics,
    plot_walk_forward_comparison,
    plot_wealth_index,
    plot_weight_evolution,
    plot_weight_vs_risk,
    plot_weights_bar,
)
from optimization_engine.scenarios import (  # noqa: E402
    NOTES_MAX_LEN,
    Scenario,
    config_signature,
    dump_scenarios_yaml,
    load_scenarios_yaml,
    now_iso,
    scenario_signature,
)
from optimization_engine.scenarios import (
    delete_scenario as _delete_scenario,
)
from optimization_engine.scenarios import (
    rename_scenario as _rename_scenario,
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
    .small-muted { color: #6b7280; font-size: 0.85rem; }
    .step-done { color: #16a34a; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_DEFAULT_STATE = {
    "scenarios": {},
    "active_scenario": None,
    "last_run_by_scenario": {},
    "scenario_load_warning": None,
    "last_run": None,
    "last_error": None,
    "walk_forward": None,
}
for key, default in _DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


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
    if getattr(cfg.optimizer, "bl_views", None) and isinstance(
        cfg.optimizer.bl_views, dict
    ):
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
    st.session_state["long_only"] = bool(_cfg.long_only)
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
# Cached data helpers
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


@st.cache_data(show_spinner=False, max_entries=16)
def _frame_hash(df: pd.DataFrame) -> str:
    return pd.util.hash_pandas_object(df, index=True).values.tobytes().hex()


@st.cache_data(show_spinner=False, max_entries=16)
def _covariance_cached(
    returns_hash: str,
    method: str,
    ewma_lambda: float,
    periods_per_year: int,
    annualize: bool,
    _returns: pd.DataFrame,
) -> pd.DataFrame:
    return covariance_matrix(
        _returns, method=method, ewma_lambda=ewma_lambda,
        periods_per_year=periods_per_year, annualize=annualize,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _historical_mu_cached(
    returns_hash: str,
    periods_per_year: int,
    _returns: pd.DataFrame,
) -> pd.Series:
    return (1 + _returns).prod() ** (periods_per_year / len(_returns)) - 1


@st.cache_data(show_spinner=False, max_entries=8)
def _quality_cached(prices_hash: str, periods_per_year: int, _prices: pd.DataFrame):
    return analyze_prices(_prices, periods_per_year=periods_per_year)


def _load_uploaded(file: io.BytesIO, sheet: str) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(file, sheet_name=sheet, index_col=0, parse_dates=True)
    if name.endswith(".csv"):
        return pd.read_csv(file, index_col=0, parse_dates=True)
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    raise ValueError(f"Unsupported file: {file.name}")


# ---------------------------------------------------------------------------
# Sidebar — step 1: data
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
        raw_prices = _load_sample(years * 252)
    elif data_source == "Upload file":
        uploaded = st.file_uploader(
            "Price file (Excel/CSV/Parquet)",
            type=["xlsx", "xls", "xlsm", "csv", "parquet"],
        )
        sheet = st.text_input("Sheet name (Excel)", value="Precios")
        if uploaded is None:
            st.info("Upload a file to continue, or switch to Sample data.")
            st.stop()
        raw_prices = _load_uploaded(uploaded, sheet)
        raw_prices.index = pd.to_datetime(raw_prices.index)
        raw_prices = raw_prices.sort_index().dropna(how="all")
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
            tickers_tuple, yahoo_period, yahoo_start, yahoo_end, yahoo_interval,
        )

        try:
            raw_prices = yahoo_prices_for_rerun(
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
        if raw_prices is None:
            st.info("Set tickers and click **Fetch from Yahoo** to download prices.")
            st.stop()

    selected_assets = st.multiselect(
        "Universe (assets to include)",
        options=list(raw_prices.columns),
        default=list(raw_prices.columns),
    )
    if not selected_assets:
        st.warning("Select at least one asset.")
        st.stop()
    raw_prices = raw_prices[selected_assets]

    st.caption(f"{raw_prices.shape[0]:,} rows × {raw_prices.shape[1]} assets loaded.")

    missing_alignment = st.selectbox(
        "Missing data",
        options=["common", "ffill", "drop_assets"],
        index=0,
        format_func=lambda m: {
            "common": "Use only dates where every asset is present",
            "ffill": "Carry the last price across short gaps, then align",
            "drop_assets": "Drop short series, then align",
        }[m],
        help=(
            "How to reconcile assets with different histories. Every choice "
            "changes the sample the covariance is estimated on, so it is "
            "made explicitly rather than silently."
        ),
    )
    max_ffill = (
        st.number_input(
            "Max gap to fill (periods)", min_value=1, max_value=60, value=5
        )
        if missing_alignment == "ffill"
        else 5
    )

# Alignment happens outside the sidebar so its log can be shown in the tabs.
prices, alignment_actions = align_panel(
    raw_prices, method=missing_alignment, max_ffill=int(max_ffill)
)
if prices.empty or prices.shape[1] == 0:
    st.error(
        "No usable data after alignment. The assets' histories may not "
        "overlap — try 'Carry the last price across short gaps' or shorten "
        "the universe."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — step 2: currency
# ---------------------------------------------------------------------------

with st.sidebar:
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

if "asset_currency" not in st.session_state or set(st.session_state.asset_currency) != set(prices.columns):
    st.session_state.asset_currency = {a: base_currency for a in prices.columns}

unique_currencies = {
    st.session_state.asset_currency.get(a, base_currency) for a in prices.columns
}
fx_note = None
if unique_currencies != {base_currency}:
    try:
        prices = convert_prices_to_base(
            prices,
            asset_currency=st.session_state.asset_currency,
            base=base_currency,
        )
        fx_note = (
            f"FX-converted {len(prices.columns)} series → {base_currency} via FRED."
        )
    except FXError as exc:
        st.sidebar.error(f"FX conversion failed: {exc}")
        st.stop()

returns = prices_to_returns(prices).dropna(how="any")
if returns.empty:
    st.error("The price panel produced no usable returns.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — step 3: method
# ---------------------------------------------------------------------------

_METHOD_ORDER = [
    "mean_variance", "min_variance", "max_sharpe", "risk_parity",
    "hrp", "black_litterman", "cvar", "max_diversification",
    "inverse_vol", "equal_weight",
]
_methods = [m for m in _METHOD_ORDER if m in available_optimizers()]
_methods += [m for m in available_optimizers() if m not in _methods]

with st.sidebar:
    st.divider()
    st.header("3 · Method")
    optimizer_name = st.selectbox(
        "Optimizer",
        options=_methods,
        index=_methods.index("mean_variance"),
        format_func=lambda m: requirements_for(m).display_name,
        key="optimizer_name",
        help="Ordered from most to least assumption-heavy.",
    )
    req = requirements_for(optimizer_name)
    st.caption(req.summary)
    ws = derive_widget_state(optimizer_name)

    st.divider()
    st.header("4 · Assumptions")
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0, max_value=0.30,
        value=0.04, step=0.005, format="%.4f",
        key="risk_free_rate",
        disabled=not ws["risk_free_rate"]["enabled"],
        help=ws["risk_free_rate"]["tooltip"]
        or "Used for Sharpe, the tangency portfolio, and CAPM/Black-Litterman.",
    )
    periods_per_year = st.number_input(
        "Periods per year", min_value=1, max_value=365, value=252,
        key="periods_per_year",
        help="252 for daily data, 52 for weekly, 12 for monthly.",
    )
    cov_options = ["ledoit_wolf", "oas", "sample", "ewma", "semi", "shrink"]
    cov_method = st.selectbox(
        "Covariance estimator",
        options=cov_options,
        index=0,
        key="cov_method",
        disabled=not ws["cov_method"]["enabled"],
        help=ws["cov_method"]["tooltip"] or "Shrinkage estimators first.",
    )
    if ws["cov_method"]["enabled"]:
        st.caption(COVARIANCE_DESCRIPTIONS.get(cov_method, ""))
    ewma_lambda = (
        st.slider(
            "EWMA λ", 0.80, 0.999, 0.94, 0.005,
            key="ewma_lambda",
            help="Effective sample is about 1/(1−λ) observations.",
        )
        if cov_method == "ewma" and ws["cov_method"]["enabled"]
        else 0.94
    )

    st.divider()
    st.header("5 · Objective")
    target_return: float | None = None
    target_volatility: float | None = None
    risk_aversion = 1.0
    cvar_alpha = 0.05
    risk_budget: dict[str, float] | None = None

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
        mode = st.radio("Mode", modes, horizontal=True, key="mv_mode")
        if mode == "Target return":
            target_return = st.number_input(
                "Target return (annual)", value=0.07, step=0.005, format="%.4f",
                key="target_return",
                help="Minimize risk subject to hitting exactly this return.",
            )
            if optimizer_name == "black_litterman":
                st.caption(
                    "Black-Litterman targets its **equilibrium posterior**, "
                    "which usually sits well below historical means. The "
                    "constraints tab reports the range this method can "
                    "actually reach."
                )
        elif mode == "Target volatility":
            target_volatility = st.number_input(
                "Target volatility (annual)", value=0.10, step=0.005, format="%.4f",
                key="target_volatility",
                help="Maximize return subject to staying under this volatility.",
            )
        else:
            risk_aversion = st.slider(
                "Risk aversion λ", 0.1, 20.0, 2.5, key="risk_aversion",
                help="Maximize μ'w − λ·w'Σw. Higher λ means a safer portfolio.",
            )
    elif not ws["risk_aversion"]["enabled"] and optimizer_name != "cvar":
        st.caption(
            f"{req.display_name} has no objective to tune — it is fully "
            "determined by the covariance and the constraints."
        )

    if optimizer_name == "cvar":
        cvar_alpha = st.slider(
            "CVaR tail probability α", 0.01, 0.20, 0.05, 0.01,
            key="cvar_alpha",
            help="0.05 ⇒ the 95% CVaR: the average of the worst 5% of periods.",
        )
        n_tail = int(cvar_alpha * len(returns))
        st.caption(
            f"About {n_tail} of {len(returns):,} observations drive the estimate."
            + ("  ⚠️ That is very few." if n_tail < 10 else "")
        )
        _tr = st.number_input(
            "Minimum return (optional, 0 = none)",
            value=0.0, step=0.005, format="%.4f", key="target_return_cvar",
        )
        target_return = None if _tr == 0.0 else _tr

    st.divider()
    st.header("6 · Exposure")
    long_only = st.checkbox(
        "Long only", value=True, key="long_only",
        help="Off allows short positions, subject to each asset's minimum weight.",
    )
    leverage_cap: float | None = None
    if not long_only:
        leverage_cap = st.slider(
            "Gross-exposure cap (Σ|w|)", 1.0, 3.0, 1.5, 0.1,
            help="1.5 means 125% long against 25% short, or any equivalent mix.",
        )
    use_turnover = st.checkbox(
        "Turnover budget",
        value=False,
        help=(
            "Limit how much of the book may change versus a previous "
            "allocation. Honoured by mean-variance and CVaR; other methods "
            "warn instead of silently ignoring it."
        ),
    )
    turnover_limit: float | None = None
    if use_turnover:
        turnover_limit = st.slider(
            "Max one-way turnover", 0.01, 2.0, 0.20, 0.01,
            help="0.20 means at most 20% of the portfolio changes hands.",
        )
        if not req.supports_turnover:
            st.warning(
                f"{req.display_name} cannot enforce a turnover budget. "
                "Use mean-variance or CVaR to make it bind."
            )

    st.divider()
    st.header("7 · Frontier")
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
# Tabs
# ---------------------------------------------------------------------------

st.markdown("---")
(
    tab_overview,
    tab_assets,
    tab_constraints,
    tab_optimize,
    tab_backtest,
    tab_compare,
    tab_whatif,
    tab_report,
) = st.tabs(
    [
        "🌐 Data",
        "📊 Assets",
        "⚙️ Assumptions & constraints",
        "🚀 Optimize",
        "📉 Backtest",
        "🆚 Compare",
        "🎚️ What-if",
        "📤 Report",
    ]
)

if st.session_state.scenario_load_warning:
    st.warning(st.session_state.scenario_load_warning)
    st.session_state.scenario_load_warning = None


quality = _quality_cached(_frame_hash(raw_prices), int(periods_per_year), raw_prices)


# ---------------------------------------------------------------------------
# 🌐 Data
# ---------------------------------------------------------------------------

with tab_overview:
    metric_row(
        [
            ("Assets", str(returns.shape[1]), None),
            ("Periods", f"{returns.shape[0]:,}", None),
            ("Start", str(returns.index.min().date()), None),
            ("End", str(returns.index.max().date()), None),
        ]
    )

    render_data_quality(quality)
    if alignment_actions:
        with st.expander("What alignment did to the panel", expanded=False):
            for action in alignment_actions:
                st.markdown(f"- {action}")
    if fx_note:
        st.caption(fx_note)

    st.subheader("Cumulative returns")
    st.plotly_chart(plot_wealth_index(returns), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Correlation")
        st.plotly_chart(
            plot_correlation_heatmap(returns.corr()), width="stretch"
        )
    with right:
        st.subheader("Coverage")
        st.caption(
            "Observations available per asset, before alignment. Wide "
            "differences mean the common sample is much shorter than the "
            "longest series."
        )
        coverage = quality.per_asset[
            ["observations", "first_date", "last_date", "annualized_vol"]
        ].copy()
        coverage["annualized_vol"] = coverage["annualized_vol"].map(
            lambda v: f"{v:.1%}" if pd.notna(v) else "—"
        )
        st.dataframe(coverage, width="stretch")


# ---------------------------------------------------------------------------
# 📊 Assets
# ---------------------------------------------------------------------------

with tab_assets:
    st.subheader("Per-asset statistics")
    extended = st.toggle(
        "Show extended metrics",
        value=False,
        help=(
            "Adds Calmar, Omega, tail ratio, hit rate, the Ulcer index, "
            "drawdown duration, and the probabilistic Sharpe ratio."
        ),
    )
    stats = summary_stats(
        returns,
        periods_per_year=int(periods_per_year),
        riskfree_rate=float(risk_free_rate),
        extended=extended,
    )
    st.dataframe(format_table(stats), width="stretch")
    if extended:
        st.caption(
            "**Prob. Sharpe > 0** is the probability the true Sharpe ratio is "
            "positive, given the sample's length, skew and kurtosis. Below "
            "~95% the headline Sharpe is not statistically distinguishable "
            "from zero."
        )

    st.subheader("Drawdowns")
    sel = st.multiselect(
        "Series to plot",
        options=list(returns.columns),
        default=list(returns.columns[:3]),
    )
    if sel:
        st.plotly_chart(plot_drawdown(returns[sel]), width="stretch")
        focus = st.selectbox("Drawdown episodes for", options=sel, index=0)
        episodes = drawdown_table(returns[focus], top=5)
        if episodes.empty:
            st.caption("No drawdown episodes in this sample.")
        else:
            display = episodes.copy()
            display["max_drawdown"] = display["max_drawdown"].map("{:.2%}".format)
            st.dataframe(
                display[
                    [
                        "peak_date", "trough_date", "recovery_date",
                        "max_drawdown", "decline_periods", "recovery_periods",
                        "recovered",
                    ]
                ],
                width="stretch", hide_index=True,
            )
            st.caption(
                "Depth is only half the story — how long a portfolio stayed "
                "underwater is what decides whether it was actually held."
            )


# ---------------------------------------------------------------------------
# ⚙️ Assumptions & constraints
# ---------------------------------------------------------------------------

with tab_constraints:
    render_method_card(req)
    st.divider()

    historical_mu = _historical_mu_cached(
        _frame_hash(returns), int(periods_per_year), returns,
    )

    if "config_table" not in st.session_state or set(st.session_state.config_table.index) != set(returns.columns):
        st.session_state.config_table = pd.DataFrame(
            {
                "Expected Return": historical_mu.round(4),
                "Min Weight": 0.0,
                "Max Weight": 1.0,
                "Group": "Other",
                "Currency": [
                    st.session_state.asset_currency.get(a, base_currency)
                    for a in returns.columns
                ],
            },
            index=returns.columns,
        )
    elif "Currency" not in st.session_state.config_table.columns:
        st.session_state.config_table["Currency"] = [
            st.session_state.asset_currency.get(a, base_currency)
            for a in returns.columns
        ]

    if ws["expected_returns_method"]["enabled"]:
        st.markdown("**Expected returns**")
        er_options = ["historical_mean", "shrunk_mean", "ema", "capm"]
        er_method = st.radio(
            "Method",
            options=er_options,
            index=er_options.index(
                st.session_state.get("expected_returns_method", "historical_mean")
            )
            if st.session_state.get("expected_returns_method", "historical_mean")
            in er_options
            else 0,
            horizontal=True,
            key="expected_returns_method",
        )
        st.caption(
            EXPECTED_RETURN_DESCRIPTIONS.get(
                "mean" if er_method == "historical_mean" else er_method, ""
            )
        )
        if er_method == "ema":
            st.session_state.ema_span = st.slider(
                "EMA span (periods)",
                min_value=30, max_value=504,
                value=int(st.session_state.get("ema_span", 180)),
                step=10,
                key="ema_span_slider",
            )
        elif er_method == "capm":
            st.session_state.market_return = st.number_input(
                "Market return (annual, optional)",
                value=float(st.session_state.get("market_return") or 0.08),
                step=0.005, format="%.4f",
                key="market_return_input",
            )
            mw_idx = list(returns.columns)
            if (
                "market_weights_table" not in st.session_state
                or set(st.session_state.market_weights_table.index) != set(mw_idx)
            ):
                st.session_state.market_weights_table = pd.DataFrame(
                    {"Market weight": [1.0 / len(mw_idx)] * len(mw_idx)}, index=mw_idx
                )
            st.session_state.market_weights_table = st.data_editor(
                st.session_state.market_weights_table,
                num_rows="fixed",
                column_config={
                    "Market weight": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                    ),
                },
            )

        if st.button("Reset μ to method default", key="reset_mu_btn"):
            from optimization_engine.data.covariance import expected_returns_from_history

            mw_for_capm = (
                pd.Series(st.session_state.market_weights_table["Market weight"])
                if er_method == "capm"
                and "market_weights_table" in st.session_state
                else None
            )
            try:
                seeded = expected_returns_from_history(
                    returns,
                    method=("mean" if er_method == "historical_mean" else er_method),
                    periods_per_year=int(periods_per_year),
                    span=int(st.session_state.get("ema_span", 180)),
                    market_return=float(st.session_state.get("market_return") or 0.0) or None,
                    risk_free_rate=float(risk_free_rate),
                    market_weights=mw_for_capm,
                    cov_matrix=_covariance_cached(
                        _frame_hash(returns), cov_method, float(ewma_lambda),
                        int(periods_per_year), True, returns,
                    ),
                )
                st.session_state.config_table["Expected Return"] = seeded.round(4)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.markdown("**Per-asset expected returns and weight bounds**")
    st.caption(
        "Edit any cell. Min/Max are weight bounds; Group drives the "
        "asset-class constraints below."
    )
    if ws["soft_bounds_caption"]["enabled"]:
        st.info(req.bounds_note)

    edited = st.data_editor(
        st.session_state.config_table,
        width="stretch",
        num_rows="fixed",
        column_config={
            "Expected Return": st.column_config.NumberColumn(
                format="%.4f",
                disabled=not ws["expected_returns_column"]["enabled"],
                help=ws["expected_returns_column"]["tooltip"],
            ),
            "Min Weight": st.column_config.NumberColumn(
                min_value=-1.0, max_value=1.0, step=0.01, format="%.2f"
            ),
            "Max Weight": st.column_config.NumberColumn(
                min_value=0.0, max_value=1.5, step=0.01, format="%.2f"
            ),
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

    if ws["group_bounds"]["enabled"]:
        st.markdown("**Group (asset-class) constraints**")
        unique_groups = sorted(edited["Group"].dropna().unique().tolist())
        if unique_groups:
            gb_default = pd.DataFrame(
                {"Min Weight": 0.0, "Max Weight": 1.0}, index=unique_groups
            )
            if "group_bounds" not in st.session_state or list(st.session_state.group_bounds.index) != unique_groups:
                st.session_state.group_bounds = gb_default
            st.session_state.group_bounds = st.data_editor(
                st.session_state.group_bounds,
                width="stretch",
                num_rows="fixed",
                column_config={
                    "Min Weight": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.5, step=0.01, format="%.2f"
                    ),
                    "Max Weight": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.5, step=0.01, format="%.2f"
                    ),
                },
            )
    else:
        st.caption(
            f"_{req.display_name} does not enforce group bounds — editor hidden._"
        )

    if optimizer_name == "risk_parity":
        st.markdown("**Risk budgets** (each asset's target share of total risk)")
        if "risk_budget" not in st.session_state or set(st.session_state.risk_budget.index) != set(returns.columns):
            st.session_state.risk_budget = pd.DataFrame(
                {"Risk Budget": 1.0 / len(returns.columns)}, index=returns.columns
            )
        st.session_state.risk_budget = st.data_editor(
            st.session_state.risk_budget,
            width="stretch",
            num_rows="fixed",
            column_config={
                "Risk Budget": st.column_config.NumberColumn(
                    min_value=0.001, max_value=1.0, step=0.01, format="%.3f",
                    help="Must be strictly positive for every asset.",
                ),
            },
        )
        budget_total = float(st.session_state.risk_budget["Risk Budget"].sum())
        if abs(budget_total - 1.0) > 1e-6:
            st.caption(
                f"Budgets sum to {budget_total:.3f}; they will be renormalized to 1."
            )
        risk_budget = st.session_state.risk_budget["Risk Budget"].to_dict()

    elif optimizer_name == "black_litterman":
        st.markdown("**Black-Litterman views**")
        st.caption(
            "An **absolute** view names the return you expect from one asset. "
            "A **relative** view names a spread — 'A beats B by 2%' — and "
            "leaves the overall market level alone, which is usually how "
            "opinions are actually held."
        )
        view_kind = st.radio(
            "View type", ["Absolute", "Relative (spread)"], horizontal=True,
            key="bl_view_kind",
        )
        if view_kind == "Absolute":
            if "bl_views" not in st.session_state or set(st.session_state.bl_views.index) != set(returns.columns):
                st.session_state.bl_views = pd.DataFrame(
                    {"View": np.nan, "Confidence (variance)": np.nan},
                    index=returns.columns,
                )
            st.session_state.bl_views = st.data_editor(
                st.session_state.bl_views,
                width="stretch",
                num_rows="fixed",
                column_config={
                    "View": st.column_config.NumberColumn(
                        format="%.4f", help="Annualized expected return."
                    ),
                    "Confidence (variance)": st.column_config.NumberColumn(
                        format="%.6f",
                        help="Variance of the view's error. Blank uses the "
                        "He-Litterman default; smaller means more confident.",
                    ),
                },
            )
        else:
            if "bl_relative_views" not in st.session_state:
                st.session_state.bl_relative_views = pd.DataFrame(
                    {
                        "Outperforms": [returns.columns[0]],
                        "Underperforms": [returns.columns[-1]],
                        "By (annual)": [0.02],
                        "Confidence (variance)": [np.nan],
                    }
                )
            st.session_state.bl_relative_views = st.data_editor(
                st.session_state.bl_relative_views,
                width="stretch",
                num_rows="dynamic",
                column_config={
                    "Outperforms": st.column_config.SelectboxColumn(
                        options=list(returns.columns)
                    ),
                    "Underperforms": st.column_config.SelectboxColumn(
                        options=list(returns.columns)
                    ),
                    "By (annual)": st.column_config.NumberColumn(format="%.4f"),
                    "Confidence (variance)": st.column_config.NumberColumn(
                        format="%.6f"
                    ),
                },
            )

        st.markdown("**Equilibrium prior**")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["bl_tau"] = st.slider(
                "τ (prior uncertainty)", 0.01, 0.5,
                float(st.session_state.get("bl_tau", 0.05)), 0.01,
                key="bl_tau_slider",
                help="Scales the prior covariance. Smaller τ means a firmer prior.",
            )
        with c2:
            st.session_state["bl_calibrate"] = st.checkbox(
                "Imply δ from a market return",
                value=bool(st.session_state.get("bl_calibrate", False)),
                help=(
                    "Back the risk-aversion coefficient out of the market's "
                    "own Sharpe ratio rather than guessing 2.5."
                ),
            )
            if st.session_state["bl_calibrate"]:
                st.session_state["bl_market_return"] = st.number_input(
                    "Market return (annual)",
                    value=float(st.session_state.get("bl_market_return", 0.07)),
                    step=0.005, format="%.4f",
                )
        if (
            "bl_market_caps_table" not in st.session_state
            or set(st.session_state.bl_market_caps_table.index) != set(returns.columns)
        ):
            st.session_state.bl_market_caps_table = pd.DataFrame(
                {"Market cap weight": [1.0 / len(returns.columns)] * len(returns.columns)},
                index=returns.columns,
            )
        st.session_state.bl_market_caps_table = st.data_editor(
            st.session_state.bl_market_caps_table,
            num_rows="fixed",
            column_config={
                "Market cap weight": st.column_config.NumberColumn(
                    min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                    help="Equal weights → equilibrium under no views.",
                ),
            },
        )

    elif optimizer_name == "hrp":
        st.session_state["hrp_linkage"] = st.selectbox(
            "HRP linkage method",
            options=["single", "average", "complete", "ward"],
            index=["single", "average", "complete", "ward"].index(
                st.session_state.get("hrp_linkage", "single")
            ),
            key="hrp_linkage_select",
            help="Hierarchical clustering linkage rule.",
        )

    if use_turnover:
        st.markdown("**Previous allocation** (what the turnover budget trades from)")
        if (
            "previous_weights_table" not in st.session_state
            or set(st.session_state.previous_weights_table.index) != set(returns.columns)
        ):
            st.session_state.previous_weights_table = pd.DataFrame(
                {"Previous weight": [1.0 / len(returns.columns)] * len(returns.columns)},
                index=returns.columns,
            )
        st.session_state.previous_weights_table = st.data_editor(
            st.session_state.previous_weights_table,
            width="stretch",
            num_rows="fixed",
            column_config={
                "Previous weight": st.column_config.NumberColumn(
                    min_value=-1.0, max_value=1.5, step=0.01, format="%.3f"
                ),
            },
        )


def _build_config() -> EngineConfig:
    """Assemble the EngineConfig from every widget and editable table."""
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

    expected_returns = {
        a: float(table.loc[a, "Expected Return"]) for a in returns.columns
    }

    spec = OptimizerSpec(
        name=optimizer_name,
        target_return=target_return,
        target_volatility=target_volatility,
        risk_free_rate=float(risk_free_rate),
        risk_aversion=float(risk_aversion),
        cvar_alpha=float(cvar_alpha),
        bl_tau=float(st.session_state.get("bl_tau", 0.05)),
        hrp_linkage=str(st.session_state.get("hrp_linkage", "single")),
    )

    if optimizer_name == "risk_parity" and "risk_budget" in st.session_state:
        spec.risk_budget = st.session_state.risk_budget["Risk Budget"].to_dict()

    if optimizer_name == "black_litterman":
        kind = st.session_state.get("bl_view_kind", "Absolute")
        if kind == "Absolute" and "bl_views" in st.session_state:
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
        elif "bl_relative_views" in st.session_state:
            rows = []
            for _, row in st.session_state.bl_relative_views.iterrows():
                over, under = row.get("Outperforms"), row.get("Underperforms")
                spread = row.get("By (annual)")
                if pd.isna(over) or pd.isna(under) or pd.isna(spread):
                    continue
                if over == under:
                    continue
                entry = {
                    "weights": {str(over): 1.0, str(under): -1.0},
                    "expected_return": float(spread),
                    "label": f"{over} > {under}",
                }
                conf = row.get("Confidence (variance)")
                if pd.notna(conf):
                    entry["confidence"] = float(conf)
                rows.append(entry)
            spec.bl_views = rows or None
        if "bl_market_caps_table" in st.session_state:
            spec.bl_market_caps = (
                st.session_state.bl_market_caps_table["Market cap weight"].to_dict()
            )
        if st.session_state.get("bl_calibrate"):
            spec.bl_calibrate_risk_aversion = True
            spec.bl_market_return = float(st.session_state.get("bl_market_return", 0.07))

    market_weights = None
    if (
        st.session_state.get("expected_returns_method") == "capm"
        and "market_weights_table" in st.session_state
    ):
        market_weights = (
            st.session_state.market_weights_table["Market weight"].to_dict()
        )

    previous_weights = None
    if use_turnover and "previous_weights_table" in st.session_state:
        previous_weights = (
            st.session_state.previous_weights_table["Previous weight"].to_dict()
        )

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
        expected_returns_method=str(
            st.session_state.get("expected_returns_method", "historical_mean")
        ),
        ema_span=int(st.session_state.get("ema_span", 180)),
        market_return=(
            float(st.session_state.get("market_return"))
            if st.session_state.get("expected_returns_method") == "capm"
            and st.session_state.get("market_return")
            else None
        ),
        market_weights=market_weights,
        optimizer=spec,
        long_only=bool(long_only),
        leverage=leverage_cap,
        previous_weights=previous_weights,
        turnover_limit=turnover_limit,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def _feasibility_cached(signature: str, returns_hash: str, _returns: pd.DataFrame):
    cfg = EngineConfig.from_dict(json.loads(signature))
    cov = covariance_matrix(
        _returns,
        method=cfg.covariance_method,
        periods_per_year=cfg.periods_per_year,
        ewma_lambda=cfg.ewma_lambda,
    )
    mu = pd.Series(cfg.expected_returns).reindex(_returns.columns).fillna(0.0)
    return analyze_feasibility(
        list(_returns.columns),
        constraints_from_config(cfg),
        expected_returns=effective_expected_returns(cfg, cov, mu),
        cov_matrix=cov,
    )


# The live feasibility panel closes the loop on the constraints tab: the
# analyst sees whether what they just typed can be solved, before solving.
with tab_constraints:
    st.divider()
    st.markdown("**Feasibility check**")
    try:
        live_config = _build_config()
        live_feasibility = _feasibility_cached(
            config_signature(live_config), _frame_hash(returns), returns
        )
        render_feasibility(live_feasibility)
    except Exception as exc:  # a half-edited table should not crash the tab
        live_feasibility = None
        st.info(f"Feasibility check unavailable: {exc}")


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
    diag = run.diagnostics
    return {
        "Scenario": name,
        "Optimizer": run.config.optimizer.name,
        "Expected Return": run.result.expected_return,
        "Expected Vol": run.result.expected_volatility,
        "Sharpe": run.result.sharpe_ratio,
        "Effective N": diag.effective_n if diag else float("nan"),
        "Div. ratio": diag.diversification_ratio if diag else float("nan"),
        "Active": int((w.abs() > 1e-4).sum()),
        "Top 3 holdings": top3,
    }


def _scenario_returns_subset(scn: Scenario, full_returns: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in full_returns.columns if c in scn.config.expected_returns]
    if not keep:
        raise ValueError("Scenario universe disjoint from loaded data.")
    return full_returns[keep]


# ---------------------------------------------------------------------------
# 🚀 Optimize
# ---------------------------------------------------------------------------

with tab_optimize:
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.subheader("Run optimization")
        st.caption(
            f"{req.display_name} · {cov_method} covariance · "
            f"{returns.shape[1]} assets · {returns.shape[0]:,} periods"
        )
    with head_r:
        solve_clicked = st.button(
            "Optimize portfolio", type="primary", width="stretch"
        )

    if live_feasibility is not None and not live_feasibility.is_feasible:
        # Show the fix here rather than pointing at another tab: the analyst is
        # about to press Optimize, and the next click should be the correction,
        # not navigation.
        render_feasibility(live_feasibility, show_ok=False)

    if solve_clicked:
        st.session_state.walk_forward = None
        try:
            config = _build_config()
            with st.spinner("Solving…"):
                st.session_state["last_run"] = run_engine(
                    returns,
                    config,
                    build_frontier=build_frontier,
                    n_frontier_points=n_frontier_points,
                )
            st.session_state["last_error"] = None
        except Exception as exc:
            st.session_state["last_run"] = None
            st.session_state["last_error"] = str(exc)

    if st.session_state.get("last_error"):
        st.error(f"Optimization failed: {st.session_state['last_error']}")
        if live_feasibility is not None and live_feasibility.issues:
            st.markdown("**Most likely cause:**")
            render_feasibility(live_feasibility, show_ok=False)

    run = st.session_state.get("last_run")
    if run is None:
        st.info(
            "Set the method and assumptions in the sidebar, review the "
            "constraints tab, then click **Optimize portfolio**."
        )
    else:
        render_compliance(run.result)
        for warning in run.warnings:
            if warning not in run.result.violations:
                st.warning(warning)

        weights = run.result.weights
        metric_row(
            [
                ("Expected Return", pct(run.result.expected_return), None),
                ("Expected Volatility", pct(run.result.expected_volatility), None),
                (
                    "Sharpe Ratio",
                    num(run.result.sharpe_ratio),
                    f"Excess of the {risk_free_rate:.2%} risk-free rate over volatility.",
                ),
                (
                    "Solve time",
                    f"{run.result.extras.get('solve_seconds', 0):.2f}s",
                    f"Solver: {run.result.extras.get('solver', '—')}",
                ),
            ]
        )
        render_portfolio_diagnostics(run.diagnostics)
        render_projection_distance(run.result)

        st.divider()
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Weights**")
            st.plotly_chart(
                plot_weights_bar(
                    weights[weights.abs() > 1e-4],
                    title="Allocation",
                    bounds=st.session_state.config_table,
                ),
                width="stretch",
            )
        with right:
            st.markdown("**Capital vs. risk**")
            decomposition = run.risk_decomposition()
            st.plotly_chart(
                plot_weight_vs_risk(
                    decomposition[decomposition["weight"].abs() > 1e-4]
                ),
                width="stretch",
            )
            st.caption(
                "A position can be small in capital and large in risk. The gap "
                "between the bars is what risk budgeting exists to close."
            )

        with st.expander("Full risk decomposition (Euler)", expanded=False):
            table = run.risk_decomposition()
            display = table.copy()
            for col in ("weight", "contribution", "share_of_risk", "standalone_vol"):
                display[col] = display[col].map("{:.2%}".format)
            display["marginal_risk"] = table["marginal_risk"].map("{:.4f}".format)
            st.dataframe(display, width="stretch")
            st.caption(
                "`contribution` is in annualized volatility units and sums "
                f"exactly to the portfolio's {run.result.expected_volatility:.2%}."
            )
            groups_rc = run.group_risk_contributions()
            if len(groups_rc) > 1:
                st.markdown("**Risk by group**")
                st.dataframe(
                    groups_rc.to_frame("Share of risk").style.format(
                        {"Share of risk": "{:.2%}"}
                    ),
                    width="stretch",
                )

        if run.frontier is not None and ws["frontier"]["enabled"]:
            st.divider()
            st.markdown("### Efficient frontier")
            c1, c2 = st.columns([1, 1])
            with c1:
                show_cal = st.checkbox(
                    "Show capital allocation line", value=True,
                    help="Risk/return reachable by mixing the tangency "
                    "portfolio with cash at the risk-free rate.",
                )
            with c2:
                show_dominated = st.checkbox(
                    "Show dominated branch", value=False,
                    help="Portfolios below the minimum-variance point: same "
                    "risk, less return.",
                )
            render_frontier_health(run.frontier)
            on_cvar_axis = run.frontier.risk_measure == "CVaR"
            if on_cvar_axis:
                st.caption(
                    "The x-axis is Conditional VaR, the risk measure this "
                    "frontier was traced against. The minimum-variance and "
                    "tangency anchors and the capital allocation line live in "
                    "volatility space and have no position here."
                )
            st.plotly_chart(
                plot_efficient_frontier(
                    run.frontier,
                    risk_free_rate=(
                        float(risk_free_rate) if show_cal and not on_cvar_axis else None
                    ),
                    current_portfolio=(
                        None
                        if on_cvar_axis
                        else (
                            run.result.expected_volatility,
                            run.result.expected_return,
                            "Your portfolio",
                        )
                    ),
                    show_dominated=show_dominated,
                ),
                width="stretch",
            )

            f1, f2 = st.columns(2)
            with f1:
                st.plotly_chart(
                    plot_portfolio_composition(
                        run.frontier.weights, "Weights along the frontier"
                    ),
                    width="stretch",
                )
            with f2:
                if (
                    run.frontier.group_weights is not None
                    and not run.frontier.group_weights.empty
                ):
                    st.plotly_chart(
                        plot_portfolio_composition(
                            run.frontier.group_weights, "Group weights along the frontier"
                        ),
                        width="stretch",
                    )

        with st.expander("Estimation quality", expanded=False):
            render_covariance_diagnostics(run.covariance_diagnostics)
            st.markdown("**Expected returns used**")
            st.dataframe(
                run.expected_returns.to_frame("Annualized").style.format(
                    {"Annualized": "{:.2%}"}
                ),
                width="stretch",
            )
            if "bl_posterior_returns" in run.result.extras:
                st.markdown("**Black-Litterman: prior vs. posterior**")
                bl = pd.DataFrame(
                    {
                        "Equilibrium (prior)": run.result.extras["bl_prior_returns"],
                        "Posterior": run.result.extras["bl_posterior_returns"],
                        "View impact": run.result.extras["bl_view_impact"],
                    }
                )
                st.dataframe(
                    bl.style.format("{:.2%}"), width="stretch"
                )
                for v in run.result.extras.get("bl_views", []):
                    st.caption(f"View: {v}")
            if "hrp_clusters" in run.result.extras:
                st.markdown("**HRP clusters**")
                for label, members in run.result.extras["hrp_clusters"].items():
                    st.caption(f"Cluster {label}: {', '.join(members)}")
            if "risk_budget_achieved" in run.result.extras:
                st.markdown("**Risk budget: target vs. achieved**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Target": run.result.extras["risk_budget_target"],
                            "Achieved": run.result.extras["risk_budget_achieved"],
                        }
                    ).style.format("{:.2%}"),
                    width="stretch",
                )

        with st.expander("Assumptions behind this result", expanded=False):
            render_assumptions(run.assumptions())


# ---------------------------------------------------------------------------
# 📉 Backtest
# ---------------------------------------------------------------------------

with tab_backtest:
    run = st.session_state.get("last_run")
    if run is None:
        st.info("Run an optimization first — the backtest replays its weights.")
    else:
        st.subheader("How would this allocation have behaved?")
        st.warning(
            "**Everything on this page except the walk-forward section is "
            "in-sample.** The optimizer estimated its inputs from these same "
            "returns, so it already knew which assets won. Treat these "
            "numbers as a description of the fit, not as a forecast."
        )

        c1, c2 = st.columns(2)
        with c1:
            frequency = st.selectbox(
                "Rebalancing",
                options=["none", "monthly", "quarterly", "annual", "weekly", "daily"],
                index=1,
                format_func=lambda f: f.title() if f != "none" else "Buy and hold",
                help=(
                    "Between rebalances, weights drift with performance. "
                    "Assuming constant weights silently assumes free, "
                    "continuous rebalancing."
                ),
            )
        with c2:
            cost_bps = st.slider(
                "Transaction cost (bps, one-way)", 0, 100, 10,
                help="Charged on traded notional at each rebalance.",
            )

        bt = run.backtest(frequency=frequency, transaction_cost_bps=float(cost_bps))
        metric_row(
            [
                (
                    "Annualized return (net)",
                    pct(
                        float(
                            (1 + bt.returns).prod() ** (periods_per_year / len(bt.returns)) - 1
                        )
                    ),
                    None,
                ),
                ("Turnover per year", num(bt.annualized_turnover, 2), "One-way."),
                ("Total cost", pct(bt.total_cost), None),
                (
                    "Cost drag",
                    pct(bt.cost_drag(int(periods_per_year))),
                    "Annualized return given up to trading.",
                ),
            ]
        )

        st.plotly_chart(
            plot_wealth_index(
                bt.returns.to_frame("portfolio"), "Portfolio wealth (start = 1)"
            ),
            width="stretch",
        )
        b1, b2 = st.columns(2)
        with b1:
            st.plotly_chart(
                plot_drawdown(bt.returns, "Drawdown"), width="stretch"
            )
        with b2:
            st.plotly_chart(
                plot_return_distribution(
                    bt.returns,
                    var=float(np.percentile(bt.returns, 5)),
                    cvar=float(bt.returns[bt.returns <= np.percentile(bt.returns, 5)].mean()),
                    title="Return distribution with tail cuts",
                ),
                width="stretch",
            )

        st.markdown("**Summary**")
        st.dataframe(
            format_table(
                bt.summary(int(periods_per_year), float(risk_free_rate))
            ),
            width="stretch",
        )

        st.divider()
        st.markdown("### Versus a benchmark")
        bench_kind = st.selectbox(
            "Benchmark",
            options=["None", "Equal weight (1/N)", "Single asset", "Custom weights"],
            index=1,
            help=(
                "Absolute numbers say what happened; relative numbers say "
                "whether the optimizer earned its fee. Alpha, tracking error "
                "and active share all need something to be active against."
            ),
        )
        benchmark_returns = None
        benchmark_weights = None
        if bench_kind == "Equal weight (1/N)":
            benchmark_weights = pd.Series(
                1.0 / returns.shape[1], index=returns.columns
            )
        elif bench_kind == "Single asset":
            bench_asset = st.selectbox(
                "Benchmark asset", options=list(returns.columns)
            )
            benchmark_weights = pd.Series(0.0, index=returns.columns)
            benchmark_weights[bench_asset] = 1.0
        elif bench_kind == "Custom weights":
            if (
                "benchmark_weights_table" not in st.session_state
                or set(st.session_state.benchmark_weights_table.index)
                != set(returns.columns)
            ):
                st.session_state.benchmark_weights_table = pd.DataFrame(
                    {"Weight": [1.0 / returns.shape[1]] * returns.shape[1]},
                    index=returns.columns,
                )
            st.session_state.benchmark_weights_table = st.data_editor(
                st.session_state.benchmark_weights_table,
                num_rows="fixed",
                column_config={
                    "Weight": st.column_config.NumberColumn(
                        min_value=-1.0, max_value=1.5, step=0.01, format="%.3f"
                    ),
                },
                key="bench_editor",
            )
            benchmark_weights = st.session_state.benchmark_weights_table["Weight"]

        if benchmark_weights is not None:
            total = float(benchmark_weights.sum())
            if abs(total) < 1e-9:
                st.warning("Benchmark weights sum to zero — pick a different mix.")
            else:
                benchmark_returns = (returns * (benchmark_weights / total)).sum(axis=1)

        if benchmark_returns is not None:
            relative = summary_relative(
                bt.returns.to_frame("portfolio"),
                benchmark_returns.reindex(bt.returns.index),
                periods_per_year=int(periods_per_year),
                riskfree_rate=float(risk_free_rate),
                extended=True,
            ).T
            share = active_share(run.result.weights, benchmark_weights / total)
            metric_row(
                [
                    (
                        "Annualized excess",
                        pct(float(relative.loc["Annualized Excess", "portfolio"])),
                        None,
                    ),
                    (
                        "Tracking error",
                        pct(float(relative.loc["Annualized T.E.", "portfolio"])),
                        None,
                    ),
                    (
                        "Information ratio",
                        num(float(relative.loc["Information Ratio", "portfolio"])),
                        "Excess return per unit of tracking error.",
                    ),
                    (
                        "Active share",
                        pct(share),
                        "Half the sum of absolute weight differences. 0 is the "
                        "benchmark; 1 shares no holding with it.",
                    ),
                ]
            )
            st.plotly_chart(
                plot_wealth_index(
                    pd.concat(
                        {
                            "Portfolio": bt.returns,
                            "Benchmark": benchmark_returns.reindex(bt.returns.index),
                        },
                        axis=1,
                    ),
                    "Portfolio vs. benchmark",
                ),
                width="stretch",
            )
            st.dataframe(
                relative.style.format("{:.4f}"), width="stretch"
            )
            t_stat = float(relative.loc["Alpha t-stat", "portfolio"])
            alpha = float(relative.loc["Alpha (annualized)", "portfolio"])
            if abs(t_stat) < 2:
                st.info(
                    f"CAPM alpha is {alpha:.2%} with a t-statistic of "
                    f"{t_stat:.2f}. Below |2| the alpha is not statistically "
                    "distinguishable from zero on this sample — and this is "
                    "still the in-sample fit."
                )
            else:
                st.success(
                    f"CAPM alpha of {alpha:.2%} (t = {t_stat:.2f}) is "
                    "statistically significant in sample. Confirm it survives "
                    "the walk-forward below before believing it."
                )

        if frequency != "daily":
            with st.expander("Weight drift between rebalances", expanded=False):
                st.plotly_chart(
                    plot_weight_evolution(bt.weights, "Actual held weights"),
                    width="stretch",
                )

        window = st.slider(
            "Rolling window (periods)",
            min_value=int(periods_per_year // 4),
            max_value=min(int(periods_per_year * 3), max(len(bt.returns) - 1, 2)),
            value=min(int(periods_per_year), max(len(bt.returns) - 1, 2)),
            help=(
                "A full-sample Sharpe cannot tell a strategy that worked "
                "throughout from one that earned everything in a single window."
            ),
        )
        st.plotly_chart(
            plot_rolling_metrics(
                rolling_metrics(
                    bt.returns, window, float(risk_free_rate), int(periods_per_year)
                )
            ),
            width="stretch",
        )

        st.divider()
        st.markdown("### Out-of-sample walk-forward")
        st.caption(
            "Re-estimates and re-solves on a rolling window, then holds each "
            "solution over returns the optimizer never saw. The gap against "
            "the in-sample curve is how much of the backtest was hindsight."
        )
        w1, w2, w3 = st.columns(3)
        with w1:
            lookback = st.number_input(
                "Estimation window (periods)",
                min_value=int(periods_per_year // 2),
                max_value=max(int(len(returns) - periods_per_year // 4), 10),
                value=min(int(periods_per_year * 2), max(len(returns) // 2, 10)),
            )
        with w2:
            rebalance_every = st.number_input(
                "Re-solve every (periods)",
                min_value=1, max_value=int(periods_per_year * 2),
                value=max(int(periods_per_year // 4), 1),
            )
        with w3:
            expanding = st.checkbox(
                "Expanding window", value=False,
                help="Grow the sample from inception instead of rolling it.",
            )

        if st.button("Run walk-forward", key="run_wf"):
            with st.spinner("Re-solving through history…"):
                try:
                    st.session_state.walk_forward = run.walk_forward(
                        lookback=int(lookback),
                        rebalance_every=int(rebalance_every),
                        transaction_cost_bps=float(cost_bps),
                        expanding=expanding,
                    )
                except Exception as exc:
                    st.session_state.walk_forward = None
                    st.error(f"Walk-forward failed: {exc}")

        wf = st.session_state.get("walk_forward")
        if wf is not None:
            metric_row(
                [
                    ("Re-solves", str(wf.n_rebalances), None),
                    (
                        "OOS annualized return",
                        pct(
                            float(
                                (1 + wf.returns).prod()
                                ** (periods_per_year / len(wf.returns))
                                - 1
                            )
                        ),
                        None,
                    ),
                    ("Turnover per year", num(wf.backtest.annualized_turnover, 2), None),
                    (
                        "Failed solves",
                        str(len(wf.failures)),
                        "Failed re-solves carry the previous book forward.",
                    ),
                ]
            )
            in_sample = bt.returns.reindex(wf.returns.index)
            st.plotly_chart(
                plot_walk_forward_comparison(in_sample, wf.returns),
                width="stretch",
            )
            comparison = run.in_vs_out_of_sample(wf, float(risk_free_rate))
            st.markdown("**In-sample vs. out-of-sample**")
            st.dataframe(format_table(comparison), width="stretch")
            degradation = comparison.loc["Sharpe Ratio", "Degradation"]
            if degradation > 0.5:
                st.error(
                    f"The Sharpe ratio falls by {degradation:.2f} out of sample. "
                    "Most of the in-sample result was fitted, not earned — "
                    "consider a more robust method (HRP, minimum variance, "
                    "risk parity) or shrunk expected returns."
                )
            elif degradation > 0.2:
                st.warning(
                    f"The Sharpe ratio falls by {degradation:.2f} out of sample — "
                    "a normal amount of optimism, but size positions on the "
                    "out-of-sample number."
                )
            else:
                st.success(
                    f"The Sharpe ratio holds up out of sample (gap "
                    f"{degradation:.2f})."
                )

            stability = wf.weight_stability()
            if not stability.empty:
                with st.expander("Weight stability across re-solves", expanded=False):
                    st.caption(
                        "Average absolute change in each asset's weight between "
                        "re-solves. Large values mean the optimizer is chasing "
                        "estimation noise, and the turnover it implies is real."
                    )
                    st.dataframe(
                        stability.sort_values(ascending=False)
                        .to_frame("Mean absolute change")
                        .style.format("{:.2%}"),
                        width="stretch",
                    )
            if wf.failures:
                with st.expander(f"{len(wf.failures)} failed re-solve(s)"):
                    for f in wf.failures[:20]:
                        st.text(f)


# ---------------------------------------------------------------------------
# 🆚 Compare
# ---------------------------------------------------------------------------

with tab_compare:
    if not st.session_state.scenarios:
        st.info(
            "Save at least one scenario from the **📚 Scenarios** sidebar block "
            "to compare."
        )
    else:
        names_all = sorted(st.session_state.scenarios.keys())
        # Default to everything saved (up to the cap): a comparison tab that
        # opens on a single scenario is not comparing anything.
        default_sel = names_all[:5]
        if (
            st.session_state.active_scenario in names_all
            and st.session_state.active_scenario not in default_sel
        ):
            default_sel = [st.session_state.active_scenario] + default_sel[:4]
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
            runs: dict[str, object] = {}
            for n in chosen:
                scn = st.session_state.scenarios[n]
                try:
                    sub = _scenario_returns_subset(scn, returns)
                    runs[n] = _solve_scenario_cached(scenario_signature(scn), sub)
                    covered = len(
                        [c for c in returns.columns if c in scn.config.expected_returns]
                    )
                    total = len(scn.config.expected_returns)
                    if covered != total:
                        st.caption(
                            f"_{n}: covers {covered}/{total} of its assets in the "
                            "loaded data._"
                        )
                except Exception as exc:
                    st.error(f"{n}: {exc}")

            if runs:
                summary_df = pd.DataFrame(
                    [_summarize_run(n, r) for n, r in runs.items()]
                ).set_index("Scenario")
                st.dataframe(
                    summary_df.style.format(
                        {
                            "Expected Return": "{:.2%}",
                            "Expected Vol": "{:.2%}",
                            "Sharpe": "{:.3f}",
                            "Effective N": "{:.1f}",
                            "Div. ratio": "{:.2f}",
                        }
                    ),
                    width="stretch",
                )

                weights_df = pd.DataFrame(
                    {n: r.result.weights for n, r in runs.items()}
                ).fillna(0.0)
                st.plotly_chart(
                    plot_portfolio_composition(weights_df, title="Weights by scenario"),
                    width="stretch",
                )
                grouped = plot_risk_contributions(weights_df)
                grouped.update_layout(title="Per-asset weights")
                st.plotly_chart(grouped, width="stretch")

                st.markdown("**Backtest comparison**")
                st.caption(
                    "In-sample, costless, rebalanced every period — a like-for-"
                    "like comparison, not a forecast for any of them."
                )
                bt_df = pd.concat(
                    {n: r.backtest_returns()["portfolio"] for n, r in runs.items()},
                    axis=1,
                )
                st.plotly_chart(
                    plot_wealth_index(bt_df, "Backtest comparison"),
                    width="stretch",
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
            "Anchor scenario", options=names_all, index=default_idx,
            key="whatif_anchor",
        )

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
                f"Live re-solve is off (optimizer='{anchor_cfg.optimizer.name}', "
                f"{n_assets} assets). Drag sliders, then press **Recompute**."
            )

        st.markdown("**Per-asset weight bounds**")
        cols = st.columns(2)
        overrides: dict[str, tuple[float, float]] = dict(
            st.session_state.get("whatif_overrides", {})
        )
        for i, a in enumerate(anchor_assets):
            lo0, hi0 = anchor_cfg.bounds.get(a, [0.0, 1.0])
            current = overrides.get(a, (float(lo0), float(hi0)))
            with cols[i % 2]:
                lo, hi = st.slider(
                    f"{a}", min_value=-1.0, max_value=1.5, step=0.01,
                    value=(float(current[0]), float(current[1])),
                    key=f"whatif_bnd_{anchor_name}_{a}",
                )
            overrides[a] = (lo, hi)
        st.session_state.whatif_overrides = overrides

        st.markdown("**Optimizer settings**")
        anchor_req = requirements_for(anchor_cfg.optimizer.name)
        extras: dict[str, object] = dict(st.session_state.get("whatif_extra", {}))

        if anchor_req.supports_target_return or anchor_req.supports_target_volatility:
            modes = []
            if anchor_req.supports_target_return:
                modes.append("Target return")
            if anchor_req.supports_target_volatility:
                modes.append("Target volatility")
            if anchor_req.supports_risk_aversion:
                modes.append("Utility")
            if anchor_cfg.optimizer.target_return is not None:
                default_mode = "Target return"
            elif anchor_cfg.optimizer.target_volatility is not None:
                default_mode = "Target volatility"
            else:
                default_mode = modes[0]
            wf_mode = st.radio(
                "Mode", modes,
                index=modes.index(default_mode) if default_mode in modes else 0,
                horizontal=True, key="whatif_mv_mode",
            )
            if wf_mode == "Target return":
                tr = st.number_input(
                    "Target return (annual)",
                    value=float(anchor_cfg.optimizer.target_return or 0.07),
                    step=0.005, format="%.4f", key="whatif_target_return",
                )
                extras = {"target_return": float(tr), "target_volatility": None}
            elif wf_mode == "Target volatility":
                tv = st.number_input(
                    "Target volatility (annual)",
                    value=float(anchor_cfg.optimizer.target_volatility or 0.10),
                    step=0.005, format="%.4f", key="whatif_target_vol",
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

        for extra in anchor_req.extras:
            if extra.kind == "scalar" and extra.key == "cvar_alpha":
                extras["cvar_alpha"] = float(
                    st.slider(
                        "CVaR tail probability α", 0.01, 0.20,
                        float(anchor_cfg.optimizer.cvar_alpha or 0.05), 0.01,
                        key="whatif_cvar_alpha",
                    )
                )
            elif extra.kind == "scalar" and extra.key == "bl_tau":
                extras["bl_tau"] = float(
                    st.slider(
                        "τ (prior uncertainty)", 0.01, 0.5,
                        float(anchor_cfg.optimizer.bl_tau or 0.05), 0.01,
                        key="whatif_bl_tau",
                    )
                )
            elif extra.kind == "choice" and extra.key == "hrp_linkage":
                extras["hrp_linkage"] = str(
                    st.selectbox(
                        "HRP linkage", list(extra.choices or ()),
                        index=(extra.choices or ("single",)).index(
                            anchor_cfg.optimizer.hrp_linkage or "single"
                        ),
                        key="whatif_hrp_linkage",
                    )
                )
            elif extra.kind == "per_asset" and extra.key == "risk_budget":
                rb_idx = list(anchor_cfg.expected_returns.keys())
                default_rb = anchor_cfg.optimizer.risk_budget or {
                    a: 1.0 / len(rb_idx) for a in rb_idx
                }
                rb_df = st.data_editor(
                    pd.DataFrame(
                        {"Risk Budget": [default_rb.get(a, 0.0) for a in rb_idx]},
                        index=rb_idx,
                    ),
                    num_rows="fixed",
                    column_config={
                        "Risk Budget": st.column_config.NumberColumn(
                            min_value=0.001, max_value=1.0, step=0.01, format="%.3f",
                        ),
                    },
                    key="whatif_rb_editor",
                )
                extras["risk_budget"] = rb_df["Risk Budget"].to_dict()

        st.session_state.whatif_extra = extras

        def _live_config():
            cfg_dict = anchor_cfg.to_dict()
            cfg_dict["bounds"] = {
                a: list(
                    st.session_state.whatif_overrides.get(
                        a, anchor_cfg.bounds.get(a, [0.0, 1.0])
                    )
                )
                for a in anchor_assets
            }
            opt = dict(cfg_dict["optimizer"])
            for k, v in st.session_state.whatif_extra.items():
                opt[k] = v
            cfg_dict["optimizer"] = opt
            return EngineConfig.from_dict(cfg_dict)

        should_solve = (
            st.button("Recompute", type="primary", key="whatif_recompute")
            if is_slow
            else True
        )

        if should_solve:
            try:
                cfg_live = _live_config()
                sub = returns[[c for c in returns.columns if c in anchor_assets]]
                st.session_state.whatif_run = _solve_scenario_cached(
                    config_signature(cfg_live), sub
                )
                st.session_state.whatif_error = None
            except Exception as exc:
                st.session_state.whatif_run = None
                st.session_state.whatif_error = str(exc)

        wf_run = st.session_state.get("whatif_run")
        if st.session_state.get("whatif_error"):
            st.error(f"Solver: {st.session_state.whatif_error}")
            try:
                cfg_live = _live_config()
                sub = returns[[c for c in returns.columns if c in anchor_assets]]
                render_feasibility(
                    _feasibility_cached(
                        config_signature(cfg_live), _frame_hash(sub), sub
                    ),
                    show_ok=False,
                )
            except Exception:
                pass
            if st.button("Reset to anchor", key="whatif_reset"):
                st.session_state.whatif_overrides = {}
                st.session_state.whatif_extra = {}
                st.session_state.whatif_error = None
                st.rerun()
        elif wf_run is not None:
            metric_row(
                [
                    ("Expected Return", pct(wf_run.result.expected_return), None),
                    ("Expected Vol", pct(wf_run.result.expected_volatility), None),
                    ("Sharpe", num(wf_run.result.sharpe_ratio), None),
                    (
                        "Effective N",
                        num(wf_run.diagnostics.effective_n, 1)
                        if wf_run.diagnostics
                        else "—",
                        None,
                    ),
                ]
            )
            try:
                anchor_run = _solve_scenario_cached(
                    scenario_signature(anchor_scn),
                    _scenario_returns_subset(anchor_scn, returns),
                )
                weights_df = pd.DataFrame(
                    {
                        "Anchor": anchor_run.result.weights,
                        "What-if": wf_run.result.weights,
                    }
                ).fillna(0.0)
                delta = weights_df["What-if"] - weights_df["Anchor"]
                st.caption(
                    f"One-way turnover versus the anchor: {delta.abs().sum():.2%}"
                )
            except Exception:
                weights_df = pd.DataFrame({"What-if": wf_run.result.weights}).fillna(0.0)
            st.plotly_chart(
                plot_portfolio_composition(weights_df, title="Anchor vs. What-if"),
                width="stretch",
            )

            if st.button("Save these as new scenario", key="whatif_save"):
                st.session_state["whatif_save_pending"] = True

            if st.session_state.get("whatif_save_pending"):
                new_name = st.text_input(
                    "Scenario name", key="whatif_save_name",
                    placeholder="e.g. Anchor + tighter equities",
                )
                cA, cB = st.columns(2)
                if cA.button("Save", key="whatif_save_confirm", type="primary"):
                    if not new_name:
                        st.error("Name is required.")
                    elif new_name in st.session_state.scenarios:
                        st.error(f"Scenario {new_name!r} already exists.")
                    else:
                        st.session_state.scenarios[new_name] = Scenario(
                            name=new_name,
                            config=_live_config(),
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
# Sidebar — 📚 Scenarios (rendered after _build_config exists)
# ---------------------------------------------------------------------------

with st.sidebar.expander("📚 Scenarios", expanded=False):
    names_in_state = sorted(st.session_state.scenarios.keys())
    sel_options = ["—"] + names_in_state
    default_idx = (
        sel_options.index(st.session_state.active_scenario)
        if st.session_state.active_scenario in names_in_state
        else 0
    )
    selected = st.selectbox(
        "Active scenario", options=sel_options, index=default_idx, key="scn_select"
    )
    has_selection = selected != "—"

    new_name = st.text_input("Name", key="scn_new_name").strip()
    notes = st.text_area(
        "Notes (optional)", key="scn_notes", height=70, max_chars=NOTES_MAX_LEN
    )
    cA, cB = st.columns(2)
    save_clicked = cA.button("Save", width="stretch", key="scn_save")
    update_clicked = cB.button(
        "Update", disabled=not has_selection, width="stretch", key="scn_update"
    )
    cC, cD = st.columns(2)
    load_clicked = cC.button(
        "Load", disabled=not has_selection, width="stretch", key="scn_load"
    )
    delete_clicked = cD.button(
        "Delete", disabled=not has_selection, width="stretch", key="scn_delete"
    )

    rename_to = st.text_input(
        "Rename to", key="scn_rename_to", disabled=not has_selection
    ).strip()
    rename_clicked = st.button(
        "Rename", disabled=(not has_selection) or (not rename_to), key="scn_rename"
    )

    st.divider()
    st.download_button(
        "⬇ Download all (YAML)",
        data=dump_scenarios_yaml(st.session_state.scenarios)
        if st.session_state.scenarios
        else "",
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

    if save_clicked:
        if not new_name:
            st.error("Name is required.")
        elif new_name in st.session_state.scenarios:
            st.error(
                f"Scenario {new_name!r} already exists. Use Update or pick "
                "another name."
            )
        else:
            ts = now_iso()
            st.session_state.scenarios[new_name] = Scenario(
                name=new_name, config=_build_config(), notes=notes,
                created_at=ts, updated_at=ts,
            )
            st.session_state.active_scenario = new_name
            st.success(f"Saved scenario {new_name!r}.")
            st.rerun()

    if update_clicked and has_selection:
        prev = st.session_state.scenarios[selected]
        st.session_state.scenarios[selected] = Scenario(
            name=selected,
            config=_build_config(),
            notes=notes or prev.notes,
            created_at=prev.created_at or now_iso(),
            updated_at=now_iso(),
        )
        st.session_state.last_run_by_scenario.pop(selected, None)
        st.success(f"Updated {selected!r}.")
        st.rerun()

    if load_clicked and has_selection:
        target_cfg = st.session_state.scenarios[selected].config
        loaded_assets = set(returns.columns)
        missing = sorted(set(target_cfg.expected_returns) - loaded_assets)
        if missing:
            kept = [a for a in target_cfg.expected_returns if a in loaded_assets]
            if not kept:
                st.error(
                    f"Cannot load {selected!r}: none of its assets are in the "
                    "loaded data."
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
                prev = st.session_state.scenarios[selected]
                st.session_state.scenarios[selected] = Scenario(
                    name=selected,
                    config=EngineConfig.from_dict(trimmed_dict),
                    notes=prev.notes,
                    created_at=prev.created_at,
                    updated_at=prev.updated_at,
                )
                st.session_state.scenario_load_warning = (
                    f"Loaded {selected!r}; dropped {len(missing)} missing "
                    "asset(s): " + ", ".join(missing)
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
            incoming = load_scenarios_yaml(upl.read().decode("utf-8"))
            applied = 0
            for k, v in incoming.items():
                if k in st.session_state.scenarios:
                    if merge_mode == "Skip":
                        continue
                    if merge_mode == "Overwrite":
                        st.session_state.scenarios[k] = v
                        st.session_state.last_run_by_scenario.pop(k, None)
                        applied += 1
                    else:
                        suffix = 2
                        candidate = f"{k} ({suffix})"
                        while candidate in st.session_state.scenarios:
                            suffix += 1
                            candidate = f"{k} ({suffix})"
                        st.session_state.scenarios[candidate] = Scenario(
                            name=candidate, config=v.config, notes=v.notes,
                            created_at=v.created_at, updated_at=v.updated_at,
                        )
                        applied += 1
                else:
                    st.session_state.scenarios[k] = v
                    applied += 1
            st.success(f"Uploaded {applied} scenario(s).")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")


# ---------------------------------------------------------------------------
# 📤 Report
# ---------------------------------------------------------------------------

with tab_report:
    run = st.session_state.get("last_run")
    if run is None:
        st.info("Run an optimization first.")
    else:
        st.subheader("Export")
        st.caption(
            "The workbook carries the assumptions and diagnostics alongside "
            "the weights, so a reader can see what the numbers rest on."
        )

        assumptions = run.assumptions()
        # Same builder the CLI uses, so a workbook exported from the app and
        # one written by `optengine optimize` carry identical sheets.
        sheets = run_sheets(
            run,
            riskfree_rate=float(risk_free_rate),
            data_quality=quality,
            walk_forward=st.session_state.get("walk_forward"),
        )

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
            type="primary",
        )
        st.caption(f"{len(sheets)} sheets: {', '.join(sheets)}")

        st.markdown("**Assumptions**")
        render_assumptions(assumptions)

        st.markdown("**Config used (YAML)**")
        import yaml as _yaml

        st.code(
            _yaml.safe_dump(run.config.to_dict(), sort_keys=False), language="yaml"
        )
