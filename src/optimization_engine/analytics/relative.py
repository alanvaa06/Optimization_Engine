"""Relative-performance metrics versus a benchmark.

Every function here aligns the portfolio and the benchmark on their common
dates before computing anything. Comparing two series that merely happen to
be the same length — one starting a month later, say — produces a beta and a
tracking error that describe no real portfolio, and nothing in the output
would reveal it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from optimization_engine.analytics.performance import (
    annualize_returns,
    annualize_volatility,
)


def spread(
    series_1: pd.Series | pd.DataFrame, series_2: pd.Series | pd.DataFrame
) -> pd.DataFrame | pd.Series:
    """Arithmetic difference of two aligned return streams."""
    if isinstance(series_2, pd.DataFrame):
        series_2 = series_2.iloc[:, 0]
    if isinstance(series_1, pd.DataFrame):
        return series_1.sub(series_2, axis=0)
    return series_1 - series_2


def _ensure_frame(s: pd.Series | pd.DataFrame) -> pd.DataFrame:
    return s.to_frame() if isinstance(s, pd.Series) else s


def _as_series(rb: pd.Series | pd.DataFrame) -> pd.Series:
    return rb.iloc[:, 0] if isinstance(rb, pd.DataFrame) else rb


def _aligned(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Restrict both inputs to the dates on which both are observed.

    Raises:
        ValueError: When the two series share no dates at all — a silent
            empty result would otherwise propagate as NaN metrics.
    """
    frame = _ensure_frame(r)
    bench = _as_series(rb)
    common = frame.dropna(how="all").index.intersection(bench.dropna().index)
    if len(common) == 0:
        raise ValueError(
            "The portfolio and the benchmark share no dates, so no relative "
            "metric can be computed. Check that both cover the same period."
        )
    return frame.loc[common], bench.loc[common]


def _geometric_capture(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame, upside: bool
) -> pd.Series:
    frame, bench = _aligned(r, rb)
    mask = bench > 0 if upside else bench < 0
    bench_periods = bench[mask]
    n_b = len(bench_periods)
    if n_b == 0:
        return pd.Series({c: float("nan") for c in frame.columns})
    bench_geo = float((1 + bench_periods).prod() ** (1 / n_b) - 1)

    out: dict[str, float] = {}
    for col in frame.columns:
        series = frame.loc[mask, col].dropna()
        n = len(series)
        if n == 0 or bench_geo == 0:
            out[col] = float("nan")
        else:
            geo = float((1 + series).prod() ** (1 / n) - 1)
            out[col] = geo / bench_geo
    return pd.Series(out)


def up_capture(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """Geometric up-capture: share of the benchmark's gains that was captured."""
    return _geometric_capture(r, rb, upside=True)


def down_capture(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """Geometric down-capture: share of the benchmark's losses that was taken."""
    return _geometric_capture(r, rb, upside=False)


def capture_ratio(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """Up-capture over down-capture. Above 1 is the asymmetry worth paying for."""
    return up_capture(r, rb) / down_capture(r, rb)


def regression_stats(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    riskfree_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Full CAPM regression of each column on the benchmark.

    Returns alpha (annualized), beta, the t-statistic of alpha, R², and
    residual (idiosyncratic) volatility.

    Beta on its own says how much market exposure a portfolio carries; it says
    nothing about whether the rest of the return was skill or noise. The alpha
    t-statistic and R² are what turn a number into a claim: an alpha of 2%
    with a t-stat of 0.4 has not been demonstrated.
    """
    frame, bench = _aligned(r, rb)
    rf = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    x = sm.add_constant((bench - rf).rename("benchmark"))

    rows: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        y = frame[col] - rf
        valid = y.notna()
        model = sm.OLS(y[valid], x.loc[valid.index[valid]]).fit()
        alpha_period = float(model.params.iloc[0])
        rows[col] = {
            "Alpha (annualized)": (1 + alpha_period) ** periods_per_year - 1,
            "Beta": float(model.params.iloc[1]),
            "Alpha t-stat": float(model.tvalues.iloc[0]),
            "R-squared": float(model.rsquared),
            "Residual Vol": float(
                np.sqrt(model.mse_resid) * np.sqrt(periods_per_year)
            ),
        }
    return pd.DataFrame(rows).T


def beta(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """OLS beta of each column of ``r`` against benchmark ``rb``."""
    frame, bench = _aligned(r, rb)
    x = sm.add_constant(bench.rename("benchmark"))
    out = {
        col: float(sm.OLS(frame[col], x).fit().params.iloc[1])
        for col in frame.columns
    }
    return pd.Series(out, name="Beta")


def conditional_beta(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """Beta measured separately in up and down benchmark periods.

    A portfolio with the same beta in both directions is symmetric; one whose
    down-beta exceeds its up-beta is the shape investors dislike most, and a
    single full-sample beta averages the two into silence.
    """
    frame, bench = _aligned(r, rb)
    rows: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        entry: dict[str, float] = {}
        for label, mask in (("Up Beta", bench > 0), ("Down Beta", bench < 0)):
            if int(mask.sum()) < 3:
                entry[label] = float("nan")
                continue
            x = sm.add_constant(bench[mask].rename("benchmark"))
            entry[label] = float(sm.OLS(frame.loc[mask, col], x).fit().params.iloc[1])
        entry["Beta Asymmetry"] = entry.get("Down Beta", np.nan) - entry.get(
            "Up Beta", np.nan
        )
        rows[col] = entry
    return pd.DataFrame(rows).T


def tracking_error(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series | float:
    """Annualized volatility of the return difference versus the benchmark."""
    frame, bench = _aligned(r, rb)
    return annualize_volatility(frame.sub(bench, axis=0), periods_per_year)


def information_ratio(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame, periods_per_year: int = 252
) -> pd.Series:
    """Annualized excess return over annualized tracking error.

    The numerator is the difference of *geometric* annualized returns while
    the denominator annualizes the volatility of the *arithmetic* difference.
    That is the industry convention, and it is worth knowing it is a mismatch:
    over long, volatile samples the two conventions can differ by enough to
    move an IR by a tenth.
    """
    frame, bench = _aligned(r, rb)
    ann_excess = annualize_returns(frame, periods_per_year) - annualize_returns(
        bench, periods_per_year
    )
    te = annualize_volatility(frame.sub(bench, axis=0), periods_per_year)
    return ann_excess / te


def active_share(weights: pd.Series, benchmark_weights: pd.Series) -> float:
    """Half the sum of absolute weight differences versus the benchmark.

    0 means the portfolio is the benchmark; 1 means it shares no holding with
    it. Unlike tracking error, active share is computed from positions alone,
    so it cannot be flattered by a quiet market.
    """
    assets = weights.index.union(benchmark_weights.index)
    w = weights.reindex(assets).fillna(0.0)
    b = benchmark_weights.reindex(assets).fillna(0.0)
    return float((w - b).abs().sum() / 2.0)


def summary_relative(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
    riskfree_rate: float = 0.0,
    extended: bool = False,
) -> pd.DataFrame:
    """Benchmark-relative summary for each column of ``r``.

    Args:
        r: Portfolio returns.
        rb: Benchmark returns.
        periods_per_year: Observations per year.
        riskfree_rate: Annual risk-free rate, used for the CAPM alpha.
        extended: Add alpha, its t-statistic, R², residual volatility, and
            up/down betas.
    """
    frame, bench = _aligned(r, rb)
    ann_excess = annualize_returns(frame, periods_per_year) - annualize_returns(
        bench, periods_per_year
    )
    te = annualize_volatility(frame.sub(bench, axis=0), periods_per_year)
    up = up_capture(frame, bench)
    down = down_capture(frame, bench)

    out = pd.DataFrame(
        {
            "Annualized Excess": ann_excess,
            "Annualized T.E.": te,
            "Information Ratio": ann_excess / te,
            "Beta": beta(frame, bench),
            "Up Capture": up,
            "Down Capture": down,
            "Capture": up / down,
        }
    )
    if extended:
        out = out.join(
            regression_stats(frame, bench, riskfree_rate, periods_per_year)[
                ["Alpha (annualized)", "Alpha t-stat", "R-squared", "Residual Vol"]
            ]
        ).join(conditional_beta(frame, bench))
    return out
