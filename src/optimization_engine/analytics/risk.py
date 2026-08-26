"""Risk metrics: VaR, CVaR, downside deviation, drawdown shape, risk contributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats


def downside_deviation(
    r: pd.Series | pd.DataFrame,
    mar: float = 0.0,
    full_sample: bool = True,
) -> float | pd.Series:
    """Root-mean-square shortfall below a minimum acceptable return.

    ``√( mean( min(r − MAR, 0)² ) )`` — the correct denominator for the
    Sortino ratio.

    The distinction that matters: the average is taken over *every* period,
    not only the losing ones. Taking the standard deviation of the negative
    subset instead (a common shortcut) both changes the denominator and
    re-centres the losses on their own mean, which understates downside risk —
    on the engine's sample panel by roughly 16%, inflating Sortino by ~19%.

    Args:
        r: Periodic returns.
        mar: Minimum acceptable return per period. ``0.0`` is the usual
            choice; pass the per-period risk-free rate for an excess-return
            Sortino.
        full_sample: Divide by the total number of periods (the standard
            definition). ``False`` divides by the number of shortfall periods
            instead, which answers "how bad is a bad period" rather than "how
            much downside risk does this carry".
    """
    shortfall = np.minimum(r - mar, 0.0)
    squared = shortfall**2
    if full_sample:
        return np.sqrt(squared.mean())
    below = (shortfall < 0).sum()
    if isinstance(r, pd.DataFrame):
        return np.sqrt(squared.sum() / below.replace(0, np.nan))
    return float(np.sqrt(squared.sum() / below)) if below else float("nan")


def semideviation(r: pd.Series | pd.DataFrame, mar: float = 0.0) -> float | pd.Series:
    """Alias for :func:`downside_deviation` with the standard denominator."""
    return downside_deviation(r, mar=mar, full_sample=True)


def skewness(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Population skewness. Negative means the left tail is the long one."""
    demeaned = r - r.mean()
    sigma = r.std(ddof=0)
    return (demeaned**3).mean() / sigma**3


def kurtosis(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Population kurtosis, *not* excess: a normal distribution scores 3."""
    demeaned = r - r.mean()
    sigma = r.std(ddof=0)
    return (demeaned**4).mean() / sigma**4


def is_normal(r: pd.Series, level: float = 0.01) -> bool:
    """Jarque-Bera normality test, returning True if we accept normality."""
    _, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r: pd.Series | pd.DataFrame, level: float = 5) -> float | pd.Series:
    """Historic Value at Risk at the ``level`` percentile (5 = 5%).

    Returned as a positive number: a VaR of 0.02 means "lose 2% or more in
    the worst 5% of periods".
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    if isinstance(r, pd.Series):
        return -np.percentile(r, level)
    raise TypeError("Expected Series or DataFrame")


def var_gaussian(
    r: pd.Series | pd.DataFrame, level: float = 5, modified: bool = False
) -> float | pd.Series:
    """Parametric Value at Risk, optionally Cornish-Fisher adjusted.

    The Cornish-Fisher expansion corrects the Gaussian quantile for skew and
    kurtosis. It is an asymptotic expansion, so it is only trustworthy for
    moderate departures from normality — with kurtosis far above ~7 the
    adjusted quantile can stop being monotone in the confidence level. Prefer
    :func:`cvar_historic` when the sample is that wild.
    """
    z = scipy.stats.norm.ppf(level / 100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (
            z
            + (z**2 - 1) * s / 6
            + (z**3 - 3 * z) * (k - 3) / 24
            - (2 * z**3 - 5 * z) * (s**2) / 36
        )
    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r: pd.Series | pd.DataFrame, level: float = 5) -> float | pd.Series:
    """Historic Conditional VaR (expected shortfall) beyond the VaR threshold."""
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    raise TypeError("Expected Series or DataFrame")


def tail_ratio(r: pd.Series | pd.DataFrame, level: float = 5) -> float | pd.Series:
    """Right-tail magnitude divided by left-tail magnitude.

    Above 1 means the good surprises are bigger than the bad ones. A useful
    companion to skewness because it reads off the actual quantiles rather
    than a third moment that a single outlier can dominate.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(tail_ratio, level=level)
    right = float(np.percentile(r, 100 - level))
    left = float(abs(np.percentile(r, level)))
    return right / left if left > 0 else float("nan")


def omega_ratio(
    r: pd.Series | pd.DataFrame, threshold: float = 0.0
) -> float | pd.Series:
    """Probability-weighted gains above a threshold over losses below it.

    Omega uses the whole return distribution rather than its first two
    moments, so it distinguishes portfolios that Sharpe cannot.
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(omega_ratio, threshold=threshold)
    excess = r - threshold
    gains = float(excess[excess > 0].sum())
    losses = float(-excess[excess < 0].sum())
    return gains / losses if losses > 0 else float("inf")


# ---------------------------------------------------------------------------
# Drawdown shape
# ---------------------------------------------------------------------------


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak, as a negative fraction."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return (wealth - peak) / peak


def drawdown_table(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """The ``top`` worst drawdown episodes, with their timing.

    Max drawdown alone says how deep the hole was. What determines whether an
    investor actually sat through it is how *long* it lasted — so this reports
    peak, trough, recovery, depth, and both the decline and recovery lengths.
    An episode still underwater at the end of the sample has ``NaT`` recovery
    and is flagged.
    """
    dd = drawdown_series(returns)
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()

    underwater = dd < -1e-12
    if not underwater.any():
        return pd.DataFrame(
            columns=[
                "peak_date", "trough_date", "recovery_date", "max_drawdown",
                "decline_periods", "recovery_periods", "total_periods", "recovered",
            ]
        )

    episodes: list[dict[str, object]] = []
    start: pd.Timestamp | None = None
    for i, (date, is_under) in enumerate(underwater.items()):
        if is_under and start is None:
            start = underwater.index[i - 1] if i > 0 else date
        elif not is_under and start is not None:
            episodes.append({"start": start, "end": date})
            start = None
    if start is not None:
        episodes.append({"start": start, "end": None})

    rows: list[dict[str, object]] = []
    for ep in episodes:
        s = ep["start"]
        e = ep["end"]
        window = dd.loc[s:] if e is None else dd.loc[s:e]
        if window.empty:
            continue
        trough_date = window.idxmin()
        depth = float(window.min())
        idx = returns.index
        s_pos = idx.get_loc(s)
        t_pos = idx.get_loc(trough_date)
        e_pos = idx.get_loc(e) if e is not None else None
        rows.append(
            {
                "peak_date": s,
                "trough_date": trough_date,
                "recovery_date": e if e is not None else pd.NaT,
                "max_drawdown": depth,
                "decline_periods": int(t_pos - s_pos),
                "recovery_periods": (
                    int(e_pos - t_pos) if e_pos is not None else np.nan
                ),
                "total_periods": (
                    int(e_pos - s_pos) if e_pos is not None else int(len(idx) - 1 - s_pos)
                ),
                "recovered": e is not None,
                "peak_wealth": float(peak.loc[s]),
            }
        )

    table = pd.DataFrame(rows).sort_values("max_drawdown").head(top)
    return table.reset_index(drop=True)


def max_drawdown_duration(returns: pd.Series) -> float:
    """Longest run of periods spent below a previous peak."""
    table = drawdown_table(returns, top=10_000)
    if table.empty:
        return 0.0
    return float(table["total_periods"].max())


def ulcer_index(returns: pd.Series) -> float:
    """Root-mean-square drawdown: depth *and* duration in one number.

    Two portfolios can share a max drawdown while one spent a month
    underwater and the other three years. The Ulcer Index separates them.
    """
    dd = drawdown_series(returns)
    return float(np.sqrt((dd**2).mean()))


# ---------------------------------------------------------------------------
# Risk decomposition
# ---------------------------------------------------------------------------


def risk_contribution(weights: np.ndarray | pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Decompose portfolio variance into per-asset risk contributions.

    Returns a Series where entries sum to 1 — each value is the share of
    total portfolio risk attributable to that asset. Because the Euler
    decomposition splits volatility and variance in the same proportions,
    this share reads correctly either way.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    total_var = float(w @ cov @ w)
    index = cov_matrix.columns if isinstance(cov_matrix, pd.DataFrame) else None
    if total_var <= 0:
        return pd.Series(np.zeros_like(w), index=index)
    marginal = cov @ w
    contribution = w * marginal / total_var
    return pd.Series(contribution, index=index)


def marginal_risk_contribution(
    weights: np.ndarray | pd.Series, cov_matrix: pd.DataFrame
) -> pd.Series:
    """``∂σ_p/∂w_i`` — the volatility added by one more unit of each asset.

    This is the number that says which position to trim first: the asset with
    the highest marginal risk per unit of expected return.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    port_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    index = cov_matrix.columns if isinstance(cov_matrix, pd.DataFrame) else None
    if port_vol <= 0:
        return pd.Series(np.zeros_like(w), index=index)
    return pd.Series((cov @ w) / port_vol, index=index)


def group_risk_contribution(
    weights: pd.Series, cov_matrix: pd.DataFrame, groups: dict[str, str]
) -> pd.Series:
    """Aggregate risk contributions up to asset-class level.

    Risk shares are additive, so the group total is simply the sum of its
    members' contributions — which is what makes "equity is 85% of our risk"
    a statement you can compute rather than estimate.
    """
    rc = risk_contribution(weights, cov_matrix)
    labels = pd.Series(
        {a: groups.get(str(a), "Unassigned") for a in rc.index}, name="group"
    )
    return rc.groupby(labels).sum().sort_values(ascending=False)
