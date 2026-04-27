"""Risk metrics: VaR, CVaR, semideviation, risk contributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats


def semideviation(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Negative-only standard deviation."""
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def skewness(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    demeaned = r - r.mean()
    sigma = r.std(ddof=0)
    return (demeaned**3).mean() / sigma**3


def kurtosis(r: pd.Series | pd.DataFrame) -> float | pd.Series:
    demeaned = r - r.mean()
    sigma = r.std(ddof=0)
    return (demeaned**4).mean() / sigma**4


def is_normal(r: pd.Series, level: float = 0.01) -> bool:
    """Jarque-Bera normality test, returning True if we accept normality."""
    _, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r: pd.Series | pd.DataFrame, level: float = 5) -> float | pd.Series:
    """Historic Value at Risk at the ``level`` percentile (5 = 5%)."""
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    if isinstance(r, pd.Series):
        return -np.percentile(r, level)
    raise TypeError("Expected Series or DataFrame")


def var_gaussian(
    r: pd.Series | pd.DataFrame, level: float = 5, modified: bool = False
) -> float | pd.Series:
    """Parametric (Cornish-Fisher) Value at Risk."""
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
    """Historic Conditional VaR (expected shortfall)."""
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    raise TypeError("Expected Series or DataFrame")


def risk_contribution(weights: np.ndarray | pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Decompose portfolio variance into per-asset risk contributions.

    Returns a Series where entries sum to 1 — each value is the share
    of total portfolio variance attributable to that asset.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    total_var = float(w @ cov @ w)
    if total_var <= 0:
        return pd.Series(np.zeros_like(w), index=getattr(cov_matrix, "columns", None))
    marginal = cov @ w
    contribution = w * marginal / total_var
    index = cov_matrix.columns if isinstance(cov_matrix, pd.DataFrame) else None
    return pd.Series(contribution, index=index)
