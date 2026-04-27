"""Covariance and expected-return estimation.

Wraps a few common estimators (sample, Ledoit-Wolf, OAS, EWMA) with a
single, ergonomic API. We deliberately avoid hard-coding ``riskfolio-lib``;
if it's installed we route ``covariance_method="shrink"`` through it,
otherwise we fall back to ``sklearn``'s Ledoit-Wolf shrinkage.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

CovarianceMethod = Literal[
    "sample", "ledoit_wolf", "oas", "shrink", "ewma", "semi"
]


def _sample(returns: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    return returns.cov(ddof=ddof)


def _ledoit_wolf(returns: pd.DataFrame) -> pd.DataFrame:
    from sklearn.covariance import LedoitWolf

    cov = LedoitWolf().fit(returns.values).covariance_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _oas(returns: pd.DataFrame) -> pd.DataFrame:
    from sklearn.covariance import OAS

    cov = OAS().fit(returns.values).covariance_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def _ewma(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """Exponentially weighted (RiskMetrics-style) covariance."""
    r = returns - returns.mean()
    weights = (1 - lam) * lam ** np.arange(len(r))[::-1]
    weights /= weights.sum()
    cov = (r.T * weights) @ r
    return cov


def _semi(returns: pd.DataFrame) -> pd.DataFrame:
    """Semi-covariance (only negative deviations from the mean)."""
    deviation = returns - returns.mean()
    deviation = deviation.where(deviation < 0, 0.0)
    return deviation.cov(ddof=1)


def _shrink(returns: pd.DataFrame) -> pd.DataFrame:
    try:
        import riskfolio as rf

        return rf.ParamsEstimation.covar_matrix(returns, method="shrink")
    except Exception:
        return _ledoit_wolf(returns)


def covariance_matrix(
    returns: pd.DataFrame,
    method: CovarianceMethod = "ledoit_wolf",
    annualize: bool = True,
    periods_per_year: int = 252,
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """Estimate a covariance matrix on returns.

    Set ``annualize=True`` to scale by ``periods_per_year`` (e.g. daily ⇒ annual).
    """
    if method == "sample":
        cov = _sample(returns)
    elif method == "ledoit_wolf":
        cov = _ledoit_wolf(returns)
    elif method == "oas":
        cov = _oas(returns)
    elif method == "shrink":
        cov = _shrink(returns)
    elif method == "ewma":
        cov = _ewma(returns, lam=ewma_lambda)
    elif method == "semi":
        cov = _semi(returns)
    else:
        raise ValueError(f"Unknown covariance method: {method}")

    if annualize:
        cov = cov * periods_per_year
    return cov


def expected_returns_from_history(
    returns: pd.DataFrame,
    method: Literal["mean", "ema", "capm"] = "mean",
    periods_per_year: int = 252,
    span: int = 180,
    market_return: float | None = None,
    risk_free_rate: float = 0.0,
    market_weights: pd.Series | None = None,
    cov_matrix: pd.DataFrame | None = None,
) -> pd.Series:
    """Build an expected-return vector from realized history.

    * ``mean`` — annualized historical mean.
    * ``ema``  — exponentially-weighted mean with the given ``span``.
    * ``capm`` — implied returns from a single-factor CAPM where the market
      portfolio is approximated by ``market_weights`` (defaulting to
      equal weights).
    """
    if method == "mean":
        return ((1 + returns).prod() ** (periods_per_year / len(returns))) - 1
    if method == "ema":
        ema = returns.ewm(span=span, adjust=False).mean().iloc[-1]
        return (1 + ema) ** periods_per_year - 1
    if method == "capm":
        if cov_matrix is None:
            cov_matrix = covariance_matrix(
                returns, method="ledoit_wolf", periods_per_year=periods_per_year
            )
        if market_weights is None:
            market_weights = pd.Series(
                np.ones(returns.shape[1]) / returns.shape[1], index=returns.columns
            )
        market_weights = market_weights.reindex(returns.columns).fillna(0.0)
        market_return_est = market_return
        if market_return_est is None:
            mkt = (returns * market_weights).sum(axis=1)
            market_return_est = (
                (1 + mkt).prod() ** (periods_per_year / len(mkt)) - 1
            )
        market_var = float(market_weights.values @ cov_matrix.values @ market_weights.values)
        betas = (cov_matrix.values @ market_weights.values) / market_var
        return pd.Series(
            risk_free_rate + betas * (market_return_est - risk_free_rate),
            index=returns.columns,
        )
    raise ValueError(f"Unknown expected-return method: {method}")
