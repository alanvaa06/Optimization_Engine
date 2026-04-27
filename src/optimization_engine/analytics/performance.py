"""Performance metrics: annualized return/volatility, Sharpe, Sortino, drawdown."""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimization_engine.analytics.risk import (
    cvar_historic,
    kurtosis,
    semideviation,
    skewness,
    var_gaussian,
)


def drawdown(return_series: pd.Series, starting_wealth: float = 1000.0) -> pd.DataFrame:
    """Wealth index, running peak, and drawdown for a return series."""
    wealth_index = starting_wealth * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame(
        {"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns}
    )


def annualize_volatility(
    r: pd.Series | pd.DataFrame, periods_per_year: int = 252, prices: bool = False
) -> float | pd.Series:
    if prices:
        r = r.pct_change().dropna()
    return r.std() * np.sqrt(periods_per_year)


def annualize_returns(
    r: pd.Series | pd.DataFrame, periods_per_year: int = 252, prices: bool = False
) -> float | pd.Series:
    if prices:
        r = r.pct_change().dropna()
    compounded = (1 + r).prod()
    n = r.shape[0]
    return compounded ** (periods_per_year / n) - 1


def sharpe_ratio(
    r: pd.Series | pd.DataFrame, riskfree_rate: float = 0.0, periods_per_year: int = 252
) -> float | pd.Series:
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess = r - rf_per_period
    ann_excess = annualize_returns(excess, periods_per_year)
    ann_vol = annualize_volatility(r, periods_per_year)
    return ann_excess / ann_vol


def sortino_ratio(
    r: pd.Series | pd.DataFrame, riskfree_rate: float = 0.0, periods_per_year: int = 252
) -> float | pd.Series:
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess = r - rf_per_period
    ann_excess = annualize_returns(excess, periods_per_year)
    downside = semideviation(r) * np.sqrt(periods_per_year)
    return ann_excess / downside


def calmar_ratio(
    r: pd.Series | pd.DataFrame, periods_per_year: int = 252
) -> float | pd.Series:
    ann = annualize_returns(r, periods_per_year)
    if isinstance(r, pd.DataFrame):
        max_dd = r.aggregate(lambda x: drawdown(x).Drawdown.min())
    else:
        max_dd = drawdown(r).Drawdown.min()
    return ann / abs(max_dd)


def summary_stats(
    r: pd.DataFrame | pd.Series,
    periods_per_year: int = 252,
    riskfree_rate: float = 0.03,
) -> pd.DataFrame:
    """Aggregate summary statistics per column of a returns frame."""
    if isinstance(r, pd.Series):
        r = r.to_frame()
    ann_r = r.aggregate(annualize_returns, periods_per_year=periods_per_year)
    ann_vol = r.aggregate(annualize_volatility, periods_per_year=periods_per_year)
    ann_sr = r.aggregate(
        sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year
    )
    ann_sortino = r.aggregate(
        sortino_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods_per_year
    )
    dd = r.aggregate(lambda s: drawdown(s).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var = r.aggregate(var_gaussian, modified=True)
    hist_cvar = r.aggregate(cvar_historic)
    return pd.DataFrame(
        {
            "Annualized Return": ann_r,
            "Annualized Vol": ann_vol,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Cornish-Fisher VaR(5%)": cf_var,
            "Historic CVaR(5%)": hist_cvar,
            "Sharpe Ratio": ann_sr,
            "Sortino Ratio": ann_sortino,
            "Max Drawdown": dd,
        }
    )
