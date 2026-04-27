"""Relative-performance metrics versus a benchmark."""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from optimization_engine.analytics.performance import (
    annualize_returns,
    annualize_volatility,
)


def spread(series_1: pd.Series | pd.DataFrame, series_2: pd.Series | pd.DataFrame) -> pd.DataFrame | pd.Series:
    if isinstance(series_2, pd.DataFrame):
        series_2 = series_2.iloc[:, 0]
    if isinstance(series_1, pd.DataFrame):
        return series_1.sub(series_2, axis=0)
    return series_1 - series_2


def _ensure_frame(s: pd.Series | pd.DataFrame) -> pd.DataFrame:
    return s.to_frame() if isinstance(s, pd.Series) else s


def up_capture(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """Geometric up-capture ratio."""
    r = _ensure_frame(r)
    rb = _ensure_frame(rb)
    bmk_up = rb[rb > 0].dropna(how="all")
    n_b = bmk_up.count()
    bmk_geo = (1 + bmk_up).prod() ** (1 / n_b) - 1
    bmk_geo_value = float(bmk_geo.iloc[0])
    out = {}
    for col in r.columns:
        rs = r[col]
        rs_up = rs.loc[rb.iloc[:, 0] > 0].dropna()
        n = rs_up.count()
        if n == 0:
            out[col] = float("nan")
        else:
            geo = (1 + rs_up).prod() ** (1 / n) - 1
            out[col] = geo / bmk_geo_value if bmk_geo_value else float("nan")
    return pd.Series(out)


def down_capture(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """Geometric down-capture ratio."""
    r = _ensure_frame(r)
    rb = _ensure_frame(rb)
    bmk_dn = rb[rb < 0].dropna(how="all")
    n_b = bmk_dn.count()
    bmk_geo = (1 + bmk_dn).prod() ** (1 / n_b) - 1
    bmk_geo_value = float(bmk_geo.iloc[0])
    out = {}
    for col in r.columns:
        rs = r[col]
        rs_dn = rs.loc[rb.iloc[:, 0] < 0].dropna()
        n = rs_dn.count()
        if n == 0:
            out[col] = float("nan")
        else:
            geo = (1 + rs_dn).prod() ** (1 / n) - 1
            out[col] = geo / bmk_geo_value if bmk_geo_value else float("nan")
    return pd.Series(out)


def capture_ratio(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    return up_capture(r, rb) / down_capture(r, rb)


def beta(r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame) -> pd.Series:
    """OLS beta of each column of ``r`` against benchmark ``rb``."""
    r = _ensure_frame(r)
    rb = _ensure_frame(rb).copy()
    rb["__alpha__"] = 1.0
    out = {}
    for col in r.columns:
        model = sm.OLS(r[col], rb).fit()
        out[col] = float(model.params.iloc[0])
    return pd.Series(out, name="Beta")


def information_ratio(
    r: pd.Series | pd.DataFrame, rb: pd.Series | pd.DataFrame, periods_per_year: int = 252
) -> pd.Series:
    if isinstance(rb, pd.DataFrame):
        rb_series = rb.iloc[:, 0]
    else:
        rb_series = rb
    excess = spread(r, rb_series)
    ann_excess = annualize_returns(r, periods_per_year) - annualize_returns(
        rb_series, periods_per_year
    )
    tracking_error = annualize_volatility(excess, periods_per_year)
    return ann_excess / tracking_error


def summary_relative(
    r: pd.Series | pd.DataFrame,
    rb: pd.Series | pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    if isinstance(rb, pd.DataFrame):
        rb_series = rb.iloc[:, 0]
    else:
        rb_series = rb
    excess = spread(r, rb_series)
    ann_excess = annualize_returns(r, periods_per_year) - annualize_returns(
        rb_series, periods_per_year
    )
    tracking_error = annualize_volatility(excess, periods_per_year)
    ir = ann_excess / tracking_error
    b = beta(r, rb_series)
    up = up_capture(r, rb_series)
    down = down_capture(r, rb_series)
    return pd.DataFrame(
        {
            "Annualized Excess": ann_excess,
            "Annualized T.E.": tracking_error,
            "Information Ratio": ir,
            "Beta": b,
            "Up Capture": up,
            "Down Capture": down,
            "Capture": up / down,
        }
    )
