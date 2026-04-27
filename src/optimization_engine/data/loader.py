"""Data loaders for prices/returns from Excel, CSV, or synthetic samples."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_prices(
    path: str | Path,
    sheet_name: str | int | None = "Precios",
    index_col: str | int | None = 0,
    date_format: str | None = None,
) -> pd.DataFrame:
    """Load a price panel from an Excel or CSV file.

    The resulting DataFrame has a `DatetimeIndex` and one column per asset.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(p, sheet_name=sheet_name, index_col=index_col, parse_dates=True)
    elif suf == ".csv":
        df = pd.read_csv(p, index_col=index_col, parse_dates=True)
    elif suf == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file extension: {suf}")
    if date_format:
        df.index = pd.to_datetime(df.index, format=date_format)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.dropna(how="all")


def prices_to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    if log:
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def sample_dataset(
    n_periods: int = 252 * 8,
    seed: int = 42,
    assets: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Synthetic but realistic-ish price panel for tutorials and tests.

    Generates correlated daily log-returns from a multivariate normal with
    a hand-tuned covariance structure across asset classes, then exponentiates
    to a price series starting at 100.
    """
    rng = np.random.default_rng(seed)

    default_assets = {
        "US_Equity":     ("Equity",       0.08,  0.18),
        "Intl_Equity":   ("Equity",       0.07,  0.20),
        "EM_Equity":     ("Equity",       0.09,  0.24),
        "Real_Estate":   ("Alternatives", 0.06,  0.19),
        "Commodities":   ("Alternatives", 0.04,  0.22),
        "Infra":         ("Alternatives", 0.07,  0.16),
        "Gold":          ("Alternatives", 0.04,  0.15),
        "US_Treasuries": ("FixedIncome",  0.03,  0.07),
        "TIPS":          ("FixedIncome",  0.03,  0.06),
        "IG_Credit":     ("FixedIncome",  0.04,  0.08),
        "HY_Credit":     ("FixedIncome",  0.05,  0.11),
        "EM_Debt":       ("FixedIncome",  0.05,  0.10),
        "Cash":          ("FixedIncome",  0.025, 0.005),
    }

    if assets is not None:
        keys = list(assets)
        meta = {k: default_assets[k] for k in keys if k in default_assets}
    else:
        meta = default_assets
        keys = list(meta.keys())

    n = len(keys)
    mu = np.array([meta[k][1] / 252 for k in keys])
    sigma = np.array([meta[k][2] / np.sqrt(252) for k in keys])

    groups = [meta[k][0] for k in keys]
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if groups[i] == groups[j]:
                base = 0.7 if groups[i] == "Equity" else 0.55
            elif {"Equity", "Alternatives"} == {groups[i], groups[j]}:
                base = 0.45
            elif {"Equity", "FixedIncome"} == {groups[i], groups[j]}:
                base = -0.05
            else:
                base = 0.2
            jitter = rng.uniform(-0.05, 0.05)
            corr[i, j] = corr[j, i] = float(np.clip(base + jitter, -0.95, 0.95))

    cov = corr * np.outer(sigma, sigma)
    cov = (cov + cov.T) / 2
    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, 1e-10, None)
    cov = (eigvec * eigval) @ eigvec.T

    log_rets = rng.multivariate_normal(mu, cov, size=n_periods)
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_periods)
    return pd.DataFrame(prices, index=dates, columns=keys)
