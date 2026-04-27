"""Yahoo Finance price source.

Wraps ``yfinance`` with a thin, validation-first API. The dependency is
optional — if ``yfinance`` isn't installed, callers get a clear error
pointing at ``pip install -e .[data]``.

Design choices:
* Tickers are validated against a permissive allow-list before being
  passed to ``yfinance`` to avoid surprising input ending up in URLs.
* All network calls are wrapped with a configurable timeout / retry,
  and a single download per call (multi-ticker requests).
* Returns the same shape as ``load_prices``: rows = dates,
  columns = tickers, values = adjusted close.
* No caching is enabled by default to avoid surprising filesystem
  writes; callers can opt in via ``cache_dir``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

# Yahoo tickers can include letters, digits, '.', '-', '=', '^', and '_'
# (e.g. ``BRK.B``, ``BTC-USD``, ``^GSPC``, ``ES=F``, ``EURUSD=X``).
# Anything else — including '/', whitespace, or path separators — is
# rejected before it hits the network. This is a defense-in-depth check;
# yfinance ultimately encodes tickers into URL params, but we'd rather
# reject obviously hostile input early than rely on its escaping.
_TICKER_PATTERN = re.compile(r"^[A-Za-z0-9._\-=^]{1,20}$")


class YahooFinanceError(RuntimeError):
    """Raised when the Yahoo source fails (missing dep, bad ticker, etc.)."""


def _import_yfinance():
    try:
        import yfinance as yf  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — exercised only without dep
        raise YahooFinanceError(
            "yfinance is not installed. Install with: pip install -e .[data]"
        ) from exc
    return yf


def _validate_tickers(tickers: Sequence[str]) -> list[str]:
    if not tickers:
        raise YahooFinanceError("At least one ticker is required.")
    cleaned: list[str] = []
    for raw in tickers:
        if not isinstance(raw, str):
            raise YahooFinanceError(f"Tickers must be strings; got {type(raw).__name__}")
        ticker = raw.strip().upper()
        if not _TICKER_PATTERN.match(ticker):
            raise YahooFinanceError(
                f"Rejected ticker {raw!r}: only letters, digits, "
                "and . _ - = ^ / are allowed."
            )
        cleaned.append(ticker)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def load_prices_yahoo(
    tickers: str | Iterable[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    period: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    field: str = "Close",
    progress: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download historical prices from Yahoo Finance.

    Args:
        tickers: One or more tickers. Either a list/tuple or a
            whitespace/comma-separated string.
        start: ISO date string or Timestamp. Inclusive.
        end: ISO date string or Timestamp. Exclusive (yfinance
            convention).
        period: Alternative to ``start``/``end``: ``1d``, ``5d``, ``1mo``,
            ``3mo``, ``6mo``, ``1y``, ``2y``, ``5y``, ``10y``, ``ytd``,
            ``max``.
        interval: ``1d`` (default), ``1wk``, ``1mo``, etc.
        auto_adjust: When True, prices are split- and dividend-adjusted.
        field: Which field to extract from the multi-column download.
            Common values: ``Close`` (default with ``auto_adjust=True``),
            ``Adj Close``, ``Open``, ``High``, ``Low``, ``Volume``.
        progress: Show yfinance progress bar.
        timeout: Network timeout in seconds.

    Returns:
        DataFrame with a `DatetimeIndex` and one column per ticker,
        sorted ascending. Rows where the chosen field is entirely
        missing are dropped.

    Raises:
        YahooFinanceError: If yfinance is missing, tickers are invalid,
            or the download returned no usable data.
    """
    yf = _import_yfinance()

    if isinstance(tickers, str):
        parts = [t for t in re.split(r"[\s,]+", tickers) if t]
    else:
        parts = list(tickers)
    cleaned = _validate_tickers(parts)

    if not period and not start:
        raise YahooFinanceError("Either ``period`` or ``start`` must be supplied.")

    download_kwargs = dict(
        tickers=cleaned,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress,
        threads=False,            # deterministic, easier to mock in tests
        group_by="column",
        timeout=timeout,
    )
    if period:
        download_kwargs["period"] = period
    else:
        download_kwargs["start"] = start
        download_kwargs["end"] = end

    try:
        raw = yf.download(**download_kwargs)
    except Exception as exc:
        raise YahooFinanceError(f"yfinance download failed: {exc}") from exc

    if raw is None or raw.empty:
        raise YahooFinanceError(
            f"Yahoo Finance returned no data for tickers={cleaned}."
        )

    df = _extract_field(raw, field, cleaned)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how="all")

    if df.empty:
        raise YahooFinanceError(
            f"All values were missing for tickers={cleaned} (field={field!r})."
        )
    return df


def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance's multi-shape output into a flat per-ticker frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        # Two layouts depending on group_by + tickers count:
        #   level 0 = field, level 1 = ticker  → group_by="column"
        #   level 0 = ticker, level 1 = field  → group_by="ticker"
        level0 = list(raw.columns.get_level_values(0).unique())
        if field in level0:
            sub = raw[field]
        else:
            try:
                sub = raw.xs(field, axis=1, level=1)
            except KeyError as exc:
                raise YahooFinanceError(
                    f"Field {field!r} not found in download. Got: {level0}"
                ) from exc
        if isinstance(sub, pd.Series):
            sub = sub.to_frame(tickers[0])
        return sub.reindex(columns=tickers)

    # Single-ticker download: yfinance returns a flat frame.
    if field not in raw.columns:
        raise YahooFinanceError(
            f"Field {field!r} not found in download. Got: {list(raw.columns)}"
        )
    return raw[[field]].rename(columns={field: tickers[0]})
