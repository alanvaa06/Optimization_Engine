"""Small state helpers for Streamlit UI reruns."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Hashable

import pandas as pd

YAHOO_PRICES_CACHE_KEY = "yahoo_prices_cache"


def yahoo_cache_key(
    tickers: tuple[str, ...],
    period: str,
    start: str | None,
    end: str | None,
    interval: str,
) -> tuple[Hashable, ...]:
    """Build a stable key for the Yahoo inputs that define a price download."""
    return (tuple(tickers), period, start, end, interval)


def yahoo_prices_for_rerun(
    *,
    fetch_clicked: bool,
    cache_key: tuple[Hashable, ...],
    state: MutableMapping[str, object],
    fetch_prices: Callable[[], pd.DataFrame],
) -> pd.DataFrame | None:
    """Return fetched Yahoo prices, reusing them after Streamlit button reruns."""
    cached = state.get(YAHOO_PRICES_CACHE_KEY)
    if (
        not fetch_clicked
        and isinstance(cached, dict)
        and cached.get("key") == cache_key
        and isinstance(cached.get("prices"), pd.DataFrame)
    ):
        return cached["prices"]

    if not fetch_clicked:
        return None

    prices = fetch_prices()
    state[YAHOO_PRICES_CACHE_KEY] = {"key": cache_key, "prices": prices}
    return prices
