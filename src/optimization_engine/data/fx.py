"""Foreign-exchange handling.

Converts a multi-currency price panel into a single base currency so
the optimizer sees a homogeneous universe.

Conventions:

* ``fx_to_base[ccy]`` is the multiplier that converts **one unit** of
  ``ccy`` into the base currency. ``fx_to_base[base] == 1.0``.
* Conversion is applied at the price level:
  ``price_base[t] = price_ccy[t] * fx_to_base[ccy][t]``.
  Returns are then computed from the converted prices, which yields the
  correct currency-hedged-return-free total return for the base
  investor.

Sources:

* FRED — default. Most major USD pairs are published as ``DEXxxUS`` /
  ``DEXUSxx`` series. We invert when the published quote is the wrong
  way around for our base.
* Direct user input — pass a DataFrame to ``convert_prices_to_base``
  via ``fx_rates``.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from optimization_engine.data.fred import FREDError, load_fred_series

# Pre-computed mapping of USD pairs to FRED series IDs. The boolean
# indicates whether the series is quoted as "X per USD" (True, so to get
# X→USD we invert) or "USD per X" (False, so the value is X→USD already).
_FRED_USD_PAIRS: dict[str, tuple[str, bool]] = {
    "MXN": ("DEXMXUS", True),    # Mexican Peso per 1 USD
    "EUR": ("DEXUSEU", False),   # USD per 1 EUR
    "GBP": ("DEXUSUK", False),   # USD per 1 GBP
    "JPY": ("DEXJPUS", True),    # JPY per 1 USD
    "CAD": ("DEXCAUS", True),    # CAD per 1 USD
    "CHF": ("DEXSZUS", True),    # CHF per 1 USD
    "AUD": ("DEXUSAL", False),   # USD per 1 AUD
    "BRL": ("DEXBZUS", True),    # BRL per 1 USD
    "KRW": ("DEXKOUS", True),    # KRW per 1 USD
    "INR": ("DEXINUS", True),    # INR per 1 USD
    "CNY": ("DEXCHUS", True),    # CNY per 1 USD
    "HKD": ("DEXHKUS", True),    # HKD per 1 USD
    "SEK": ("DEXSDUS", True),    # SEK per 1 USD
    "NOK": ("DEXNOUS", True),    # NOK per 1 USD
    "DKK": ("DEXDNUS", True),    # DKK per 1 USD
    "ZAR": ("DEXSFUS", True),    # ZAR per 1 USD
    "TWD": ("DEXTAUS", True),    # TWD per 1 USD
    "SGD": ("DEXSIUS", True),    # SGD per 1 USD
    "MYR": ("DEXMAUS", True),    # MYR per 1 USD
}


SUPPORTED_CURRENCIES = sorted({"USD", *_FRED_USD_PAIRS.keys()})


class FXError(RuntimeError):
    """Raised when FX conversion can't be completed."""


def supported_currencies() -> list[str]:
    """Currencies the built-in FRED FX source can handle (out of the box)."""
    return list(SUPPORTED_CURRENCIES)


def fetch_fx_to_usd(
    currencies: Iterable[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch X→USD rates from FRED for the given currencies.

    Returns a DataFrame indexed by date with one column per currency.
    USD itself is included as a constant ``1.0`` column for convenience.
    """
    cleaned = _normalize_currencies(currencies)
    requested_usd = "USD" in cleaned
    needed = sorted({c for c in cleaned if c != "USD"})
    bad = [c for c in needed if c not in _FRED_USD_PAIRS]
    if bad:
        raise FXError(
            f"No built-in FRED mapping for currencies: {bad}. "
            f"Supported: {supported_currencies()}"
        )

    if not needed:
        # USD-only request: return a single 1.0 column on today's date.
        if requested_usd:
            today = pd.DatetimeIndex([pd.Timestamp.today().normalize()])
            return pd.DataFrame({"USD": [1.0]}, index=today)
        return pd.DataFrame()

    series_ids = [_FRED_USD_PAIRS[c][0] for c in needed]
    try:
        raw = load_fred_series(series_ids, start=start, end=end)
    except FREDError as exc:
        raise FXError(f"FRED fetch failed: {exc}") from exc

    out = pd.DataFrame(index=raw.index)
    for ccy in needed:
        sid, inverted = _FRED_USD_PAIRS[ccy]
        column = raw[sid]
        out[ccy] = (1.0 / column) if inverted else column
    if requested_usd:
        out["USD"] = 1.0
    return out


def fetch_fx_to_base(
    currencies: Iterable[str],
    base: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch X→base rates by triangulating through USD.

    Returns a DataFrame with one column per currency; ``base`` is
    included as a constant 1.0 column. Indexed by date.
    """
    base = base.upper()
    cleaned = _normalize_currencies(currencies)
    universe = sorted({base, *cleaned})

    if universe == [base]:
        return pd.DataFrame({base: [1.0]}, index=pd.DatetimeIndex([pd.Timestamp.today().normalize()]))

    fx_to_usd = fetch_fx_to_usd(universe, start=start, end=end)
    if fx_to_usd.empty:
        raise FXError(f"FRED returned no FX data for currencies={universe}.")

    if base == "USD":
        out = fx_to_usd.copy()
        out["USD"] = 1.0
    else:
        if base not in fx_to_usd.columns:
            raise FXError(f"Could not source base→USD rate for {base}.")
        base_to_usd = fx_to_usd[base]
        out = fx_to_usd.div(base_to_usd, axis=0)
        out[base] = 1.0
    return out[[c for c in universe if c in out.columns]].sort_index()


def convert_prices_to_base(
    prices: pd.DataFrame,
    asset_currency: Mapping[str, str],
    base: str,
    fx_rates: pd.DataFrame | None = None,
    fill: str = "ffill",
) -> pd.DataFrame:
    """Convert a multi-currency price panel into a single base currency.

    Args:
        prices: Price panel (rows = dates, cols = assets). Index must
            be a DatetimeIndex.
        asset_currency: ``asset -> ISO currency code``. Assets missing
            from this map are assumed to already be in ``base``.
        base: ISO code of the desired base currency.
        fx_rates: Optional DataFrame of pre-fetched X→base rates. If
            ``None``, this function fetches them from FRED for the
            range of ``prices``.
        fill: How to fill FX gaps when business-day calendars don't
            align ("ffill" by default; pass ``None`` to skip).

    Returns:
        A new DataFrame, same shape as ``prices``, with values in
        ``base`` currency.
    """
    base = base.upper()
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise FXError("Price index must be a DatetimeIndex.")

    needed = {asset_currency.get(a, base).upper() for a in prices.columns}
    if needed == {base}:
        return prices.copy()

    if fx_rates is None:
        fx_rates = fetch_fx_to_base(
            sorted(needed),
            base=base,
            start=prices.index.min(),
            end=prices.index.max(),
        )

    fx_rates = fx_rates.copy()
    fx_rates.index = pd.to_datetime(fx_rates.index)
    aligned = fx_rates.reindex(prices.index)
    if fill == "ffill":
        aligned = aligned.ffill().bfill()
    elif fill is not None:
        aligned = aligned.fillna(method=fill)

    out = prices.copy()
    for asset in prices.columns:
        ccy = asset_currency.get(asset, base).upper()
        if ccy == base:
            continue
        if ccy not in aligned.columns:
            raise FXError(f"FX rate for {ccy}->{base} not available.")
        out[asset] = prices[asset].astype(float) * aligned[ccy].astype(float)
    return out


def _normalize_currencies(currencies: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for c in currencies:
        if not isinstance(c, str):
            raise FXError(f"Currency must be a string; got {type(c).__name__}")
        code = c.strip().upper()
        if not code or len(code) != 3 or not code.isalpha():
            raise FXError(f"Invalid ISO 4217 currency code: {c!r}")
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out
