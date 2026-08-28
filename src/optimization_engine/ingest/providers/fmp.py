"""Financial Modeling Prep — the reference key-based provider.

FMP is the adapter to copy when adding a commercial source. It shows the three
things a paid API forces you to get right: the key travels in a query
parameter and therefore must never reach a log or an exception, the response
shape has changed between API generations and both are still in the wild, and
the free tier answers "you need to pay for this" with a status code that is
neither an auth failure nor a not-found.

Adjustment convention: FMP publishes ``close`` unadjusted and ``adjClose``
adjusted for splits and dividends, so ``adjClose`` becomes
:data:`~optimization_engine.ingest.fields.CLOSE` and ``close`` becomes
:data:`~optimization_engine.ingest.fields.CLOSE_RAW`. Getting that backwards
is the single most common way to produce a backtest that quietly ignores
dividends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    ProviderResponseError,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta
from optimization_engine.ingest.providers.base import PriceProvider, ProviderCapabilities
from optimization_engine.ingest.spec import IngestRequest

_BASE_URL = "https://financialmodelingprep.com/api/v3/historical-price-full"

#: FMP field name -> homogenized name. ``adjClose`` is the total-return
#: series; ``close`` is the raw print.
_COLUMN_MAP = {
    "open": F.OPEN,
    "high": F.HIGH,
    "low": F.LOW,
    "adjClose": F.CLOSE,
    "close": F.CLOSE_RAW,
    "volume": F.VOLUME,
    "vwap": F.VWAP,
}


class FinancialModelingPrep(PriceProvider):
    """Daily OHLCV plus adjusted closes from Financial Modeling Prep."""

    name = "fmp"
    description = (
        "Split- and dividend-adjusted OHLCV for global equities, ETFs and "
        "indices, with a generous free tier. Requires a free API key."
    )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            fields=frozenset(
                {F.OPEN, F.HIGH, F.LOW, F.CLOSE, F.CLOSE_RAW, F.VOLUME, F.VWAP}
            ),
            intervals=frozenset({"1d"}),
            kinds=frozenset(
                {
                    F.InstrumentKind.EQUITY,
                    F.InstrumentKind.ETF,
                    F.InstrumentKind.INDEX,
                    F.InstrumentKind.FX,
                    F.InstrumentKind.CRYPTO,
                    F.InstrumentKind.COMMODITY,
                }
            ),
            requires_key=True,
            supports_batch=False,
            signup_url="https://site.financialmodelingprep.com/developer/docs",
            rate_limit_per_minute=300,
            notes="Adjusted closes and true VWAP; one request per symbol.",
        )

    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        symbol = identifier.strip()
        payload = self._get_json(
            f"{_BASE_URL}/{_quote(symbol)}",
            params={
                "from": request.start.isoformat(),  # type: ignore[union-attr]
                "to": request.end.isoformat(),  # type: ignore[union-attr]
                "serietype": "line" if request.fields == F.PRICE_ONLY else "",
            },
            # The key goes here and nowhere else, so no error path can
            # interpolate it: the base class raises on status codes only.
            secret_params={"apikey": self._api_key or ""},
            endpoint=f"FMP historical prices for {symbol}",
        )
        rows = _historical_rows(payload, symbol)
        frames = _rows_to_frames(rows, identifier=identifier, symbol=symbol)
        return PricePanel.from_frames(
            frames,
            {
                identifier: SeriesMeta(
                    identifier=identifier,
                    provider_symbol=symbol,
                    provider=self.name,
                    kind=classify(symbol),
                    currency=_payload_currency(payload),
                )
            },
        )


def classify(symbol: str) -> F.InstrumentKind:
    """Infer an instrument kind from FMP's symbol conventions."""
    cleaned = symbol.strip().upper()
    if cleaned.startswith("^"):
        return F.InstrumentKind.INDEX
    if cleaned.endswith("USD") and len(cleaned) == 6:
        return F.InstrumentKind.FX
    return F.InstrumentKind.UNKNOWN


def _quote(symbol: str) -> str:
    """Percent-encode a symbol for the path segment.

    ``^GSPC`` is a legal FMP index symbol and an illegal raw URL path
    character, so this is load-bearing rather than defensive.
    """
    from urllib.parse import quote

    return quote(symbol, safe="")


def _historical_rows(payload: object, symbol: str) -> Sequence[Mapping[str, object]]:
    """Pull the observation list out of either FMP response generation.

    The v3 endpoint wraps rows in ``{"symbol": ..., "historical": [...]}``;
    the newer endpoints return the list directly. Both are live, and which one
    a key sees depends on when the key was issued — so the adapter accepts
    either rather than pinning a version that half of users cannot call.
    """
    if isinstance(payload, dict):
        if "Error Message" in payload:
            raise ProviderResponseError(
                f"FMP rejected the request for {symbol!r}: {payload['Error Message']}"
            )
        rows = payload.get("historical")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ProviderResponseError(
            f"FMP returned an unexpected payload type for {symbol!r}: "
            f"{type(payload).__name__}."
        )

    if not rows:
        raise IdentifierNotFoundError(
            f"FMP has no historical prices for symbol {symbol!r} in this window."
        )
    if not isinstance(rows, list):
        raise ProviderResponseError(
            f"FMP returned a non-list 'historical' block for {symbol!r}."
        )
    return rows


def _payload_currency(payload: object) -> str | None:
    if isinstance(payload, dict):
        currency = payload.get("currency")
        if isinstance(currency, str) and len(currency) == 3:
            return currency.upper()
    return None


def _rows_to_frames(
    rows: Sequence[Mapping[str, object]], *, identifier: str, symbol: str
) -> dict[str, pd.DataFrame]:
    """Convert FMP observation dicts into homogenized single-column frames."""
    frame = pd.DataFrame(list(rows))
    if "date" not in frame.columns:
        raise ProviderResponseError(
            f"FMP rows for {symbol!r} have no 'date' field; got {list(frame.columns)}."
        )
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce"))
    frame = frame.loc[index.notna()]
    index = index[index.notna()]

    frames: dict[str, pd.DataFrame] = {}
    for source, target in _COLUMN_MAP.items():
        if source not in frame.columns:
            continue
        series = pd.to_numeric(frame[source], errors="coerce")
        series.index = index
        series = series.sort_index()
        if target is F.VOLUME and not (series.fillna(0.0) > 0).any():
            # An index or a fund with no reported turnover: absent, not zero.
            continue
        frames[target] = series.to_frame(identifier)

    if F.CLOSE not in frames:
        # ``serietype=line`` responses carry only ``close``. It is unadjusted,
        # but it is the only price there is, so promote it and say so.
        raw = frames.pop(F.CLOSE_RAW, None)
        if raw is None:
            raise ProviderResponseError(
                f"FMP rows for {symbol!r} contain neither adjClose nor close."
            )
        frames[F.CLOSE] = raw
    return frames


__all__ = ["FinancialModelingPrep", "classify"]
