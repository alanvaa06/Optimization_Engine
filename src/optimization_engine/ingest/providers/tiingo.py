"""Tiingo — the adapter that keeps its key out of the URL.

Tiingo accepts the token either as a query parameter or as an
``Authorization: Token …`` header, and this adapter always uses the header.
That is a small choice with a real consequence: the key never appears in a
URL, so it cannot end up in a proxy log, a crash report, or the ``str()`` of
an exception raised three frames away.

Its adjusted series are the reason to reach for it over Yahoo when the
numbers have to be defensible: ``adjClose`` and friends are recomputed from a
maintained corporate-actions history rather than carried forward from a
vendor snapshot.
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

_BASE_URL = "https://api.tiingo.com/tiingo/daily"

_COLUMN_MAP = {
    "adjOpen": F.OPEN,
    "adjHigh": F.HIGH,
    "adjLow": F.LOW,
    "adjClose": F.CLOSE,
    "close": F.CLOSE_RAW,
    "adjVolume": F.VOLUME,
}

#: Tiingo's own resampling codes.
_FREQUENCIES = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}


class Tiingo(PriceProvider):
    """End-of-day adjusted prices from Tiingo."""

    name = "tiingo"
    description = (
        "Carefully maintained split- and dividend-adjusted end-of-day prices "
        "for US and international equities and ETFs. Requires a free API key."
    )

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Adjusted OHLCV plus the raw close, key required, one symbol per call.

        Returns:
            Capabilities covering equities and ETFs, rate-limited to 50 requests a
            minute. The token travels in a header, never in the URL.
        """
        return ProviderCapabilities(
            fields=frozenset({F.OPEN, F.HIGH, F.LOW, F.CLOSE, F.CLOSE_RAW, F.VOLUME}),
            intervals=frozenset(_FREQUENCIES),
            kinds=frozenset({F.InstrumentKind.EQUITY, F.InstrumentKind.ETF}),
            requires_key=True,
            supports_batch=False,
            signup_url="https://www.tiingo.com/documentation/general/overview",
            rate_limit_per_minute=50,
            notes="Token travels in a header, never in the URL.",
        )

    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        """Fetch one symbol's adjusted end-of-day history.

        Args:
            identifier: The Tiingo ticker. Lower-cased for the request.
            request: The run's window and interval.

        Returns:
            A single-column panel carrying adjusted OHLCV and the raw close.

        Raises:
            IdentifierNotFoundError: Tiingo has no history for this ticker.
            ProviderCredentialsError: The token is missing or rejected.
            ProviderResponseError: The payload could not be parsed.
        """
        symbol = identifier.strip()
        payload = self._get_json(
            f"{_BASE_URL}/{symbol.lower()}/prices",
            params={
                "startDate": request.start.isoformat(),  # type: ignore[union-attr]
                "endDate": request.end.isoformat(),  # type: ignore[union-attr]
                "resampleFreq": _FREQUENCIES[request.interval],
                "format": "json",
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Token {self._api_key or ''}",
            },
            endpoint=f"Tiingo daily prices for {symbol}",
        )
        rows = _price_rows(payload, symbol)
        frames = _rows_to_frames(rows, identifier=identifier, symbol=symbol)
        return PricePanel.from_frames(
            frames,
            {
                identifier: SeriesMeta(
                    identifier=identifier,
                    provider_symbol=symbol,
                    provider=self.name,
                    kind=F.InstrumentKind.UNKNOWN,
                )
            },
        )


def _price_rows(payload: object, symbol: str) -> Sequence[Mapping[str, object]]:
    if isinstance(payload, dict):
        # Tiingo reports business errors in a 200 body with a `detail` key.
        detail = payload.get("detail")
        if detail:
            raise ProviderResponseError(f"Tiingo rejected {symbol!r}: {detail}")
        raise ProviderResponseError(
            f"Tiingo returned an object rather than a price list for {symbol!r}."
        )
    if not isinstance(payload, list):
        raise ProviderResponseError(
            f"Tiingo returned an unexpected payload type for {symbol!r}: "
            f"{type(payload).__name__}."
        )
    if not payload:
        raise IdentifierNotFoundError(
            f"Tiingo has no prices for symbol {symbol!r} in this window."
        )
    return payload


def _rows_to_frames(
    rows: Sequence[Mapping[str, object]], *, identifier: str, symbol: str
) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(list(rows))
    if "date" not in frame.columns:
        raise ProviderResponseError(
            f"Tiingo rows for {symbol!r} have no 'date' field."
        )
    # Tiingo stamps every bar with a UTC midnight timestamp; the tz makes the
    # index incomparable with every other provider's naive dates.
    index = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="coerce", utc=True))
    index = index.tz_localize(None).normalize()
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
            continue
        frames[target] = series.to_frame(identifier)

    if F.CLOSE not in frames:
        raise ProviderResponseError(
            f"Tiingo rows for {symbol!r} contain no adjClose column."
        )
    return frames


__all__ = ["Tiingo"]
