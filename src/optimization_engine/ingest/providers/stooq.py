"""Stooq — keyless daily history, and the reason index universes work.

Stooq publishes a plain CSV per symbol at a stable URL with no key, no quota
form, and no client library. That makes it the provider that keeps the "just
show me an index backtest" path working on a laptop with no credentials: the
world's major indices (``^SPX``, ``^NDX``, ``^DJI``, ``^MEXIPC``, ``^DAX``)
are one HTTP GET away.

It also makes the volume story concrete. Stooq's index CSVs either omit the
volume column or fill it with zeros, and this adapter refuses to launder that
into a number: for an index it drops volume entirely and marks the series
:attr:`~optimization_engine.ingest.fields.InstrumentKind.INDEX`, so the cost
model downstream knows it must price impact without an ADV rather than
reading a zero as "no liquidity at all".

Caveats worth knowing: the CSVs are split-adjusted but not dividend-adjusted,
so ``m_close`` here is a price return, not a total return. The adapter says so
in :attr:`Stooq.description` rather than hiding it.
"""

from __future__ import annotations

import csv
import io

import pandas as pd

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    ProviderResponseError,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta
from optimization_engine.ingest.providers.base import PriceProvider, ProviderCapabilities
from optimization_engine.ingest.spec import IngestRequest

_BASE_URL = "https://stooq.com/q/d/l/"

#: Stooq's own interval codes.
_INTERVAL_CODES = {"1d": "d", "1wk": "w", "1mo": "m"}

_COLUMN_MAP = {
    "open": F.OPEN,
    "high": F.HIGH,
    "low": F.LOW,
    "close": F.CLOSE,
    "volume": F.VOLUME,
}


class Stooq(PriceProvider):
    """Keyless daily/weekly/monthly bars from stooq.com."""

    name = "stooq"
    description = (
        "Keyless CSV history for world indices, FX and major equities. "
        "Split-adjusted (not dividend-adjusted), so its close is a price "
        "return; indices carry no volume."
    )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            fields=frozenset({F.OPEN, F.HIGH, F.LOW, F.CLOSE, F.VOLUME}),
            intervals=frozenset(_INTERVAL_CODES),
            kinds=frozenset(
                {
                    F.InstrumentKind.INDEX,
                    F.InstrumentKind.EQUITY,
                    F.InstrumentKind.ETF,
                    F.InstrumentKind.FX,
                    F.InstrumentKind.COMMODITY,
                }
            ),
            requires_key=False,
            supports_batch=False,
            signup_url=None,
            notes="No key, no quota. Best free source for index levels.",
        )

    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        symbol = identifier.strip()
        kind = classify(symbol)
        body = self._get_text(
            _BASE_URL,
            params={
                "s": symbol.lower(),
                "d1": request.start.strftime("%Y%m%d"),  # type: ignore[union-attr]
                "d2": request.end.strftime("%Y%m%d"),  # type: ignore[union-attr]
                "i": _INTERVAL_CODES[request.interval],
            },
            endpoint=f"stooq daily CSV for {symbol}",
        )
        frames = self._parse_csv(body, identifier=identifier, symbol=symbol, kind=kind)
        return PricePanel.from_frames(
            frames,
            {
                identifier: SeriesMeta(
                    identifier=identifier,
                    provider_symbol=symbol,
                    provider=self.name,
                    kind=kind,
                )
            },
        )

    @staticmethod
    def _parse_csv(
        body: str,
        *,
        identifier: str,
        symbol: str,
        kind: F.InstrumentKind,
    ) -> dict[str, pd.DataFrame]:
        """Turn a Stooq CSV body into homogenized single-column frames.

        Stooq answers an unknown symbol with a 200 and a one-line body, so
        "not found" has to be recognized from the content rather than the
        status code.
        """
        text = body.strip()
        if not text or "No data" in text[:200] or "Exceeded" in text[:200]:
            if "Exceeded" in text[:200]:
                raise ProviderResponseError(
                    f"Stooq refused the request for {symbol!r}: daily limit exceeded."
                )
            raise IdentifierNotFoundError(f"Stooq has no data for symbol {symbol!r}.")

        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or "Date" not in reader.fieldnames:
            raise IdentifierNotFoundError(
                f"Stooq returned no price table for symbol {symbol!r}."
            )

        available = {
            name.lower(): name for name in reader.fieldnames if name.lower() in _COLUMN_MAP
        }
        records: dict[str, list] = {"date": []}
        for lower in available:
            records[lower] = []

        for row in reader:
            raw_date = (row.get("Date") or "").strip()
            if not raw_date:
                continue
            records["date"].append(raw_date)
            for lower, original in available.items():
                records[lower].append(_to_float(row.get(original)))

        if not records["date"]:
            raise IdentifierNotFoundError(
                f"Stooq returned an empty price table for symbol {symbol!r}."
            )

        index = pd.DatetimeIndex(pd.to_datetime(records["date"], errors="coerce"))
        frames: dict[str, pd.DataFrame] = {}
        for lower, values in records.items():
            if lower == "date":
                continue
            series = pd.Series(values, index=index, dtype="float64").sort_index()
            if lower == "volume":
                # An index has no volume. Stooq expresses that as a zero column
                # or a missing one; either way, publishing zeros would let a
                # liquidity model conclude the name cannot be traded at all.
                if kind is F.InstrumentKind.INDEX or not (series.fillna(0.0) > 0).any():
                    continue
            frames[_COLUMN_MAP[lower]] = series.to_frame(identifier)

        if F.CLOSE not in frames:
            raise ProviderResponseError(
                f"Stooq response for {symbol!r} has no Close column."
            )
        return frames


def classify(symbol: str) -> F.InstrumentKind:
    """Guess an instrument kind from a Stooq symbol.

    Stooq publishes no metadata endpoint, so the symbol itself is the only
    signal: a leading ``^`` is always an index, and a six-letter all-alpha
    symbol is one of its FX crosses.
    """
    cleaned = symbol.strip().upper()
    if cleaned.startswith("^"):
        return F.InstrumentKind.INDEX
    if len(cleaned) == 6 and cleaned.isalpha():
        return F.InstrumentKind.FX
    return F.InstrumentKind.EQUITY


def _to_float(raw: str | None) -> float:
    """Parse a CSV cell, mapping Stooq's blanks and ``N/A`` to NaN."""
    if raw is None:
        return float("nan")
    text = raw.strip()
    if not text or text in {"N/A", "-", "null"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


__all__ = ["Stooq", "classify"]
