"""Yahoo Finance — the default provider, and the one that needs no account.

Wraps ``yfinance``, which is already the library's optional market-data
dependency, and which is genuinely good at the thing the ingest layer needs
most: one request for a whole universe. :meth:`Yahoo.fetch_batch` downloads
every identifier in a single call, so a twelve-asset panel costs one round
trip rather than twelve.

Two subtleties this adapter handles so nothing downstream has to:

* **Adjustment.** ``auto_adjust=True`` makes ``Close`` a total-return series,
  which is what :data:`~optimization_engine.ingest.fields.CLOSE` means. The
  unadjusted print lives under a different flag, so
  :data:`~optimization_engine.ingest.fields.CLOSE_RAW` costs a second
  download and is only fetched when actually requested.
* **Volume.** Yahoo returns a zero-filled volume column for indices. Zero is
  not "no volume", it is "cannot trade" to any liquidity model, so an
  all-zero column is dropped and the series is marked as an index.
"""

from __future__ import annotations

import pandas as pd

from optimization_engine.data.yahoo import YahooFinanceError, extract_field
from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    ProviderResponseError,
    ProviderTransientError,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta
from optimization_engine.ingest.providers.base import PriceProvider, ProviderCapabilities
from optimization_engine.ingest.spec import IngestRequest

#: Homogenized field -> the column name yfinance uses for it.
_ADJUSTED_COLUMNS = {
    F.OPEN: "Open",
    F.HIGH: "High",
    F.LOW: "Low",
    F.CLOSE: "Close",
    F.VOLUME: "Volume",
}

_MAX_BATCH = 50


class Yahoo(PriceProvider):
    """Split- and dividend-adjusted bars from Yahoo Finance, no key required."""

    name = "yahoo"
    description = (
        "Total-return OHLCV for equities, ETFs, indices, FX and crypto. "
        "No key; unofficial API, so treat availability as best-effort."
    )

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Total-return OHLCV, keyless, whole universe in one request.

        Returns:
            Capabilities covering equities, ETFs, indices, FX, crypto and
            commodities. The API is unofficial, so treat availability as
            best-effort rather than as a service level.
        """
        return ProviderCapabilities(
            fields=frozenset(
                {F.OPEN, F.HIGH, F.LOW, F.CLOSE, F.CLOSE_RAW, F.VOLUME}
            ),
            intervals=frozenset({"1d", "1wk", "1mo"}),
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
            requires_key=False,
            supports_batch=True,
            max_batch_size=_MAX_BATCH,
            notes="One request covers the whole universe.",
        )

    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        """One identifier, by way of :meth:`fetch_batch`.

        Args:
            identifier: The Yahoo ticker.
            request: The run's window, interval and requested fields.

        Returns:
            A single-column panel. See :meth:`fetch_batch` for what can be raised.
        """
        return self.fetch_batch((identifier,), request)

    def fetch_batch(
        self, identifiers: tuple[str, ...], request: IngestRequest
    ) -> PricePanel:
        # yfinance upper-cases the tickers it echoes back, so the download
        # runs on upper-cased symbols and the columns are renamed to the
        # requested identifiers at the end. Anything else silently drops a
        # lower-case ticker as an unknown column.
        """Fetch the whole universe in a single download.

        Symbols are upper-cased for the request and the columns renamed back to
        the requested identifiers afterwards, because yfinance echoes tickers
        upper-cased and anything else silently drops a lower-case ticker as an
        unknown column.

        Args:
            identifiers: Yahoo tickers.
            request: The run's window, interval and requested fields.

        Returns:
            A panel over whichever identifiers came back with data.

        Raises:
            IdentifierNotFoundError: If no identifier returned any history.
            ProviderResponseError: If the response could not be understood.
        """
        requested_by_symbol = {i.strip().upper(): i for i in identifiers}
        symbols = list(requested_by_symbol)
        adjusted_fields = [f for f in request.fields if f in _ADJUSTED_COLUMNS]

        frames: dict[str, pd.DataFrame] = {}
        raw = self._download(symbols, request, auto_adjust=True)
        for name in adjusted_fields:
            column = _ADJUSTED_COLUMNS[name]
            try:
                frames[name] = extract_field(raw, column, symbols)
            except YahooFinanceError as exc:
                if name is F.CLOSE:
                    raise ProviderResponseError(
                        f"Yahoo returned no Close column for {', '.join(symbols)}: {exc}"
                    ) from None
                # An optional field the download did not carry is not a
                # failure; the coverage table will show it as absent.
                continue

        if F.CLOSE_RAW in request.fields:
            unadjusted = self._download(symbols, request, auto_adjust=False)
            try:
                frames[F.CLOSE_RAW] = extract_field(unadjusted, "Close", symbols)
            except YahooFinanceError:
                pass

        if F.CLOSE not in frames:
            raise ProviderResponseError(
                f"Yahoo returned no usable close prices for {', '.join(symbols)}."
            )

        kinds = {symbol: classify(symbol) for symbol in symbols}
        frames = _drop_phantom_volume(frames, kinds)
        frames = _drop_empty_identifiers(frames, symbols)

        alive = list(frames[F.CLOSE].columns)
        meta = {
            requested_by_symbol[symbol]: SeriesMeta(
                identifier=requested_by_symbol[symbol],
                provider_symbol=symbol,
                provider=self.name,
                kind=kinds[symbol],
            )
            for symbol in alive
        }
        frames = {
            name: frame.rename(columns=requested_by_symbol)
            for name, frame in frames.items()
        }
        return PricePanel.from_frames(frames, meta)

    def _download(
        self, symbols: list[str], request: IngestRequest, *, auto_adjust: bool
    ) -> pd.DataFrame:
        """One ``yf.download`` call, with the library's errors translated.

        ``end`` is exclusive in yfinance, so the request's inclusive end date
        is advanced by one bar; otherwise the last day silently disappears.
        """
        from optimization_engine.data.yahoo import _import_yfinance

        yf = _import_yfinance()
        try:
            raw = yf.download(
                tickers=symbols,
                start=str(request.start),
                end=str(request.end + pd.Timedelta(days=1)),
                interval=request.interval,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
                group_by="column",
                timeout=int(self._timeout),
            )
        except Exception as exc:
            raise ProviderTransientError(
                f"Yahoo download failed for {', '.join(symbols)}: {type(exc).__name__}."
            ) from None

        if raw is None or raw.empty:
            raise IdentifierNotFoundError(
                f"Yahoo returned no data for {', '.join(symbols)} "
                f"between {request.start} and {request.end}."
            )
        return raw


def classify(symbol: str) -> F.InstrumentKind:
    """Infer an instrument kind from Yahoo's symbol conventions.

    Yahoo does publish a ``quoteType``, but only through a per-symbol metadata
    call that costs a round trip each and is rate-limited harder than the
    price endpoint. The suffix grammar is unambiguous for exactly the kinds
    that change behaviour downstream — indices and rates — so the lookup is
    not worth its cost.

    Args:
        symbol: A Yahoo ticker.

    Returns:
        The inferred kind, or ``UNKNOWN`` when the symbol says nothing.
    """
    cleaned = symbol.strip().upper()
    if cleaned.startswith("^"):
        # ^TNX, ^IRX, ^FVX and ^TYX are Treasury yields published as levels,
        # not investable index prices.
        if cleaned in {"^TNX", "^IRX", "^FVX", "^TYX"}:
            return F.InstrumentKind.RATE
        return F.InstrumentKind.INDEX
    if cleaned.endswith("=X"):
        return F.InstrumentKind.FX
    if cleaned.endswith("=F"):
        return F.InstrumentKind.COMMODITY
    if cleaned.endswith(("-USD", "-EUR", "-USDT")):
        return F.InstrumentKind.CRYPTO
    return F.InstrumentKind.UNKNOWN


def _drop_phantom_volume(
    frames: dict[str, pd.DataFrame], kinds: dict[str, F.InstrumentKind]
) -> dict[str, pd.DataFrame]:
    """Blank out volume columns that are zeros standing in for "no volume".

    Yahoo reports 0 for every index bar. Left alone, an ADV-based cost model
    reads that as an untradeable name and charges infinite impact — the exact
    failure this whole layer exists to prevent.
    """
    volume = frames.get(F.VOLUME)
    if volume is None:
        return frames

    volume = volume.copy()
    for column in volume.columns:
        series = volume[column]
        is_index = kinds.get(column, F.InstrumentKind.UNKNOWN) is F.InstrumentKind.INDEX
        if is_index or not (series.fillna(0.0) > 0).any():
            volume[column] = float("nan")

    if volume.dropna(how="all").empty:
        frames.pop(F.VOLUME, None)
        return frames
    frames[F.VOLUME] = volume
    return frames


def _drop_empty_identifiers(
    frames: dict[str, pd.DataFrame], symbols: list[str]
) -> dict[str, pd.DataFrame]:
    """Remove identifiers Yahoo returned as an all-NaN column.

    A batch download answers for a delisted or misspelled ticker with an empty
    column rather than an error. Keeping it would put a phantom asset in the
    universe; dropping it here lets the service record it as not-found.
    """
    close = frames[F.CLOSE]
    alive = [s for s in symbols if s in close.columns and not close[s].dropna().empty]
    if not alive:
        raise IdentifierNotFoundError(
            f"Yahoo returned only empty series for {', '.join(symbols)}."
        )
    return {name: frame.reindex(columns=alive) for name, frame in frames.items()}


__all__ = ["Yahoo", "classify"]
