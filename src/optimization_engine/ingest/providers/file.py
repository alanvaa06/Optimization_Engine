"""Local files — the provider for data that never came from an API.

Most real mandates run on a panel somebody maintains by hand: a spreadsheet of
fund NAVs, an extract from a custodian, a series that exists nowhere public.
Making that a *provider* rather than a special case means such a panel gets
the same validation, the same coverage table and the same provenance record as
a Yahoo pull, and can be merged with one.

Two layouts are accepted, distinguished by their columns:

* **Wide** — one row per date, one column per identifier, values are closes.
  This is what the engine's existing loaders produce and what most people have.
* **Long** — one row per (date, identifier), with named ``open`` / ``high`` /
  ``low`` / ``close`` / ``volume`` columns. The only way to express OHLCV in a
  flat file, and what a database export looks like.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    ProviderConfigurationError,
    ProviderResponseError,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta
from optimization_engine.ingest.providers.base import PriceProvider, ProviderCapabilities
from optimization_engine.ingest.spec import IngestRequest

#: Long-format column names, lower-cased, mapped onto the vocabulary.
_LONG_COLUMNS = {
    "open": F.OPEN,
    "high": F.HIGH,
    "low": F.LOW,
    "close": F.CLOSE,
    "adj_close": F.CLOSE,
    "adjclose": F.CLOSE,
    "close_raw": F.CLOSE_RAW,
    "volume": F.VOLUME,
    "vwap": F.VWAP,
}

_IDENTIFIER_COLUMNS = ("identifier", "ticker", "symbol", "asset", "id")
_DATE_COLUMNS = ("date", "datetime", "timestamp", "fecha")


class LocalFile(PriceProvider):
    """A price panel read from a CSV, Excel or Parquet file on disk."""

    name = "file"
    description = (
        "A panel you already have: CSV, Excel or Parquet, in wide "
        "(date × asset) or long (date, identifier, OHLCV) layout."
    )

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        sheet_name: str | int | None = "Precios",
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(api_key=api_key, timeout=timeout)
        self._path = Path(path) if path is not None else None
        self._sheet_name = sheet_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            fields=frozenset(F.MARKET_FIELDS),
            intervals=frozenset({"1d", "1wk", "1mo"}),
            kinds=frozenset({F.InstrumentKind.UNKNOWN}),
            requires_key=False,
            supports_batch=True,
            max_batch_size=10_000,
            accepts_any_symbol=True,
            is_offline=True,
            notes="No network. What the file says is what you get.",
        )

    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        return self.fetch_batch((identifier,), request)

    def fetch_batch(
        self, identifiers: tuple[str, ...], request: IngestRequest
    ) -> PricePanel:
        if self._path is None:
            raise ProviderConfigurationError(
                "The file provider needs a path: LocalFile(path=...) or "
                "`--file-path` on the command line."
            )
        if not self._path.is_file():
            raise ProviderConfigurationError(f"No such file: {self._path}")

        raw = self._read()
        frames = _to_field_frames(raw, source=self._path)
        frames = _slice_window(frames, request)
        frames = _select_identifiers(frames, identifiers, source=self._path)

        meta = {
            identifier: SeriesMeta(
                identifier=identifier,
                provider_symbol=identifier,
                provider=self.name,
                kind=F.InstrumentKind.UNKNOWN,
                currency=request.currency,
                name=str(self._path.name),
            )
            for identifier in frames[F.CLOSE].columns
        }
        return PricePanel.from_frames(frames, meta)

    def _read(self) -> pd.DataFrame:
        suffix = self._path.suffix.lower()  # type: ignore[union-attr]
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(self._path)
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            return pd.read_excel(self._path, sheet_name=self._sheet_name)
        if suffix == ".parquet":
            return pd.read_parquet(self._path)
        raise ProviderConfigurationError(
            f"Unsupported file extension {suffix!r}. "
            "Use .csv, .xlsx, .xls, .xlsm or .parquet."
        )


def _to_field_frames(raw: pd.DataFrame, *, source: Path) -> dict[str, pd.DataFrame]:
    """Detect the layout and return homogenized per-field frames."""
    lowered = {str(c).strip().lower(): c for c in raw.columns}
    identifier_column = next(
        (lowered[c] for c in _IDENTIFIER_COLUMNS if c in lowered), None
    )
    date_column = next((lowered[c] for c in _DATE_COLUMNS if c in lowered), None)

    if identifier_column is not None and date_column is not None:
        return _from_long(raw, date_column, identifier_column, source=source)

    # Wide layout: the first column is the date index and every remaining
    # numeric column is one asset's close.
    frame = raw.copy()
    index_column = date_column or frame.columns[0]
    index = pd.DatetimeIndex(pd.to_datetime(frame[index_column], errors="coerce"))
    frame = frame.drop(columns=[index_column])
    frame = frame.loc[index.notna()]
    index = index[index.notna()]
    frame.index = index
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").sort_index()
    if numeric.empty or numeric.shape[1] == 0:
        raise ProviderResponseError(
            f"{source.name} has no numeric price columns after parsing its "
            "first column as dates."
        )
    return {F.CLOSE: numeric}


def _from_long(
    raw: pd.DataFrame, date_column: object, identifier_column: object, *, source: Path
) -> dict[str, pd.DataFrame]:
    """Pivot a long-format file into one frame per field."""
    frame = raw.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame[date_column], errors="coerce"))
    frame = frame.loc[index.notna()].copy()
    frame["__date__"] = index[index.notna()]
    frame["__id__"] = frame[identifier_column].astype(str).str.strip().str.upper()

    frames: dict[str, pd.DataFrame] = {}
    for column in raw.columns:
        target = _LONG_COLUMNS.get(str(column).strip().lower())
        if target is None or target in frames:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        pivoted = (
            pd.DataFrame({"__date__": frame["__date__"], "__id__": frame["__id__"], "v": values})
            .pivot_table(index="__date__", columns="__id__", values="v", aggfunc="last")
            .sort_index()
        )
        pivoted.columns.name = None
        pivoted.index.name = "date"
        if target is F.VOLUME and not (pivoted.fillna(0.0) > 0).any().any():
            continue
        frames[target] = pivoted

    if F.CLOSE not in frames:
        raise ProviderResponseError(
            f"{source.name} is in long format but has no close column "
            f"(looked for: {', '.join(sorted(_LONG_COLUMNS))})."
        )
    return frames


def _slice_window(
    frames: dict[str, pd.DataFrame], request: IngestRequest
) -> dict[str, pd.DataFrame]:
    """Trim every frame to the request's window.

    A file usually holds more history than the run wants. Trimming here rather
    than downstream keeps the panel honest: its index is the window that was
    asked for, and the coverage table reports what the file actually had
    inside it.
    """
    start = pd.Timestamp(request.start)
    end = pd.Timestamp(request.end)
    return {
        name: frame.loc[(frame.index >= start) & (frame.index <= end)]
        for name, frame in frames.items()
    }


def _select_identifiers(
    frames: dict[str, pd.DataFrame], identifiers: tuple[str, ...], *, source: Path
) -> dict[str, pd.DataFrame]:
    """Keep the requested identifiers, matching case-insensitively."""
    close = frames[F.CLOSE]
    by_upper = {str(c).strip().upper(): c for c in close.columns}
    # Files are written by people, so a column may be ``Cash`` where the run
    # asked for ``CASH``. Match without regard to case, then present the
    # column under the name the caller used.
    pairs = [
        (by_upper[i.strip().upper()], i)
        for i in identifiers
        if i.strip().upper() in by_upper
    ]
    if not pairs:
        raise IdentifierNotFoundError(
            f"{source.name} has none of the requested identifiers "
            f"({', '.join(identifiers)}). It contains: "
            f"{', '.join(sorted(by_upper)[:12])}"
            f"{'…' if len(by_upper) > 12 else ''}."
        )
    wanted = [original for original, _ in pairs]
    renamed = dict(pairs)
    return {
        name: frame.reindex(columns=wanted).rename(columns=renamed)
        for name, frame in frames.items()
    }


__all__ = ["LocalFile"]
