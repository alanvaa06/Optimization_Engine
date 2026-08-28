"""The request that describes one ingest run.

Everything the pipeline needs is on one frozen, validated object: which
identifiers, from which provider, over what window, at what frequency, and
which fields. Nothing is read from module-level state, so the same request
replayed tomorrow asks for exactly the same thing, and a run can be recorded
in a config file, diffed, and attached to a backtest as evidence of what its
inputs were.

Validation happens in :meth:`IngestRequest.__post_init__`, before any network
call. A window that runs backwards, an interval nobody publishes, or an empty
universe are all mistakes worth catching in microseconds rather than after
eight HTTP round-trips.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import ProviderConfigurationError

#: Bar sizes the engine understands, with how many of them make a year. The
#: mapping is the single source of truth for annualization downstream, so a
#: weekly ingest cannot silently be annualized as if it were daily.
INTERVALS: Mapping[str, int] = {
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}

#: Shorthand windows, resolved against today at request-construction time.
PERIODS: Mapping[str, int] = {
    "1y": 365,
    "2y": 2 * 365,
    "3y": 3 * 365,
    "5y": 5 * 365,
    "8y": 8 * 365,
    "10y": 10 * 365,
    "20y": 20 * 365,
}

#: Identifiers are placed into URLs by every adapter. Rather than trust each
#: vendor's escaping, anything outside this grammar is rejected before it can
#: leave the process. It is deliberately permissive about the punctuation real
#: tickers use (``BRK.B``, ``BTC-USD``, ``^GSPC``, ``ES=F``, ``7203.T``) and
#: absolute about everything else — no slashes, no whitespace, no controls.
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9._\-=^:]{1,32}$")

_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class IngestRequest:
    """One reproducible description of "fetch me this data".

    Attributes:
        identifiers: The universe, in the order the panel should present it.
        provider: Registered provider name (see
            :func:`~optimization_engine.ingest.registry.available_providers`).
        start: Inclusive first date. Resolved from ``period`` when omitted.
        end: Inclusive last date. Defaults to today.
        period: Shorthand window (``"5y"``) used when ``start`` is not given.
        interval: Bar size — a key of :data:`INTERVALS`.
        fields: Which of the homogenized market fields to request. Defaults to
            :data:`~optimization_engine.ingest.fields.PRICE_ONLY`, because that
            is all the optimizer needs and asking for less is faster, smaller,
            and works on providers that publish nothing else.
        currency: Convert every series into this base currency after the
            fetch. ``None`` leaves each series in its native quote.
        require_volume: Fail the run when an identifier that *should* have
            volume returns none. Off by default: an index universe has no
            volume by construction, and the backtest is designed to price
            impact without it. Turn it on when the strategy's costs genuinely
            depend on traded volume and a silent fallback would flatter the
            result.
        max_workers: Identifiers fetched concurrently. Providers are required
            to be thread-safe; 1 forces a deterministic sequential fetch.
        cache_dir: Where to read and write the on-disk response cache.
            ``None`` disables caching entirely — no surprise filesystem writes.
        cache_ttl_seconds: How long a cached response stays fresh.
        metadata: Free-form annotations carried into the run log. Never sent
            to a provider.
    """

    identifiers: tuple[str, ...]
    provider: str = "sample"
    start: dt.date | None = None
    end: dt.date | None = None
    period: str | None = None
    interval: str = "1d"
    fields: tuple[str, ...] = F.PRICE_ONLY
    currency: str | None = None
    require_volume: bool = False
    max_workers: int = 8
    cache_dir: str | None = None
    cache_ttl_seconds: int = 24 * 60 * 60
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifiers", _clean_identifiers(self.identifiers))
        object.__setattr__(self, "provider", str(self.provider).strip().lower())
        object.__setattr__(self, "fields", F.normalize_fields(self.fields))

        if not self.provider:
            raise ProviderConfigurationError("A provider name is required.")

        if self.interval not in INTERVALS:
            raise ProviderConfigurationError(
                f"Unsupported interval {self.interval!r}. "
                f"Choose one of: {', '.join(INTERVALS)}."
            )

        if self.period is not None and self.period not in PERIODS:
            raise ProviderConfigurationError(
                f"Unsupported period {self.period!r}. "
                f"Choose one of: {', '.join(PERIODS)}."
            )

        end = _as_date(self.end, "end") or dt.date.today()
        start = _as_date(self.start, "start")
        if start is None:
            days = PERIODS[self.period or "5y"]
            start = end - dt.timedelta(days=days)
        if start >= end:
            raise ProviderConfigurationError(
                f"start ({start}) must be strictly before end ({end})."
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

        if self.currency is not None:
            currency = str(self.currency).strip().upper()
            if not _CURRENCY_PATTERN.match(currency):
                raise ProviderConfigurationError(
                    f"currency must be a three-letter ISO code; got {self.currency!r}."
                )
            object.__setattr__(self, "currency", currency)

        if int(self.max_workers) < 1:
            raise ProviderConfigurationError("max_workers must be at least 1.")
        object.__setattr__(self, "max_workers", int(self.max_workers))

        if int(self.cache_ttl_seconds) < 0:
            raise ProviderConfigurationError("cache_ttl_seconds cannot be negative.")
        object.__setattr__(self, "cache_ttl_seconds", int(self.cache_ttl_seconds))
        object.__setattr__(self, "metadata", dict(self.metadata))

    # -- derived ----------------------------------------------------------

    @property
    def periods_per_year(self) -> int:
        """How many bars of this interval make a year, for annualization."""
        return INTERVALS[self.interval]

    @property
    def wants_volume(self) -> bool:
        return F.VOLUME in self.fields

    def for_identifiers(self, identifiers: Iterable[str]) -> IngestRequest:
        """The same request, narrowed to a subset of the universe."""
        return replace(self, identifiers=tuple(identifiers))

    def for_provider(self, provider: str) -> IngestRequest:
        """The same window and fields, routed to a different provider.

        This is the whole point of the homogenized vocabulary: switching
        providers is one call, and nothing downstream can tell.
        """
        return replace(self, provider=provider)

    def fingerprint(self) -> str:
        """A stable hash of everything that affects the returned data.

        ``cache_dir``, ``max_workers`` and ``metadata`` are excluded — they
        change how the fetch runs, not what it returns, so a cached response
        stays valid across them.
        """
        payload = {
            "identifiers": list(self.identifiers),
            "provider": self.provider,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "interval": self.interval,
            "fields": list(self.fields),
            "currency": self.currency,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, object]:
        """A JSON-serializable record of the request, for configs and logs."""
        return {
            "provider": self.provider,
            "identifiers": list(self.identifiers),
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "interval": self.interval,
            "fields": list(self.fields),
            "currency": self.currency,
            "require_volume": self.require_volume,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> IngestRequest:
        """Rebuild a request from :meth:`to_dict` or a config file section."""
        known = {
            "provider", "identifiers", "start", "end", "period", "interval",
            "fields", "currency", "require_volume", "max_workers",
            "cache_dir", "cache_ttl_seconds", "metadata",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ProviderConfigurationError(
                f"Unknown ingest option(s): {', '.join(unknown)}. "
                f"Known options: {', '.join(sorted(known))}."
            )
        payload = dict(data)
        raw_fields = payload.get("fields") or F.PRICE_ONLY
        return cls(
            identifiers=tuple(payload.get("identifiers") or ()),
            provider=str(payload.get("provider", "sample")),
            start=_as_date(payload.get("start"), "start"),
            end=_as_date(payload.get("end"), "end"),
            period=payload.get("period"),  # type: ignore[arg-type]
            interval=str(payload.get("interval", "1d")),
            fields=tuple(raw_fields),  # type: ignore[arg-type]
            currency=payload.get("currency"),  # type: ignore[arg-type]
            require_volume=bool(payload.get("require_volume", False)),
            max_workers=int(payload.get("max_workers", 8)),  # type: ignore[arg-type]
            cache_dir=payload.get("cache_dir"),  # type: ignore[arg-type]
            cache_ttl_seconds=int(payload.get("cache_ttl_seconds", 24 * 60 * 60)),  # type: ignore[arg-type]
            metadata=payload.get("metadata") or {},  # type: ignore[arg-type]
        )


def _clean_identifiers(identifiers: object) -> tuple[str, ...]:
    """Split, strip, validate and de-duplicate a universe.

    Accepts a comma- or whitespace-separated string as well as an iterable,
    because that is how identifiers arrive from a CLI flag and a text box.
    """
    if isinstance(identifiers, str):
        parts: list[str] = [p for p in re.split(r"[\s,]+", identifiers) if p]
    else:
        try:
            parts = [str(i).strip() for i in identifiers]  # type: ignore[union-attr]
        except TypeError as exc:
            raise ProviderConfigurationError(
                f"identifiers must be a string or an iterable; "
                f"got {type(identifiers).__name__}."
            ) from exc

    parts = [p for p in parts if p]
    if not parts:
        raise ProviderConfigurationError("At least one identifier is required.")

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in parts:
        if not _IDENTIFIER_PATTERN.match(raw):
            raise ProviderConfigurationError(
                f"Rejected identifier {raw!r}: only letters, digits and "
                ". _ - = ^ : are allowed, up to 32 characters."
            )
        # The identifier becomes the panel's column name and therefore the key
        # every config, constraint and saved scenario is written against, so
        # its case is the caller's to choose. Providers upper-case for their
        # own queries and rename the result back; de-duplication is
        # case-insensitive because no provider distinguishes ``spy`` from
        # ``SPY``.
        folded = raw.upper()
        if folded not in seen:
            seen.add(folded)
            cleaned.append(raw)
    return tuple(cleaned)


def _as_date(value: object, label: str) -> dt.date | None:
    """Coerce a date-ish value to ``datetime.date``, or ``None``."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip()[:10])
        except ValueError as exc:
            raise ProviderConfigurationError(
                f"{label} must be an ISO date (YYYY-MM-DD); got {value!r}."
            ) from exc
    # pandas.Timestamp and anything else exposing .date()
    to_date = getattr(value, "date", None)
    if callable(to_date):
        result = to_date()
        if isinstance(result, dt.date):
            return result
    raise ProviderConfigurationError(
        f"{label} must be a date or ISO string; got {type(value).__name__}."
    )


__all__ = ["INTERVALS", "PERIODS", "IngestRequest"]
