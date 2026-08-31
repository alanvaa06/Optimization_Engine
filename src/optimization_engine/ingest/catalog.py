"""One name per instrument, whatever the provider calls it.

The S&P 500 is ``^GSPC`` on Yahoo, ``^SPX`` on Stooq, ``SP500`` on FRED and
``^GSPC`` again on FMP. Without a translation layer, changing providers means
rewriting the universe, every saved scenario silently refers to a different
thing, and a panel merged from two sources has the same index in it twice
under two names.

The catalog fixes the *engine-side* name and translates on the way out. Ask
for ``SP500`` and Yahoo is queried for ``^GSPC``, FRED for ``SP500``, and the
resulting column is called ``SP500`` either way — so a scenario, a benchmark
reference and a constraint written against that name keep working when the
provider changes.

Symbols not in the catalog pass through untouched. This is a convenience for
the instruments people benchmark against constantly, not a gate: any ticker a
provider accepts still works.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from optimization_engine.ingest import fields as F


@dataclass(frozen=True)
class CatalogEntry:
    """One instrument and the symbol each provider knows it by.

    Attributes:
        key: The engine-side name. What appears in panels and configs.
        name: Human-readable description for the picker.
        kind: What sort of instrument it is — decides whether a missing
            volume column is expected.
        currency: The currency its level is quoted in.
        symbols: Provider name to that provider's symbol. A provider absent
            from this mapping cannot serve the instrument.
    """

    key: str
    name: str
    kind: F.InstrumentKind
    currency: str
    symbols: Mapping[str, str]

    def symbol_for(self, provider: str) -> str | None:
        """This instrument's symbol at one provider.

        Args:
            provider: Registered provider name, e.g. ``"stooq"``.

        Returns:
            The provider's own symbol, or ``None`` when it does not carry this
            instrument — which is what lets a universe written against catalog
            names survive a change of provider.
        """
        return self.symbols.get(provider)


def _index(
    key: str,
    name: str,
    currency: str,
    *,
    yahoo: str | None = None,
    stooq: str | None = None,
    fred: str | None = None,
    fmp: str | None = None,
) -> CatalogEntry:
    symbols = {
        "yahoo": yahoo,
        "stooq": stooq,
        "fred": fred,
        "fmp": fmp or yahoo,
    }
    return CatalogEntry(
        key=key,
        name=name,
        kind=F.InstrumentKind.INDEX,
        currency=currency,
        symbols={p: s for p, s in symbols.items() if s},
    )


#: The indices people actually benchmark against. Kept small on purpose: an
#: exhaustive symbol database is a product of its own, and a short curated
#: list that is always right beats a long one that is right most of the time.
_ENTRIES: tuple[CatalogEntry, ...] = (
    _index("SP500", "S&P 500", "USD", yahoo="^GSPC", stooq="^spx", fred="SP500"),
    _index("NASDAQ100", "Nasdaq 100", "USD", yahoo="^NDX", stooq="^ndx"),
    _index(
        "NASDAQCOM", "Nasdaq Composite", "USD",
        yahoo="^IXIC", stooq="^ndq", fred="NASDAQCOM",
    ),
    _index("DJIA", "Dow Jones Industrial Average", "USD", yahoo="^DJI", stooq="^dji", fred="DJIA"),
    _index("RUSSELL2000", "Russell 2000", "USD", yahoo="^RUT", stooq="^rut"),
    _index("VIX", "CBOE Volatility Index", "USD", yahoo="^VIX", stooq="^vix", fred="VIXCLS"),
    _index("IPC", "S&P/BMV IPC (Mexico)", "MXN", yahoo="^MXX", stooq="^mexipc"),
    _index("STOXX50", "Euro Stoxx 50", "EUR", yahoo="^STOXX50E", stooq="^stx50"),
    _index("DAX", "DAX 40", "EUR", yahoo="^GDAXI", stooq="^dax"),
    _index("FTSE100", "FTSE 100", "GBP", yahoo="^FTSE", stooq="^ukx"),
    _index("NIKKEI225", "Nikkei 225", "JPY", yahoo="^N225", stooq="^nkx"),
    _index("HANGSENG", "Hang Seng", "HKD", yahoo="^HSI", stooq="^hsi"),
    _index("BOVESPA", "Ibovespa", "BRL", yahoo="^BVSP", stooq="^bvp"),
    _index("TSX", "S&P/TSX Composite", "CAD", yahoo="^GSPTSE", stooq="^tsx"),
    _index("ASX200", "S&P/ASX 200", "AUD", yahoo="^AXJO", stooq="^axjo"),
    _index("MSCIWORLD", "MSCI World (URTH ETF proxy)", "USD", yahoo="URTH", fmp="URTH"),
    _index("MSCIEM", "MSCI Emerging Markets (EEM ETF proxy)", "USD", yahoo="EEM", fmp="EEM"),
)

_BY_KEY: Mapping[str, CatalogEntry] = {entry.key: entry for entry in _ENTRIES}


def lookup(key: str) -> CatalogEntry | None:
    """The catalog entry for an engine-side name, if there is one.

    Args:
        key: An engine-side name, e.g. ``"SP500"``. Case-insensitive.

    Returns:
        The :class:`CatalogEntry`, or ``None`` when the name is not in the
        catalog — which is not an error: a raw ticker is passed through to the
        provider unchanged.
    """
    return _BY_KEY.get(str(key).strip().upper())


def catalog_entries() -> tuple[CatalogEntry, ...]:
    """Every catalog entry, in presentation order."""
    return _ENTRIES


def entries_for(provider: str) -> tuple[CatalogEntry, ...]:
    """Catalog entries a given provider can actually serve.

    Args:
        provider: A registered provider name.

    Returns:
        Every entry carrying a symbol for that provider, in catalog order.
    """
    return tuple(entry for entry in _ENTRIES if provider in entry.symbols)


def translate(
    identifiers: tuple[str, ...], provider: str, *, passthrough: bool = False
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Map engine-side names onto one provider's symbols.

    Args:
        identifiers: Engine-side names, as validated by the request.
        provider: Registered provider name.
        passthrough: Skip translation entirely. Set for providers whose
            universe is whatever they are handed — the synthetic generator and
            local files — where a catalog name is a perfectly good column name
            and reporting it as unsupported would be nonsense.

    Returns:
        A ``(symbol_by_identifier, unsupported)`` pair. ``unsupported`` names
        the catalog instruments this provider has no symbol for — the service
        reports them as skipped rather than sending a name the API will
        reject with something unhelpful.
    """
    if passthrough:
        return {identifier: identifier for identifier in identifiers}, ()

    resolved: dict[str, str] = {}
    unsupported: list[str] = []
    for identifier in identifiers:
        entry = lookup(identifier)
        if entry is None:
            # Not a catalog instrument: pass the ticker straight through.
            resolved[identifier] = identifier
            continue
        symbol = entry.symbol_for(provider)
        if symbol is None:
            unsupported.append(identifier)
            continue
        resolved[identifier] = symbol
    return resolved, tuple(unsupported)


def kind_for(identifier: str) -> F.InstrumentKind | None:
    """The instrument kind the catalog knows for a name, if any.

    Trusted over a provider's own guess: the catalog states that ``SP500`` is
    an index, whereas the Yahoo adapter can only infer it from a caret.

    Args:
        identifier: An engine-side name.

    Returns:
        The kind, or ``None`` when the name is not in the catalog.
    """
    entry = lookup(identifier)
    return entry.kind if entry else None


def currency_for(identifier: str) -> str | None:
    """The currency the catalog knows for a name, if any.

    Args:
        identifier: An engine-side name, e.g. ``"SP500"``.

    Returns:
        An ISO code, or ``None`` when the name is not in the catalog. Trusted
        over a provider's own guess, for the same reason as :func:`kind_for`.
    """
    entry = lookup(identifier)
    return entry.currency if entry else None


__all__ = [
    "CatalogEntry",
    "catalog_entries",
    "currency_for",
    "entries_for",
    "kind_for",
    "lookup",
    "translate",
]
