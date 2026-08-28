"""Multi-provider data ingestion.

The engine used to reach for whatever loader was nearest: a Yahoo call here,
a FRED call there, a spreadsheet somewhere else, each with its own column
names, its own idea of what "close" means, and its own failure mode. That is
fine until a result has to be defended, at which point nobody can say which
provider a number came from or whether it was dividend-adjusted.

This package replaces that with one path. A request goes in
(:class:`~optimization_engine.ingest.spec.IngestRequest`), a validated panel
comes out (:class:`~optimization_engine.ingest.panel.PricePanel`) with a
per-identifier record of where every column came from, and the provider is a
string in between.

Typical use::

    from optimization_engine.ingest import IngestRequest, ingest

    result = ingest(IngestRequest(
        identifiers=["SPY", "AGG", "GLD"],
        provider="yahoo",
        period="5y",
    ))
    prices = result.prices           # DataFrame the engine optimizes on
    print(result.summary())          # what actually loaded

Switching providers changes one field, and the column names do not move::

    result = ingest(request.for_provider("stooq"))

Volume is optional throughout. Index universes have none, so
:attr:`~optimization_engine.ingest.service.IngestResult.volumes` returns
``None`` and the backtest prices impact from a fixed participation rate
instead — see :mod:`optimization_engine.backtest.costs`.
"""

from optimization_engine.ingest.cache import CacheEntry, PanelCache
from optimization_engine.ingest.catalog import (
    CatalogEntry,
    catalog_entries,
    entries_for,
    lookup,
)
from optimization_engine.ingest.credentials import (
    KeyStatus,
    env_var_for,
    key_status,
    load_dotenv,
    resolve_api_key,
)
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    IngestError,
    PanelValidationError,
    ProviderConfigurationError,
    ProviderCredentialsError,
    ProviderNotFoundError,
    ProviderResponseError,
    ProviderTransientError,
)
from optimization_engine.ingest.fields import (
    CLOSE,
    HIGH,
    LOW,
    MARKET_FIELDS,
    OHLC,
    OHLCV,
    OPEN,
    PRICE_ONLY,
    VOLUME,
    InstrumentKind,
)
from optimization_engine.ingest.panel import PricePanel, SeriesMeta
from optimization_engine.ingest.providers.base import (
    PriceProvider,
    ProviderCapabilities,
)
from optimization_engine.ingest.registry import (
    available_providers,
    describe_providers,
    get_provider,
    provider_entry,
    register_provider,
)
from optimization_engine.ingest.service import (
    IdentifierOutcome,
    IngestResult,
    ingest,
)
from optimization_engine.ingest.spec import INTERVALS, PERIODS, IngestRequest

__all__ = [
    "CLOSE",
    "HIGH",
    "INTERVALS",
    "LOW",
    "MARKET_FIELDS",
    "OHLC",
    "OHLCV",
    "OPEN",
    "PERIODS",
    "PRICE_ONLY",
    "VOLUME",
    "CacheEntry",
    "CatalogEntry",
    "IdentifierNotFoundError",
    "IdentifierOutcome",
    "IngestError",
    "IngestRequest",
    "IngestResult",
    "InstrumentKind",
    "KeyStatus",
    "PanelCache",
    "PanelValidationError",
    "PriceProvider",
    "PricePanel",
    "ProviderCapabilities",
    "ProviderConfigurationError",
    "ProviderCredentialsError",
    "ProviderNotFoundError",
    "ProviderResponseError",
    "ProviderTransientError",
    "SeriesMeta",
    "available_providers",
    "catalog_entries",
    "describe_providers",
    "entries_for",
    "env_var_for",
    "get_provider",
    "ingest",
    "key_status",
    "load_dotenv",
    "lookup",
    "provider_entry",
    "register_provider",
    "resolve_api_key",
]
