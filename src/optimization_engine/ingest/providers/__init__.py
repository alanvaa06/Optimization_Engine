"""Provider adapters: one class per data source.

Each adapter translates a vendor's schema into the homogenized vocabulary in
:mod:`optimization_engine.ingest.fields` and declares, in
:class:`~optimization_engine.ingest.providers.base.ProviderCapabilities`, what
it can and cannot serve. Nothing here knows about caching, concurrency or
currency conversion — that is
:func:`~optimization_engine.ingest.service.ingest`'s job.

To add your own, subclass
:class:`~optimization_engine.ingest.providers.base.PriceProvider` and call
:func:`~optimization_engine.ingest.registry.register_provider`; no core code
needs to change.
"""

from optimization_engine.ingest.providers.base import (
    PriceProvider,
    ProviderCapabilities,
    ProviderInfo,
)
from optimization_engine.ingest.providers.file import LocalFile
from optimization_engine.ingest.providers.fmp import FinancialModelingPrep
from optimization_engine.ingest.providers.fred import Fred
from optimization_engine.ingest.providers.sample import Sample
from optimization_engine.ingest.providers.stooq import Stooq
from optimization_engine.ingest.providers.tiingo import Tiingo
from optimization_engine.ingest.providers.yahoo import Yahoo

__all__ = [
    "FinancialModelingPrep",
    "Fred",
    "LocalFile",
    "PriceProvider",
    "ProviderCapabilities",
    "ProviderInfo",
    "Sample",
    "Stooq",
    "Tiingo",
    "Yahoo",
]
