"""Where provider names become provider objects.

The registry is what lets a provider be a string in a config file. It keeps
the built-ins, resolves aliases, injects the right API key, and — through
:func:`register_provider` — accepts providers that live outside this package
entirely, so a desk with an in-house tick store can plug it in without
forking anything.

Two deliberate choices:

* **Lazy construction.** Entries hold a factory, not an instance, so listing
  the available providers never builds an HTTP client or reads a key.
  :func:`describe_providers` is safe to call on every Streamlit rerun.
* **A real error for an unknown name.** Data-Curator returns a null-object
  provider that answers every call with "not found", which keeps a run alive
  at the cost of hiding a typo until the panel comes back empty. Here a bad
  name raises immediately and the message lists what is available — the
  universe is small enough that the suggestion is always useful.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.credentials import key_status, resolve_api_key
from optimization_engine.ingest.errors import ProviderNotFoundError
from optimization_engine.ingest.providers.base import PriceProvider, ProviderCapabilities
from optimization_engine.ingest.spec import INTERVALS

ProviderFactory = Callable[..., PriceProvider]


@dataclass(frozen=True)
class ProviderEntry:
    """One registered provider, described without being instantiated."""

    name: str
    factory: ProviderFactory
    description: str
    capabilities: ProviderCapabilities
    aliases: tuple[str, ...] = ()

    @property
    def requires_key(self) -> bool:
        """Whether this provider needs an API key to be usable."""
        return self.capabilities.requires_key


_REGISTRY: dict[str, ProviderEntry] = {}
_ALIASES: dict[str, str] = {}


def register_provider(
    name: str,
    factory: ProviderFactory,
    *,
    description: str = "",
    capabilities: ProviderCapabilities | None = None,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> ProviderEntry:
    """Add a provider to the registry.

    Args:
        name: The name used in configs and on the command line.
        factory: Callable returning a :class:`PriceProvider`. Called with
            ``api_key=`` and any extra options from
            :func:`get_provider`, so it must accept keyword arguments.
        description: One line shown in the picker. Defaults to the class's own.
        capabilities: Declared capabilities. Defaults to a probe instance's.
        aliases: Alternative names that resolve here.
        replace: Allow overwriting an existing registration. Off by default so
            a plugin cannot silently shadow a built-in.

    Returns:
        The stored entry.

    Raises:
        ValueError: If the name is taken and ``replace`` is False.
    """
    key = str(name).strip().lower()
    if not key:
        raise ValueError("A provider name cannot be empty.")
    if key in _REGISTRY and not replace:
        raise ValueError(
            f"Provider {key!r} is already registered. "
            "Pass replace=True to override it deliberately."
        )

    if capabilities is None or not description:
        probe = factory()
        capabilities = capabilities or probe.capabilities
        description = description or probe.description

    entry = ProviderEntry(
        name=key,
        factory=factory,
        description=description,
        capabilities=capabilities,
        aliases=tuple(a.strip().lower() for a in aliases),
    )
    _REGISTRY[key] = entry
    for alias in entry.aliases:
        _ALIASES[alias] = key
    return entry


def resolve_name(name: str) -> str:
    """Map a name or alias onto a registered provider name.

    Args:
        name: A provider name or one of its aliases. Case-insensitive.

    Returns:
        The canonical registered name.

    Raises:
        ProviderNotFoundError: With the closest match, when there is one.
    """
    key = str(name).strip().lower()
    if key in _REGISTRY:
        return key
    if key in _ALIASES:
        return _ALIASES[key]

    known = sorted({*_REGISTRY, *_ALIASES})
    suggestion = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
    raise ProviderNotFoundError(
        f"Unknown data provider {name!r}. Available: {', '.join(sorted(_REGISTRY))}.{hint}"
    )


def get_provider(
    name: str, *, api_key: str | None = None, **options: Any
) -> PriceProvider:
    """Build a provider instance by name, with its key resolved.

    Args:
        name: Registered name or alias.
        api_key: Overrides the environment. Useful in tests and notebooks.
        **options: Passed to the factory — ``path=`` for the file provider,
            ``seed=`` for the sample provider, and so on.

    Returns:
        A ready provider. Whether its key is valid is not checked here; that
        is :meth:`PriceProvider.preflight`'s job, which the service calls.
    """
    key = resolve_name(name)
    entry = _REGISTRY[key]
    resolved = resolve_api_key(key, api_key)
    return entry.factory(api_key=resolved, **options)


def available_providers() -> tuple[str, ...]:
    """Registered provider names, sorted."""
    return tuple(sorted(_REGISTRY))


def provider_entry(name: str) -> ProviderEntry:
    """The registry entry for a name, without instantiating the provider.

    Args:
        name: A provider name or alias.

    Returns:
        Its :class:`ProviderEntry`, carrying the description and capabilities.
        Useful for ``optengine providers``, which reports what each source can
        do without constructing any of them.

    Raises:
        ProviderNotFoundError: If the name resolves to nothing.
    """
    return _REGISTRY[resolve_name(name)]


def describe_providers() -> tuple[Mapping[str, Any], ...]:
    """A renderable summary of every provider, including key readiness.

    Builds no provider and reads no key value — only whether one is set. This
    is what the CLI's ``providers`` command and the app's source picker draw.
    """
    rows = []
    for name in available_providers():
        entry = _REGISTRY[name]
        caps = entry.capabilities
        status = key_status(
            name, required=caps.requires_key, signup_url=caps.signup_url
        )
        rows.append(
            {
                "provider": name,
                "description": entry.description,
                # Canonical order, not alphabetical: a reader scanning
                # "open, high, low, close, volume" understands it instantly,
                # where "close, high, low, open, volume" reads as noise. Same
                # for bar sizes, which have an obvious ascending order.
                "fields": tuple(f for f in F.MARKET_FIELDS if f in caps.fields),
                "intervals": tuple(
                    i for i in INTERVALS if i in caps.intervals
                ),
                "serves_volume": caps.serves_volume,
                "requires_key": caps.requires_key,
                "key_present": status.present,
                "ready": status.ready,
                "key_env_var": status.env_var,
                "key_label": status.label,
                "signup_url": caps.signup_url,
                "batch": caps.supports_batch,
                "offline": caps.is_offline,
                "notes": caps.notes,
            }
        )
    return tuple(rows)


def _register_builtins() -> None:
    """Register the providers that ship with the library.

    Imports are local so that a missing optional dependency — ``yfinance``,
    say — cannot break the registry for the providers that do not need it.
    """
    from optimization_engine.ingest.providers.file import LocalFile
    from optimization_engine.ingest.providers.fmp import FinancialModelingPrep
    from optimization_engine.ingest.providers.fred import Fred
    from optimization_engine.ingest.providers.sample import Sample
    from optimization_engine.ingest.providers.stooq import Stooq
    from optimization_engine.ingest.providers.tiingo import Tiingo
    from optimization_engine.ingest.providers.yahoo import Yahoo

    for cls, aliases in (
        (Sample, ("synthetic", "demo")),
        (Yahoo, ("yfinance", "yahoo_finance")),
        (Stooq, ()),
        (Fred, ("stlouisfed",)),
        (FinancialModelingPrep, ("financial_modeling_prep", "financialmodelingprep")),
        (Tiingo, ()),
        (LocalFile, ("local", "csv", "excel", "parquet")),
    ):
        register_provider(
            cls.name,
            cls,
            description=cls.description,
            aliases=aliases,
            replace=True,
        )


_register_builtins()


__all__ = [
    "ProviderEntry",
    "available_providers",
    "describe_providers",
    "get_provider",
    "provider_entry",
    "register_provider",
    "resolve_name",
]
