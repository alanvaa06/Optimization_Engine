"""The contract every price provider implements, and the plumbing they share.

A provider's job is narrow on purpose: take an
:class:`~optimization_engine.ingest.spec.IngestRequest` and one identifier,
and return a :class:`~optimization_engine.ingest.panel.PricePanel` whose
columns are named in the homogenized vocabulary. Everything else — retries,
concurrency, caching, currency conversion, the run log — belongs to the
service above, so a new provider is one class and a parse function, not a
pipeline.

Two pieces make that work.

:class:`ProviderCapabilities` is a provider's honest self-description: which
fields it can serve, which intervals, which instrument kinds, whether it needs
a key. The service reads it *before* fetching and fails a request that asks
for something impossible — asking Stooq for volume on an index, or FRED for
intraday bars — with a message naming the provider and the gap, instead of a
panel that is quietly missing a column.

:class:`PriceProvider` carries the shared HTTP behaviour: bounded retries with
exponential backoff on the failures that are worth retrying, an error taxonomy
mapped from status codes, and a hard rule that no exception message may ever
contain the request URL, because provider keys travel in query strings.

Implementations must be thread-safe. The service fetches identifiers
concurrently, and the only state a provider is expected to hold is its own
configuration.
"""

from __future__ import annotations

import abc
import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.errors import (
    IdentifierNotFoundError,
    ProviderConfigurationError,
    ProviderCredentialsError,
    ProviderResponseError,
    ProviderTransientError,
)
from optimization_engine.ingest.panel import PricePanel
from optimization_engine.ingest.spec import IngestRequest

_LOG = logging.getLogger(__name__)

_USER_AGENT = (
    "optimization-engine/0.3 "
    "(+https://github.com/alanvaa06/Optimization_Engine)"
)

#: Statuses worth trying again: the server is overloaded, throttling, or a
#: gateway blinked. Everything else in 4xx is the request's own fault and
#: retrying it just burns the rate limit.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider can actually do, declared rather than discovered.

    Attributes:
        fields: The homogenized fields it can serve.
        intervals: The bar sizes it publishes.
        kinds: The instrument kinds it covers.
        requires_key: Whether an API key is mandatory.
        supports_batch: Whether one request can carry several identifiers.
            When true the service calls :meth:`PriceProvider.fetch_batch` once
            instead of :meth:`PriceProvider.fetch_one` per identifier.
        max_batch_size: Cap on identifiers per batched request.
        accepts_any_symbol: Whether the provider serves whatever name it is
            given. True for the synthetic and file providers, whose universe
            is whatever the caller says it is; the symbol catalog is then
            bypassed rather than reporting a catalog instrument as
            unsupported.
        is_offline: Whether a fetch touches the network at all. A caller can
            run an offline provider without asking permission — nothing
            leaves the machine and nothing can be rate-limited — which is
            what lets the app land on working data instead of an empty page.
        signup_url: Where a user goes to get a key.
        rate_limit_per_minute: Documented request ceiling, or ``None``.
        notes: One line the UI shows under the provider's name.
    """

    fields: frozenset[str] = frozenset(F.PRICE_ONLY)
    intervals: frozenset[str] = frozenset({"1d"})
    kinds: frozenset[F.InstrumentKind] = frozenset({F.InstrumentKind.UNKNOWN})
    requires_key: bool = False
    supports_batch: bool = False
    max_batch_size: int = 1
    accepts_any_symbol: bool = False
    is_offline: bool = False
    signup_url: str | None = None
    rate_limit_per_minute: int | None = None
    notes: str = ""

    @property
    def serves_volume(self) -> bool:
        return F.VOLUME in self.fields

    def missing_fields(self, requested: tuple[str, ...]) -> tuple[str, ...]:
        """Requested fields this provider cannot serve, in request order."""
        return tuple(f for f in requested if f not in self.fields)


class PriceProvider(abc.ABC):
    """Base class for every price source.

    Subclasses implement :meth:`fetch_one` (and :meth:`fetch_batch` when the
    API supports multi-symbol requests) plus :attr:`capabilities`, and get the
    HTTP behaviour below for free.
    """

    #: Registered name. Used in configs, the CLI, and the run log.
    name: str = "base"
    #: One-line description shown in the provider picker.
    description: str = ""

    _MAX_ATTEMPTS = 4
    _BACKOFF_BASE_SECONDS = 0.4
    _TIMEOUT_SECONDS = 30.0

    def __init__(self, *, api_key: str | None = None, timeout: float | None = None) -> None:
        self._api_key = api_key
        self._timeout = float(timeout) if timeout is not None else self._TIMEOUT_SECONDS

    # -- contract ---------------------------------------------------------

    @property
    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """What this provider can serve. Read before every fetch."""

    @abc.abstractmethod
    def fetch_one(self, identifier: str, request: IngestRequest) -> PricePanel:
        """Return a single-identifier panel.

        Args:
            identifier: The provider-side symbol, already translated by the
                catalog and validated by the request.
            request: The run's window, interval and requested fields.

        Returns:
            A panel with exactly one column, named ``identifier``, carrying at
            least :data:`~optimization_engine.ingest.fields.CLOSE`.

        Raises:
            IdentifierNotFoundError: The provider has no such symbol.
            ProviderCredentialsError: The key is missing or rejected.
            ProviderTransientError: A retryable failure survived the retries.
            ProviderResponseError: The payload could not be parsed.
        """

    def fetch_batch(self, identifiers: tuple[str, ...], request: IngestRequest) -> PricePanel:
        """Return a panel for several identifiers in one call.

        The default fans out to :meth:`fetch_one` and merges, which is correct
        for every provider and optimal for none — providers whose API takes a
        symbol list override this and set
        :attr:`ProviderCapabilities.supports_batch`.
        """
        panel: PricePanel | None = None
        for identifier in identifiers:
            fetched = self.fetch_one(identifier, request)
            panel = fetched if panel is None else panel.merge(fetched)
        if panel is None:
            raise ProviderConfigurationError("fetch_batch requires at least one identifier.")
        return panel

    def validate_credentials(self) -> bool | None:
        """Whether the configured key is usable.

        Returns ``None`` for providers that need no key. The default answers
        from configuration alone; providers with a cheap validation endpoint
        override it to actually ask.
        """
        if not self.capabilities.requires_key:
            return None
        return bool(self._api_key)

    def preflight(self, request: IngestRequest) -> None:
        """Reject a request this provider cannot serve, before any network call.

        Raises:
            ProviderConfigurationError: The interval or a requested field is
                outside :attr:`capabilities`.
            ProviderCredentialsError: A key is required and none is configured.
        """
        caps = self.capabilities
        if request.interval not in caps.intervals:
            raise ProviderConfigurationError(
                f"{self.name} does not publish {request.interval!r} bars "
                f"(supports: {', '.join(sorted(caps.intervals))})."
            )
        missing = caps.missing_fields(request.fields)
        if missing:
            raise ProviderConfigurationError(
                f"{self.name} cannot serve {', '.join(missing)} "
                f"(serves: {', '.join(sorted(caps.fields))}). "
                "Drop the field, or choose a provider that publishes it."
            )
        if caps.requires_key and not self._api_key:
            from optimization_engine.ingest.credentials import env_var_for

            where = f" Get one at {caps.signup_url}." if caps.signup_url else ""
            raise ProviderCredentialsError(
                f"{self.name} requires an API key. Set {env_var_for(self.name)} "
                f"in the environment or in a .env file.{where}"
            )

    # -- shared HTTP ------------------------------------------------------

    def _get_text(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        secret_params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        endpoint: str = "endpoint",
    ) -> str:
        """GET a URL with bounded retries, returning the body as text.

        Args:
            url: Base URL, without a query string.
            params: Query parameters safe to name in an error message.
            secret_params: Query parameters that must never be logged or
                raised — API keys go here, and only here.
            headers: Extra request headers. Values are never logged.
            endpoint: Short label used in error messages instead of the URL.

        Raises:
            ProviderCredentialsError: On 401 or 403.
            IdentifierNotFoundError: On 404.
            ProviderTransientError: When retries are exhausted.
            ProviderResponseError: On a non-retryable failure.
        """
        query = {**(params or {}), **(secret_params or {})}
        full_url = f"{url}?{urllib.parse.urlencode(query)}" if query else url
        request_headers = {"User-Agent": _USER_AGENT, **(headers or {})}

        last_error: Exception | None = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                req = urllib.request.Request(full_url, headers=request_headers)
                with urllib.request.urlopen(req, timeout=self._timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    return response.read().decode(charset, errors="replace")
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                if status in (401, 403):
                    raise ProviderCredentialsError(
                        f"{self.name} rejected the API key on {endpoint} "
                        f"(HTTP {status}). Check the key and its entitlements."
                    ) from None
                if status == 404:
                    raise IdentifierNotFoundError(
                        f"{self.name} has no data at {endpoint} (HTTP 404)."
                    ) from None
                if status == 402:
                    raise ProviderCredentialsError(
                        f"{self.name} says this data needs a paid plan "
                        f"(HTTP 402 on {endpoint})."
                    ) from None
                if status not in _RETRYABLE_STATUS:
                    # ``exc`` stringifies with the URL, which carries the key.
                    raise ProviderResponseError(
                        f"{self.name} returned HTTP {status} on {endpoint}."
                    ) from None
                last_error = ProviderTransientError(
                    f"{self.name} returned HTTP {status} on {endpoint}."
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # Only the exception *type* is safe to report: URLError's
                # message can embed the request URL.
                last_error = ProviderTransientError(
                    f"{self.name} could not reach {endpoint} "
                    f"({type(exc).__name__})."
                )

            if attempt < self._MAX_ATTEMPTS:
                self._sleep_before_retry(attempt, endpoint)

        raise last_error or ProviderTransientError(
            f"{self.name} failed on {endpoint} after {self._MAX_ATTEMPTS} attempts."
        )

    def _get_json(self, url: str, **kwargs: object) -> object:
        """GET a URL and parse the body as JSON."""
        endpoint = str(kwargs.get("endpoint", "endpoint"))
        body = self._get_text(url, **kwargs)  # type: ignore[arg-type]
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"{self.name} returned a body that is not JSON on {endpoint}: {exc}."
            ) from None

    def _sleep_before_retry(self, attempt: int, endpoint: str) -> None:
        """Exponential backoff with jitter.

        The jitter matters when the service fetches eight identifiers at once
        and they all hit the same 429: without it they retry in lockstep and
        get throttled again together.
        """
        delay = self._BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        delay += random.uniform(0.0, self._BACKOFF_BASE_SECONDS)
        _LOG.warning(
            "%s: retrying %s (attempt %d of %d) in %.2fs",
            self.name, endpoint, attempt + 1, self._MAX_ATTEMPTS, delay,
        )
        time.sleep(delay)


@dataclass(frozen=True)
class ProviderInfo:
    """A provider's registry entry, renderable without instantiating it."""

    name: str
    description: str
    capabilities: ProviderCapabilities
    aliases: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "PriceProvider",
    "ProviderCapabilities",
    "ProviderInfo",
]
