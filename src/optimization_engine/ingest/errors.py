"""The error taxonomy every ingest provider speaks.

A data pull can fail in ways that call for very different responses, and a
single ``RuntimeError`` erases the difference. A missing API key is fixed by
the user in ten seconds; an unknown ticker means the universe is wrong; a 503
should simply be retried. The service layer routes on these types — it retries
:class:`ProviderTransientError`, drops the identifier on
:class:`IdentifierNotFoundError`, and aborts the whole run on
:class:`ProviderCredentialsError`, because every other identifier is going to
fail the same way.

All of them derive from :class:`IngestError`, so a caller that does not care
about the distinction can still catch one thing.
"""

from __future__ import annotations


class IngestError(RuntimeError):
    """Base class for every failure raised by the ingest layer."""


class ProviderNotFoundError(IngestError):
    """The requested provider name is not registered."""


class ProviderConfigurationError(IngestError):
    """The request asks a provider for something it cannot serve.

    Raised before any network call — for an interval the provider does not
    publish, a field it never returns, or an identifier its symbol grammar
    rejects.
    """


class ProviderCredentialsError(IngestError):
    """The provider needs an API key that is missing, malformed, or rejected.

    Fatal for the whole run: no other identifier is going to authenticate
    either.
    """


class ProviderTransientError(IngestError):
    """A failure worth retrying — timeout, connection reset, 5xx, throttling."""


class ProviderResponseError(IngestError):
    """The provider answered, but the payload could not be understood."""


class IdentifierNotFoundError(IngestError):
    """The provider has no data for this identifier.

    Scoped to one identifier, so the service records it and carries on with
    the rest of the universe rather than aborting the run.
    """


class PanelValidationError(IngestError):
    """A fetched panel failed a structural or economic sanity check."""


__all__ = [
    "IdentifierNotFoundError",
    "IngestError",
    "PanelValidationError",
    "ProviderConfigurationError",
    "ProviderCredentialsError",
    "ProviderNotFoundError",
    "ProviderResponseError",
    "ProviderTransientError",
]
