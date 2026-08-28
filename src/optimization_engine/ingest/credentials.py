"""Where API keys come from, and how they are kept out of everything else.

One convention, so adding a provider does not add a configuration question:
``OPTENGINE_API_KEY_<PROVIDER>``, upper-cased, e.g. ``OPTENGINE_API_KEY_FMP``.
Keys are read from the process environment and, optionally, from a ``.env``
file next to the project, which is the ergonomics people actually want and the
thing they otherwise reimplement badly.

Two rules the rest of the codebase depends on:

* A key is never returned by anything that renders. :func:`key_status` reports
  *whether* a key is present and what it looks like at a glance
  (``sk-…f42a``), never the key.
* A key is never interpolated into an exception message. Providers put keys in
  query strings, and ``str(error)`` on most HTTP libraries includes the URL —
  so adapters are required to raise on the status code and never on the
  library's own message.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_ENV_PREFIX = "OPTENGINE_API_KEY_"

#: A key that is obviously not a key. Vendors' placeholder strings show up in
#: copied ``.env`` files constantly, and failing at fetch time with a 401 is a
#: much worse experience than saying so up front.
_PLACEHOLDERS = frozenset(
    {
        "", "none", "null", "todo", "changeme", "your_api_key", "your-api-key",
        "xxx", "xxxx", "<your key>", "insert_key_here", "api_key",
    }
)

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def env_var_for(provider: str) -> str:
    """The environment variable a provider's key is read from."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(provider)).strip("_").upper()
    return f"{_ENV_PREFIX}{slug}"


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> int:
    """Load ``KEY=value`` pairs from a ``.env`` file into the environment.

    Deliberately minimal: no interpolation, no multi-line values, no shell
    semantics. Anything more expressive is a footgun in a file that holds
    secrets.

    Args:
        path: The file to read. Defaults to ``.env`` in the current directory.
        override: Replace variables already set in the environment. Off by
            default, so a real environment variable always wins over a file
            someone forgot to delete.

    Returns:
        How many variables were set.
    """
    target = Path(path) if path is not None else Path.cwd() / ".env"
    if not target.is_file():
        return 0

    loaded = 0
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not override and name in os.environ:
            continue
        os.environ[name] = value
        loaded += 1
    return loaded


def resolve_api_key(provider: str, explicit: str | None = None) -> str | None:
    """Find a provider's API key.

    Args:
        provider: Registered provider name.
        explicit: A key passed directly, which always wins — that is how a
            caller injects a key in a test or a notebook without touching the
            environment.

    Returns:
        The key, or ``None`` when none is configured or the configured value
        is a recognizable placeholder.
    """
    candidate = explicit if explicit is not None else os.environ.get(env_var_for(provider))
    if candidate is None:
        return None
    candidate = candidate.strip()
    if candidate.lower() in _PLACEHOLDERS:
        return None
    return candidate or None


def mask(secret: str | None) -> str:
    """Render a secret as something safe to print.

    Short secrets are fully masked rather than partially revealed — showing
    four of eight characters is not a summary, it is a leak.
    """
    if not secret:
        return "—"
    if len(secret) <= 8:
        return "•" * len(secret)
    return f"{secret[:3]}…{secret[-4:]}"


@dataclass(frozen=True)
class KeyStatus:
    """What the UI and CLI are allowed to say about a provider's key.

    Attributes:
        provider: Registered provider name.
        env_var: The variable its key is read from.
        required: Whether the provider needs one at all.
        present: Whether a usable key was found.
        hint: The masked key, or a note about where to get one.
    """

    provider: str
    env_var: str
    required: bool
    present: bool
    hint: str

    @property
    def ready(self) -> bool:
        """Whether the provider can be used right now."""
        return self.present or not self.required

    @property
    def label(self) -> str:
        if not self.required:
            return "No key needed"
        return f"Key set ({self.hint})" if self.present else f"Key missing — set {self.env_var}"


def key_status(provider: str, *, required: bool, signup_url: str | None = None) -> KeyStatus:
    """Report a provider's key readiness without revealing the key."""
    key = resolve_api_key(provider)
    return KeyStatus(
        provider=provider,
        env_var=env_var_for(provider),
        required=required,
        present=key is not None,
        hint=mask(key) if key else (signup_url or "no key configured"),
    )


__all__ = [
    "KeyStatus",
    "env_var_for",
    "key_status",
    "load_dotenv",
    "mask",
    "resolve_api_key",
]
