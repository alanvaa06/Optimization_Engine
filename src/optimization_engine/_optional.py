"""Optional dependencies, and the error you get when one is missing.

The core install is deliberately small — numpy, pandas, scipy, cvxpy and
pyyaml. Everything that only some callers need lives behind an extra:
plotting behind ``viz``, Excel I/O behind ``excel``, the estimators that
wrap scikit-learn and statsmodels behind ``stats``.

The cost of that split is paid here. A missing optional dependency has to
fail with the install command that fixes it, not with a bare
``ModuleNotFoundError`` naming an import the caller never wrote — nobody
reading ``No module named 'statsmodels'`` from a call to ``beta()`` knows
that ``pip install finport-optengine[stats]`` is the answer.

Two shapes, depending on how the dependency is used:

:func:`require` is for a single call site, and imports eagerly when reached.
:class:`LazyModule` is for a module that is referenced dozens of times
across a file — it stands in at module level and imports on first attribute
access, so the import stays deferred without threading a function call
through every use.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

#: Distribution name, used to build the install hint. Kept here rather than
#: read from importlib.metadata: the hint has to be right even when the
#: package is running from a source checkout that was never installed.
_DISTRIBUTION = "finport-optengine"


class MissingDependencyError(ImportError):
    """An optional dependency is needed for this code path and is absent.

    Subclasses :class:`ImportError` so that callers already catching import
    failures — and ``try: ... except ImportError`` around an optional
    feature — keep working unchanged.
    """


def _message(module: str, extra: str, purpose: str) -> str:
    return (
        f"{module} is required for {purpose}, and is not installed.\n"
        f"Install it with: pip install '{_DISTRIBUTION}[{extra}]'"
    )


def require(module: str, *, extra: str, purpose: str) -> ModuleType:
    """Import ``module``, or raise telling the caller which extra provides it.

    Args:
        module: Importable name, e.g. ``"statsmodels.api"``.
        extra: The extra that installs it, e.g. ``"stats"``.
        purpose: What breaks without it, phrased to complete the sentence
            "X is required for ...". Keep it about the caller's intent
            ("regression statistics"), not the mechanism ("importing sm").

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the module is not installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover — needs the dep absent
        raise MissingDependencyError(_message(module, extra, purpose)) from exc


class LazyModule:
    """A module stand-in that imports on first attribute access.

    Bound at module level in place of a real import, so that a file using
    ``go.Figure`` in sixty places keeps reading the way it did while the
    import itself only happens if one of those lines actually runs.

    The resolved module is cached on the instance, so the import cost and
    the ``sys.modules`` lookup are paid once.

    Note that this defers the *import*, not the *dependency*: any attribute
    access raises :class:`MissingDependencyError` when the module is
    absent. Annotations are the reason that is enough — under
    ``from __future__ import annotations`` a return type of ``go.Figure``
    is never evaluated, so a module can be imported, and every function in
    it introspected, without plotly present.
    """

    def __init__(self, module: str, *, extra: str, purpose: str) -> None:
        self._module = module
        self._extra = extra
        self._purpose = purpose
        self._resolved: ModuleType | None = None

    def __getattr__(self, name: str) -> Any:
        # Only reached for names that are not instance attributes, i.e. the
        # ones the caller means for the underlying module.
        resolved = self.__dict__.get("_resolved")
        if resolved is None:
            resolved = require(
                self.__dict__["_module"],
                extra=self.__dict__["_extra"],
                purpose=self.__dict__["_purpose"],
            )
            self.__dict__["_resolved"] = resolved
        return getattr(resolved, name)

    def __repr__(self) -> str:
        state = "loaded" if self.__dict__.get("_resolved") is not None else "deferred"
        return f"<LazyModule {self._module!r} ({state})>"
