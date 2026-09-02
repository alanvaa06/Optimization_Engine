"""The mypy allowlist is allowed to shrink. This is what stops it growing.

``[tool.mypy]``'s ``ignore_errors`` override names the modules whose type
errors are deferred rather than fixed, and mypy is perfectly happy for that
list to get longer: appending a module silences a new failure exactly as well
as fixing one does, and costs nothing at review time. So the cost lives here
instead. The list is held to a ceiling, which closes the cheap way out of a red
``typecheck`` job and leaves only the expensive one — making the module clean.

The second failure mode is staleness. An entry naming a module that no longer
exists suppresses nothing at all, yet still counts against the ceiling and
still reads like outstanding work; a rename would leave one behind in silence.
So every entry has to name a file that is really there.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - only the 3.9 and 3.10 CI cells take this branch
    # Not an optional dependency: pytest itself requires tomli below 3.11, so
    # if this import fails there is no pytest to have run it.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
SRC = ROOT / "src"

#: How many modules were on the allowlist the day it was written — 45, holding
#: 171 errors between them under mypy 1.20 with pandas-stubs installed. This is
#: a ceiling, not a target: lower it by one every time a module is fixed and
#: comes off the list. Never raise it. Raising it is the single move this
#: module exists to prevent.
ALLOWLIST_CEILING = 45


def _allowlisted_modules() -> list[str]:
    """Every module named by an ``ignore_errors`` override, in file order.

    Selected on ``ignore_errors`` rather than by position, because
    ``[tool.mypy]`` carries other overrides (cvxpy's ``follow_imports``) that
    defer nothing and must not count against the ceiling.
    """
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return [
        module
        for override in config["tool"]["mypy"].get("overrides", [])
        if override.get("ignore_errors")
        for module in override["module"]
    ]


def test_the_allowlist_has_not_grown():
    modules = _allowlisted_modules()
    assert len(modules) <= ALLOWLIST_CEILING, (
        f"{len(modules)} modules have their type errors deferred, against a ceiling of "
        f"{ALLOWLIST_CEILING}. A module that does not type-check gets fixed, not listed: "
        f"the ceiling moves down as entries come off, and never up to admit one."
    )


def test_every_allowlisted_module_exists():
    missing = []
    for module in _allowlisted_modules():
        # A wildcard would defeat the count outright — ``optimization_engine.*``
        # is one entry and silences the whole package — so it is refused here
        # rather than left to fail the file check with a confusing message.
        assert "*" not in module, (
            f"{module!r} is a wildcard. One entry that covers a subtree keeps the count "
            f"under the ceiling while suppressing an unbounded number of modules; name "
            f"each module you are deferring."
        )
        parts = module.split(".")
        stem = SRC.joinpath(*parts)
        if not stem.with_suffix(".py").is_file() and not (stem / "__init__.py").is_file():
            missing.append(module)
    assert not missing, (
        f"allowlisted modules that no longer exist: {missing}. An entry for a module that "
        f"was renamed or deleted suppresses nothing and hides how much is really left."
    )


def test_the_allowlist_is_sorted_and_free_of_duplicates():
    # The list is only reviewable as a burn-down if a diff against it is
    # readable, which means one module per line in a fixed order. Sorted also
    # makes a duplicate — two lines, one module, two counts against the
    # ceiling — visible at a glance.
    modules = _allowlisted_modules()
    assert modules == sorted(modules), "keep the allowlist in alphabetical order"
    assert len(modules) == len(set(modules)), "the allowlist names a module twice"
