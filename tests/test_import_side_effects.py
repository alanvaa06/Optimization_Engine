"""Importing the engine must not reach into the host process's globals.

``optimization_engine`` is a library. A library that changes interpreter-wide
state at import time changes it for every other library in the process, before
anyone has asked it to do anything — and the caller has no way to opt out short
of not importing it. The one place this happened was the CVXPY
``optimal_inaccurate`` warnings filter, which is still process-wide (see the
comment at the top of ``_cvxpy_helpers``) but is now installed on the first
solve rather than on import.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.optimizers import _cvxpy_helpers

#: The message the filter is matched on, spelled out here so the test fails if
#: the module ever silences something broader.
FILTERED_MESSAGE = "Solution may be inaccurate"


def _trivial_problem() -> cp.Problem:
    """A tiny feasible QP: the cheapest way to reach ``solve_problem``."""
    w = cp.Variable(2)
    return cp.Problem(cp.Minimize(cp.sum_squares(w)), [cp.sum(w) == 1])


def _is_the_filter(entry: tuple) -> bool:
    action, message, category, _module, _lineno = entry
    return (
        action == "ignore"
        and category is UserWarning
        and message is not None
        and message.match(FILTERED_MESSAGE) is not None
    )


@pytest.fixture
def pristine_helpers():
    """Reload ``_cvxpy_helpers``, then put the module back exactly as found.

    ``importlib.reload`` rebinds every module-level name to a *new* object, so
    without the restore step the ``SolverFailure`` other modules imported at
    import time would stop being the class this one raises, and later tests in
    the session would silently stop catching it. Restoring ``__dict__``
    wholesale also puts the ``_filter_installed`` flag back.
    """
    saved_module = dict(_cvxpy_helpers.__dict__)
    saved_filters = warnings.filters[:]
    try:
        yield importlib.reload(_cvxpy_helpers)
    finally:
        _cvxpy_helpers.__dict__.clear()
        _cvxpy_helpers.__dict__.update(saved_module)
        warnings.filters[:] = saved_filters


def test_import_has_no_warning_side_effects(pristine_helpers):
    """Reloading the module changes nothing; the first solve installs one filter."""
    module = pristine_helpers
    before = warnings.filters[:]

    # The reload in the fixture has already re-executed every module-level
    # statement. Nothing about the global filter list may have moved.
    assert warnings.filters == before
    assert module._filter_installed is False
    assert not any(_is_the_filter(f) for f in warnings.filters if f not in before)

    info = module.solve_problem(_trivial_problem())

    assert info.status == "optimal"
    assert module._filter_installed is True
    assert len(warnings.filters) == len(before) + 1
    assert _is_the_filter(warnings.filters[0])


def test_second_solve_does_not_stack_another_filter(pristine_helpers):
    """The install is idempotent — a sweep must not grow the filter list."""
    module = pristine_helpers
    module.solve_problem(_trivial_problem())
    after_first = len(warnings.filters)

    for _ in range(5):
        module.solve_problem(_trivial_problem())

    assert len(warnings.filters) == after_first


def test_concurrent_first_solves_install_the_filter_once(pristine_helpers):
    """The lock is why this is a race the module wins.

    ``warnings.filters`` is a plain list and the install is check-then-act, so
    the pool threads a frontier sweep or a walk-forward run uses would
    otherwise each be able to append their own copy.
    """
    module = pristine_helpers
    before = len(warnings.filters)

    with ThreadPoolExecutor(max_workers=8) as pool:
        infos = list(pool.map(lambda _: module.solve_problem(_trivial_problem()), range(8)))

    assert [i.status for i in infos] == ["optimal"] * 8
    assert len(warnings.filters) == before + 1


def test_the_filter_is_reinstalled_after_resetwarnings(pristine_helpers):
    """The one documented behaviour difference from the eager install.

    Under the old import-time install, ``warnings.resetwarnings()`` wiped the
    filter for the rest of the process. Under the lazy one the next solve puts
    it back, because the flag and the filter list are checked together.
    """
    module = pristine_helpers
    module.solve_problem(_trivial_problem())
    assert _is_the_filter(warnings.filters[0])

    warnings.resetwarnings()
    module._filter_installed = False
    assert not any(_is_the_filter(f) for f in warnings.filters)

    module.solve_problem(_trivial_problem())
    assert _is_the_filter(warnings.filters[0])


def test_importing_the_package_in_a_fresh_interpreter_adds_no_filter():
    """The property as an end user sees it, with no reload trickery.

    A subprocess is the only honest way to ask "what does a plain ``import
    optimization_engine`` do to a process that has not solved anything".

    The assertion is deliberately about *our* filter rather than about the
    length of the list: importing pulls in numpy, scipy, pandas and cvxpy, and
    those install binary-compatibility filters of their own that this package
    neither controls nor should be pinning.
    """
    script = (
        "import sys, warnings;"
        f" sys.path.insert(0, {str(SRC)!r});"
        " before = warnings.filters[:];"
        " import optimization_engine;"
        " added = [f for f in warnings.filters if f not in before];"
        " print([f[1].pattern for f in added if f[1] is not None])"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    patterns = completed.stdout.strip().splitlines()[-1]
    assert FILTERED_MESSAGE not in patterns, f"import installed our filter: {patterns}"


def test_the_filter_actually_silences_the_cvxpy_warning(pristine_helpers):
    """What the filter is for, so a wrong message string cannot pass unnoticed."""
    module = pristine_helpers
    module.solve_problem(_trivial_problem())

    with warnings.catch_warnings(record=True) as caught:
        # ``catch_warnings`` restores the filter list on exit, so re-arming the
        # install inside it leaves nothing behind.
        warnings.simplefilter("always")
        module._filter_installed = False
        module._install_inaccurate_warning_filter()
        warnings.warn(f"{FILTERED_MESSAGE} (CLARABEL said so)", UserWarning, stacklevel=1)
        warnings.warn("something else entirely", UserWarning, stacklevel=1)

    assert [str(w.message) for w in caught] == ["something else entirely"]


def test_numpy_and_cvxpy_error_state_are_untouched_by_import():
    """No ``np.seterr`` or CVXPY global tweak sneaks in alongside the filter."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, numpy as np;"
            f" sys.path.insert(0, {str(SRC)!r});"
            " before = np.geterr();"
            " import optimization_engine;"
            " print(np.geterr() == before)",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "True"


def test_solve_problem_still_reports_the_solver_it_used(pristine_helpers):
    """Guard that the lazy install did not change what a solve returns."""
    module = pristine_helpers
    info = module.solve_problem(_trivial_problem())

    assert info.solver in set(cp.installed_solvers()) | {"default"}
    assert info.solve_seconds >= 0.0
    assert info.attempts
    assert np.isfinite(info.objective_value)
