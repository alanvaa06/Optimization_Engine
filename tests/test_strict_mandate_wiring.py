"""``config.strict_mandate`` has to reach the optimizer, and then behave.

The gate itself lives on :class:`~optimization_engine.optimizers.base.BaseOptimizer`
and is exercised in ``tests/test_audit.py``. What is pinned here is the wiring
above it — the config field reaching the instance ``run_engine`` builds — and
the two consequences that follow from where the raise happens, both of which
are easy to assume the wrong way round:

* it is **not** a ``SolverFailure``, so a caller guarding a solve with that one
  ``except`` does not catch it; and
* inside a walk-forward it does not propagate, because every window's solve is
  wrapped in a bare ``except Exception`` that turns a raise into a ``failed:``
  window carrying the previous book forward — unless *every* window refuses, in
  which case the walk-forward stops with a plain ``ValueError``.

The second is coherent and surprising in equal measure, which is exactly why it
gets a test rather than a comment.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.config import EngineConfig, OptimizerSpec  # noqa: E402
from optimization_engine.data.loader import (  # noqa: E402
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.engine import run_engine  # noqa: E402
from optimization_engine.optimizers._cvxpy_helpers import SolverFailure  # noqa: E402
from optimization_engine.optimizers.audit import MandateViolationError  # noqa: E402


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 3, seed=11))


def _breaching_config(returns: pd.DataFrame, *, strict: bool) -> EngineConfig:
    """HRP under a mandate its projection provably cannot satisfy.

    A turnover budget and a tracking-error cap are the two limits *every*
    projection drops — a projection is not a trade, and it is not a risk
    statement either — so a soft-bounds method handed both returns a book that
    breaches them. That is the case ``strict_mandate`` exists for.
    """
    assets = list(returns.columns)
    equal = {a: 1.0 / len(assets) for a in assets}
    return EngineConfig(
        expected_returns=dict.fromkeys(assets, 0.05),
        optimizer=OptimizerSpec(name="hrp"),
        benchmark_weights=equal,
        max_tracking_error=0.001,
        previous_weights={a: (1.0 if a == assets[0] else 0.0) for a in assets},
        turnover_limit=0.05,
        strict_mandate=strict,
    )


def test_the_config_field_reaches_the_optimizer_run_engine_builds(returns):
    """Off by default, and on when the config says so — read off the instance.

    The assertion is on the *optimizer*, not on the outcome, because that is
    the wiring being pinned: the gate is deliberately set on the instance so
    that every entry into a solve honours it, not only the one this function
    makes.
    """
    from optimization_engine.data.covariance import covariance_from_config
    from optimization_engine.optimizers.factory import optimizer_factory

    config = _breaching_config(returns, strict=False)
    cov = covariance_from_config(returns, config)
    assert optimizer_factory(config, cov, returns=returns).strict_mandate is False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = run_engine(returns, config, check_feasibility=False)
    # Off: the breach is reported and the book still comes back.
    assert run.result.audit is not None
    assert not run.result.audit.is_clean
    assert run.warnings


def test_strict_mandate_refuses_the_same_book_run_engine_would_have_returned(returns):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(MandateViolationError) as excinfo:
            run_engine(
                returns, _breaching_config(returns, strict=True), check_feasibility=False
            )
    kinds = {v.kind for v in excinfo.value.report.violations}
    assert {"tracking_error", "turnover"} <= kinds


def test_it_is_not_a_solver_failure_and_run_engine_does_not_convert_it(returns):
    """``except SolverFailure`` around the solve must not swallow this.

    ``run_engine`` re-raises a ``SolverFailure`` with the feasibility report
    attached when the pre-flight had found the mandate impossible. A mandate
    violation is the opposite case — the solve *succeeded* — and dressing it up
    as a solver failure would send the caller to check constraints that are
    perfectly satisfiable.
    """
    assert not issubclass(MandateViolationError, SolverFailure)
    assert not issubclass(SolverFailure, MandateViolationError)

    config = _breaching_config(returns, strict=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(MandateViolationError):
            try:
                run_engine(returns, config, check_feasibility=True)
            except SolverFailure as exc:  # pragma: no cover - the bug this pins
                pytest.fail(f"a mandate violation arrived as a SolverFailure: {exc}")


def test_inside_a_walk_forward_it_becomes_a_failed_window_not_a_raise(returns):
    """Documented in ``run_engine``'s docstring, and true.

    The walk-forward catches every exception a window's solve raises and
    records the window as ``failed:``. So a refusal under ``strict_mandate``
    does not stop a backtest — it converts that window into a carried-forward
    one, and with no prior book the desk holds cash. The solver here raises the
    real exception on the first two decisions and complies afterwards, which is
    the mixed case; the all-refused case is the next test.
    """
    from optimization_engine.backtest.walkforward import walk_forward_run
    from optimization_engine.optimizers.audit import AuditReport
    from optimization_engine.optimizers.diagnostics import ConstraintViolation

    assets = list(returns.columns)
    equal = pd.Series(1.0 / len(assets), index=assets)
    refused: list[int] = []

    def solve(window: pd.DataFrame) -> pd.Series:
        if len(refused) < 2:
            refused.append(len(window))
            raise MandateViolationError(
                AuditReport(
                    violations=(
                        ConstraintViolation(
                            kind="turnover",
                            label="Turnover",
                            limit=0.05,
                            actual=1.99,
                        ),
                    )
                )
            )
        return equal

    walk = walk_forward_run(
        returns, solve, lookback=252, rebalance_every=126
    )

    assert walk.n_failures == 2, walk.failures
    assert all("breach the mandate" in f for f in walk.failures), walk.failures
    assert len(walk.weights_history) > walk.n_failures, "the run still produced books"
    # The two refused windows are the first two, and they hold cash: there was
    # no compliant book before them to carry forward.
    cash = walk.weights_history.iloc[:2].abs().to_numpy().sum()
    assert float(cash) == pytest.approx(0.0)
    assert walk.weights_history.iloc[-1].sum() == pytest.approx(1.0)
    assert walk.run.meta.notes["n_failed_solves"] == 2
    assert walk.run.meta.notes["periods_in_cash_after_failed_solve"] > 0


def test_a_walk_forward_whose_every_window_refuses_stops_rather_than_reporting_cash(
    returns,
):
    """The one case that is not silent — and it is a ``ValueError``, not the audit.

    ``strict_mandate`` on a method that can never satisfy the mandate refuses
    every window, and a track record that is cash from end to end is not a
    result. The walk-forward says so, quoting the first window's message; the
    exception type is the generic one, so a caller cannot reach
    ``exc.report`` here the way they can off a single solve.
    """
    config = _breaching_config(returns, strict=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anchor = run_engine(returns, config, check_feasibility=False)
        anchor.config.strict_mandate = True
        with pytest.raises(ValueError, match="Every walk-forward solve failed") as exc:
            anchor.walk_forward_run(lookback=252, rebalance_every=126)
    assert "breach the mandate" in str(exc.value)
    assert not isinstance(exc.value, MandateViolationError)
