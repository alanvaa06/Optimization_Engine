"""The post-solve mandate audit, and the gate that turns a report into a refusal.

The checking these tests exercise is not new — ``check_constraints`` has always
run on the way out of every solve. What is new is that the answer has a name, a
structure, and a switch: ``result.audit`` instead of a list of sentences, and
``strict_mandate`` for the caller who would rather have no book than a
non-compliant one.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.constraints import ConstraintLayer  # noqa: E402
from optimization_engine.data.covariance import covariance_matrix  # noqa: E402
from optimization_engine.data.loader import (  # noqa: E402
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.optimizers.audit import (  # noqa: E402
    AuditReport,
    MandateViolationError,
    audit_weights,
)
from optimization_engine.optimizers.base import PortfolioConstraints  # noqa: E402
from optimization_engine.optimizers.diagnostics import (  # noqa: E402
    DEFAULT_TOLERANCE,
    ConstraintViolation,
)
from optimization_engine.optimizers.hrp import HRPOptimizer  # noqa: E402
from optimization_engine.optimizers.mean_variance import (  # noqa: E402
    MeanVarianceOptimizer,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    prices = sample_dataset(n_periods=252 * 4, seed=7)
    return prices_to_returns(prices)


@pytest.fixture(scope="module")
def cov(returns: pd.DataFrame) -> pd.DataFrame:
    return covariance_matrix(returns, method="ledoit_wolf")


@pytest.fixture(scope="module")
def mu(returns: pd.DataFrame) -> pd.Series:
    return (1 + returns).prod() ** (252 / len(returns)) - 1


# ---------------------------------------------------------------------------
# audit_weights: the function, on weights from nowhere in particular
# ---------------------------------------------------------------------------


def test_audit_weights_reports_a_breach_as_an_object_not_a_sentence():
    weights = pd.Series({"a": 0.7, "b": 0.3})
    constraints = PortfolioConstraints(bounds={"a": (0.0, 0.5), "b": (0.0, 1.0)})

    report = audit_weights(weights, ["a", "b"], constraints)

    assert not report.is_clean
    assert [v.kind for v in report.violations] == ["bound"]
    breach = report.violations[0]
    assert breach.limit == pytest.approx(0.5)
    assert breach.actual == pytest.approx(0.7)
    assert breach.magnitude == pytest.approx(0.2)
    assert isinstance(breach, ConstraintViolation)


def test_audit_weights_checks_the_mandates_universe_not_the_vectors():
    """An asset the weights never mention still owes its floor.

    ``check_constraints`` iterates the series it is handed, so a name dropped
    from the book — by a solver, by a spreadsheet, by a reindex nobody noticed —
    escapes its *lower* bound entirely. Passing the universe is what closes
    that: the missing name is audited at zero.
    """
    weights = pd.Series({"a": 0.6, "b": 0.4})
    constraints = PortfolioConstraints(
        bounds={"a": (0.0, 1.0), "b": (0.0, 1.0), "c": (0.05, 1.0)}
    )

    without_universe = audit_weights(weights, None, constraints)
    with_universe = audit_weights(weights, ["a", "b", "c"], constraints)

    assert without_universe.is_clean
    assert not with_universe.is_clean
    assert with_universe.violations[0].label == "c lower bound"
    assert with_universe.violations[0].actual == pytest.approx(0.0)


def test_audit_weights_keeps_the_packages_own_tolerance():
    """1e-6, not the design note's 1e-5, and the caller can still override.

    Two tolerances would let a book be compliant by one measure and not the
    other. ``DEFAULT_TOLERANCE`` is also what ``_clean_weights`` calls dust, so
    the audit and the weight cleaning agree on what counts as zero.
    """
    assert DEFAULT_TOLERANCE == 1e-6

    weights = pd.Series({"a": 0.5 + 1e-4, "b": 0.5 - 1e-4})
    constraints = PortfolioConstraints(bounds={"a": (0.0, 0.5), "b": (0.0, 1.0)})

    assert audit_weights(weights, ["a", "b"], constraints).tolerance == 1e-6
    assert not audit_weights(weights, ["a", "b"], constraints).is_clean
    assert audit_weights(weights, ["a", "b"], constraints, tolerance=1e-3).is_clean


def test_audit_weights_without_a_covariance_cannot_see_a_tracking_error_budget():
    """A clean report is "nothing I could check failed", not "nothing failed"."""
    weights = pd.Series({"a": 1.0, "b": 0.0})
    cov_matrix = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]], index=["a", "b"], columns=["a", "b"]
    )
    constraints = PortfolioConstraints(
        benchmark_weights={"a": 0.5, "b": 0.5}, max_tracking_error=0.01
    )

    assert audit_weights(weights, ["a", "b"], constraints).is_clean
    seen = audit_weights(weights, ["a", "b"], constraints, cov_matrix)
    assert [v.kind for v in seen.violations] == ["tracking_error"]


def test_audit_weights_defaults_to_the_default_mandate_not_to_no_mandate():
    """``constraints=None`` is a long-only, fully-invested book — and it checks."""
    assert not audit_weights(pd.Series({"a": 0.4, "b": 0.4})).is_clean
    assert audit_weights(pd.Series({"a": 0.5, "b": 0.5})).is_clean


# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------


def test_a_clean_report_describes_the_tolerance_it_was_clean_at():
    report = AuditReport()
    assert report.is_clean
    assert report.worst is None
    assert "1e-06" in report.describe()


def test_describe_lists_every_breach_and_worst_picks_the_largest():
    small = ConstraintViolation("bound", "a upper bound", 0.5, 0.52)
    large = ConstraintViolation("group", "Equity bucket", 0.30, 0.62)
    report = AuditReport(violations=(small, large))

    assert report.worst is large
    text = report.describe()
    assert "2 constraint(s) breached" in text
    assert "a upper bound" in text
    assert "Equity bucket" in text


# ---------------------------------------------------------------------------
# The two acceptance tests: what projects, and what does not
# ---------------------------------------------------------------------------


def test_audit_catches_projected_breach(returns: pd.DataFrame, cov: pd.DataFrame):
    """A soft-bounds method returns a book its mandate does not permit.

    HRP allocates down its own hierarchy and then projects onto the constraint
    set. The projection is a weights problem: it can move a book inside its box
    and inside every bucket budget, and it does so here. What it cannot
    represent is a limit that is not a function of the weights alone — a
    tracking-error budget is a *risk* statement, and ``_bounds.py`` drops the
    turnover budget outright because a projection is not a trade. Those come
    back through the solve untouched, and the audit is what says so.

    (The plan expected the bucket cap itself to be the unsatisfiable one. It is
    not: ``project_to_constraints`` puts every layer of the policy into the QP,
    and the assertions below pin that down so the day it stops being true is a
    failure here rather than a surprise in production.)
    """
    assets = list(returns.columns)
    equities = assets[:3]
    benchmark = {a: 1.0 / len(assets) for a in assets}
    previous = {a: (1.0 if a == assets[0] else 0.0) for a in assets}
    constraints = PortfolioConstraints(
        bounds={a: (0.0, 0.30) for a in assets},
        groups={a: ("Equity" if a in equities else "Other") for a in assets},
        group_bounds={"Equity": (0.0, 0.20)},
        benchmark_weights=benchmark,
        max_tracking_error=0.005,
        previous_weights=previous,
        turnover_limit=0.10,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = HRPOptimizer(cov_matrix=cov, constraints=constraints).optimize()

    assert result.extras["bounds_mode"] == "soft_iterated"
    report = result.audit
    assert report is not None
    assert not report.is_clean, "the projection is not supposed to fix these"

    kinds = {v.kind for v in report.violations}
    assert {"tracking_error", "turnover"} <= kinds

    # What the projection *did* honour, and must keep honouring.
    assert "bound" not in kinds and "group" not in kinds
    assert result.weights.max() <= 0.30 + 1e-6
    assert sum(result.weights[a] for a in equities) <= 0.20 + 1e-6

    # The report is the same finding the old string list carried, structured.
    assert sorted(result.violations) == sorted(
        v.describe() for v in report.violations
    )

    # And under `strict_mandate` the same solve refuses instead of reporting.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(MandateViolationError) as excinfo:
            HRPOptimizer(
                cov_matrix=cov, constraints=constraints, strict_mandate=True
            ).optimize()

    raised = excinfo.value
    assert {v.kind for v in raised.report.violations} == kinds
    assert "Tracking error" in str(raised)


def test_audit_passes_hard_solve(
    returns: pd.DataFrame, cov: pd.DataFrame, mu: pd.Series
):
    """A method that enforces the mandate inside the program audits clean.

    Same shape of mandate as the projected case — a box, a bucket budget, a
    turnover budget against a real prior book — but mean-variance puts all of
    it into the convex program, so there is nothing left for the audit to find.
    Strict mode is then a no-op, which is the property that makes it safe to
    turn on.
    """
    assets = list(returns.columns)
    equities = assets[:3]
    layer = ConstraintLayer(
        name="asset_class",
        assignments={a: ("Equity" if a in equities else "Other") for a in assets},
        limits={"Equity": (0.0, 0.20)},
    )
    constraints = PortfolioConstraints(
        bounds={a: (0.0, 0.30) for a in assets},
        constraint_layers=(layer,),
        previous_weights={a: 1.0 / len(assets) for a in assets},
        turnover_limit=0.50,
    )

    result = MeanVarianceOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        constraints=constraints,
        risk_aversion=3.0,
    ).optimize()

    assert result.extras["bounds_mode"] == "hard"
    assert result.audit is not None
    assert result.audit.is_clean, result.audit.describe()
    assert result.audit.worst is None
    assert result.violations == []

    strict = MeanVarianceOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        constraints=constraints,
        risk_aversion=3.0,
        strict_mandate=True,
    ).optimize()
    assert np.allclose(strict.weights.values, result.weights.values)


# ---------------------------------------------------------------------------
# The opt-out, and what an absent audit means
# ---------------------------------------------------------------------------


def test_skipping_the_post_solve_pass_reports_nothing_rather_than_compliance(
    cov: pd.DataFrame, mu: pd.Series
):
    """``audit=None`` says "nobody looked", which is not "nothing was wrong"."""
    constraints = PortfolioConstraints(bounds={a: (0.0, 0.30) for a in cov.columns})
    optimizer = MeanVarianceOptimizer(
        expected_returns=mu, cov_matrix=cov, constraints=constraints, risk_aversion=3.0
    )

    result = optimizer.optimize(run_post_solve_diagnostics=False)

    assert result.audit is None
    assert "diagnostics" not in result.extras
    assert "violations" not in result.extras
    # The weights are still cleaned and still carry the solve's own record.
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert result.extras["solver_status"] == "optimal"


def test_strict_mandate_cannot_fire_when_nothing_was_audited(cov: pd.DataFrame):
    """The gate is on the audit, so opting out of the audit opts out of the gate.

    Worth pinning: it is exactly the combination NCO's nested layers use, and a
    strict outer solve must not turn into a refusal one layer down against a
    constraint set that is not the mandate.
    """
    constraints = PortfolioConstraints(previous_weights={}, turnover_limit=0.0)
    optimizer = HRPOptimizer(
        cov_matrix=cov, constraints=constraints, strict_mandate=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimizer.optimize(run_post_solve_diagnostics=False)
    assert result.audit is None


# ---------------------------------------------------------------------------
# The audit reaches a machine consumer
# ---------------------------------------------------------------------------


def test_the_audit_reaches_the_json_payload(cov: pd.DataFrame):
    """``optimization_payload`` enumerates its keys, so a new field needs adding."""
    from optimization_engine.reporting.payloads import (
        SCHEMA_VERSION,
        audit_payload,
        optimization_payload,
    )

    assert audit_payload(None) is None

    constraints = PortfolioConstraints(
        previous_weights={a: (1.0 if a == cov.columns[0] else 0.0) for a in cov.columns},
        turnover_limit=0.10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = HRPOptimizer(cov_matrix=cov, constraints=constraints).optimize()

    class _Run:
        pass

    run = _Run()
    run.result = result
    payload = optimization_payload(run)

    assert SCHEMA_VERSION.startswith("2.")
    audit = payload["audit"]
    assert audit["clean"] is False
    assert audit["tolerance"] == pytest.approx(1e-6)
    breach = next(v for v in audit["violations"] if v["kind"] == "turnover")
    # Numbers, not a repr a consumer has to parse back out of prose.
    assert breach["limit"] == pytest.approx(0.10)
    assert breach["actual"] > 0.10
    assert breach["magnitude"] == pytest.approx(breach["actual"] - breach["limit"])
    assert "Turnover" in breach["message"]
