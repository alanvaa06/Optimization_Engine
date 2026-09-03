"""The conventions this package commits to, pinned so a change has to break a test.

Every number below could reasonably have been defined the other way. The point
of this file is not that these choices are the only defensible ones — it is
that the package makes *one* choice per quantity, states it, and computes it
that way everywhere. A convention that lives only in a docstring is a
convention a refactor can silently reverse, and the reader who trusted the
docstring will never find out.

Five conventions, one section each:

1. **One Sharpe.** ``analytics.performance.sharpe_ratio`` is the only
   implementation, and it is arithmetic by default.
2. **Arithmetic μ.** ``expected_returns_from_history(..., "mean")`` is
   ``r̄ · periods_per_year``; the compounding estimator answers to
   ``"geometric_mean"``.
3. **Risk aversion carries no ½.** The utility is ``μ'w − λ·w'Σw``, which is
   why Black-Litterman hands its mean-variance sub-solve ``δ/2``.
4. **CVaR's ``alpha`` is the tail probability.** ``alpha=0.05`` means the worst
   5%, not the best 95%.
5. **Every labelled quantity is what its label says** — ``sqrt_t_scaled`` is
   scaled by √T, ``zeta`` is a threshold and ``objective`` is an objective, a
   reported ``bounds_mode`` is one the registry admits, and a fallback never
   describes itself as an optimum.

Where a convention already has an exhaustive test elsewhere, that test is named
in the docstring rather than copied. This file is the index; the cross-
references are load-bearing.
"""

from __future__ import annotations

import ast
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

from optimization_engine.analytics.performance import rolling_metrics, sharpe_ratio
from optimization_engine.analytics.selection import _period_sharpe
from optimization_engine.backtest.sweep import _cell_metrics
from optimization_engine.data.covariance import expected_returns_from_history
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.optimizers.base import PortfolioConstraints

PPY = 252


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=PPY * 3, seed=5)).dropna()


@pytest.fixture(scope="module")
def small(returns: pd.DataFrame) -> pd.DataFrame:
    """Six assets — enough for a non-degenerate solve, small enough to be quick."""
    return returns[list(returns.columns)[:6]]


@pytest.fixture(scope="module")
def stream(returns: pd.DataFrame) -> pd.Series:
    return returns.mean(axis=1).rename("equal_weight")


# ---------------------------------------------------------------------------
# 1. One Sharpe definition site
# ---------------------------------------------------------------------------


def _function_node(module_path: Path, name: str) -> ast.FunctionDef:
    """The AST node for a top-level or nested function called ``name``."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{module_path.name} no longer defines {name}()")


#: Every function in the tree that reports a Sharpe ratio, and the module it
#: lives in. Each one must *call* the single definition rather than write the
#: ratio out again — which is how all three used to disagree.
_SHARPE_CONSUMERS = (
    ("analytics/selection.py", "_period_sharpe"),
    ("analytics/performance.py", "rolling_metrics"),
    ("backtest/sweep.py", "_cell_metrics"),
)


def test_sharpe_has_exactly_one_implementation() -> None:
    """Structural pin: every Sharpe in the tree defers to the definition site.

    Three implementations used to coexist — this one, an open-coded arithmetic
    ratio inside ``rolling_metrics``, and a per-period one in
    ``analytics.selection`` — so ``deflated_sharpe_ratio`` deflated an
    arithmetic number against a distribution of geometric ones. The numbers
    agreeing today (checked below) is a property of the current source; this
    check is what stops someone restoring the local copy for speed and
    re-opening the same gap. ``rolling_metrics`` in particular carries a
    comment measuring the cost of *not* open-coding it, which is exactly the
    trade a future editor will be tempted to reverse.

    The reference is looked for as a *name*, not as a call: ``rolling_metrics``
    hands ``sharpe_ratio`` to ``rolling.apply`` rather than invoking it, and
    that is still deferring to the one definition.
    """
    for relative, function in _SHARPE_CONSUMERS:
        node = _function_node(SRC / "optimization_engine" / relative, function)
        names = {
            child.id for child in ast.walk(node) if isinstance(child, ast.Name)
        }
        assert "sharpe_ratio" in names, (
            f"{relative}::{function} no longer refers to sharpe_ratio; if it "
            "computes the ratio itself again, the two definitions will drift "
            "and nothing else in the suite will say so"
        )


def test_every_sharpe_in_the_tree_agrees_to_1e_12(stream: pd.Series) -> None:
    """All three consumers return the single definition's number on one input.

    The structural check above says they call it; this says the call is
    parameterized so as to mean the same thing. ``tests/test_sharpe_convention.py
    ::test_sharpe_single_source`` checks the rolling column window by window —
    this is the convention pin, one input, all three at once.

    The tolerance is 1e-12 rather than a relative approx because these are the
    *same computation*, not two estimators of one quantity: anything above
    float noise means a second formula has appeared somewhere.
    """
    # Selection's per-period Sharpe is this one on a one-period-per-year clock.
    assert float(sharpe_ratio(stream, periods_per_year=1)) == pytest.approx(
        _period_sharpe(stream), abs=1e-12, rel=0.0
    )

    # The rolling frame's last row is the Sharpe of the last window.
    window = 126
    rolling = rolling_metrics(stream, window, riskfree_rate=0.02, periods_per_year=PPY)
    assert float(rolling["rolling_sharpe"].iloc[-1]) == pytest.approx(
        float(sharpe_ratio(stream.iloc[-window:], 0.02, PPY)), abs=1e-12, rel=0.0
    )

    # And a sweep cell's ``sharpe`` column is the same number over the cell's
    # own history.
    assert _cell_metrics(stream, PPY)["sharpe"] == pytest.approx(
        float(sharpe_ratio(stream, 0.0, PPY)), abs=1e-12, rel=0.0
    )


def test_a_sweep_cell_is_scored_on_its_own_window(returns: pd.DataFrame) -> None:
    """The per-cell column measures the cell, not the block the cells share.

    The complement to ``test_dsr_uses_arithmetic_trials``, which checks the
    *aligned* Sharpes on a grid whose cells all span one index — where the two
    readings coincide and the distinction is invisible. Cells of unequal length
    are where a mixed convention would show up: ``SweepResults.trial_sharpes()``
    deliberately re-measures on the shared block so the deflation and the
    overfitting report describe one sample, while the ``sharpe`` column stays a
    description of the cell. Both are right; conflating them is not.
    """
    long_stream = returns.mean(axis=1)
    short_stream = long_stream.iloc[-200:]

    assert _cell_metrics(short_stream, PPY)["sharpe"] == pytest.approx(
        float(sharpe_ratio(short_stream, 0.0, PPY)), abs=1e-12, rel=0.0
    )
    # Not the same number as the full history's, or the test proves nothing.
    assert _cell_metrics(short_stream, PPY)["sharpe"] != pytest.approx(
        _cell_metrics(long_stream, PPY)["sharpe"], abs=1e-6
    )


def test_the_default_is_arithmetic_and_geometric_is_genuinely_different() -> None:
    """The default annualizes the mean; ``"geometric"`` compounds it.

    Constructed rather than measured on the sample panel, so that the sign of
    the difference is derivable instead of observed. Deriving it is what makes
    this a statement about the convention rather than about one panel;
    ``tests/test_sharpe_convention.py`` carries the panel's numbers, 0.5950
    against 0.6238.

    The construction is not arbitrary. Two terms separate the two annualized
    numerators: variance drag ``ppy·σ²/2``, which pulls the geometric number
    down, and the convexity of compounding ``ppy(ppy−1)/2·ḡ²``, which pushes it
    back up. The second wins on a near-constant series — which is why
    ``test_expected_returns.py`` excludes this panel's ``Cash`` from its own
    version of this comparison — so the mean is re-centred to a value where
    ``σ ≈ 50·ḡ``, comfortably past the ``σ > √(ppy−1)·ḡ ≈ 16·ḡ`` crossover.
    """
    rng = np.random.default_rng(20260902)
    drawn = pd.Series(rng.normal(0.0, 0.015, 4000))
    # Re-centred so the mean is exact rather than sampled: the crossover above
    # is a statement about ``ḡ``, and a drawn mean would leave it to luck.
    series = drawn - drawn.mean() + 0.0003

    arithmetic = float(sharpe_ratio(series, 0.0, PPY, method="arithmetic"))
    geometric = float(sharpe_ratio(series, 0.0, PPY, method="geometric"))

    assert float(sharpe_ratio(series, 0.0, PPY)) == arithmetic, (
        "the unqualified Sharpe must be the arithmetic one: it is the estimator "
        "the deflated and probabilistic Sharpe ratios are derived for"
    )
    assert arithmetic == pytest.approx(
        float(series.mean()) * PPY / float(series.std(ddof=1) * np.sqrt(PPY))
    )
    assert geometric < arithmetic, "variance drag has the wrong sign"
    # And the gap is the size the convention predicts — half the annualized
    # variance over the annualized volatility — not float noise. The residual
    # (about 4% here) is the compounding term named above, which is why this
    # is a 15% tolerance and not a 1e-12 one.
    assert arithmetic - geometric == pytest.approx(
        float(series.var(ddof=1)) * PPY / 2.0 / float(series.std(ddof=1) * np.sqrt(PPY)),
        rel=0.15,
    )


# ---------------------------------------------------------------------------
# 2. μ is the arithmetic mean
# ---------------------------------------------------------------------------


def test_mean_is_arithmetic_and_geometric_mean_compounds(small: pd.DataFrame) -> None:
    """The two estimators, and which name each answers to.

    Mean-variance is a single-period model: it trades ``μ'w`` against
    ``λ·w'Σw`` over one period, so μ must be the expectation of one period's
    return. The geometric mean is the realized compound growth rate, lower by
    roughly ``σ²/2`` — the very quantity being penalized on the other side of
    the trade-off, which is why pairing it with an arithmetic Σ is not
    conservatism but a units error.

    ``tests/test_expected_returns.py`` is the full treatment: the ``σ²/2`` gap,
    shrinkage, CAPM, and an AST scan proving the *compounding* formula has one
    definition site. Pinned again here, briefly, because this is the file a
    reader opens to ask "what is μ in this package".
    """
    pd.testing.assert_series_equal(
        expected_returns_from_history(small, method="mean", periods_per_year=PPY),
        small.mean() * PPY,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        expected_returns_from_history(
            small, method="geometric_mean", periods_per_year=PPY
        ),
        ((1 + small).prod() ** (PPY / len(small))) - 1,
        check_names=False,
    )


#: Trees scanned for a second copy of the arithmetic annualizer.
_SCANNED = ("src", "app", "scripts")

#: Tokens that mark a multiplier as an *annualization* rather than arithmetic
#: that happens to involve a mean. Mirrors ``test_expected_returns._ANNUALIZERS``.
_ANNUALIZERS = ("periods_per_year", "ppy", "252", "12", "52")

#: The functions permitted to write ``mean() * periods_per_year`` out, and why.
#: Two annualized-arithmetic-mean quantities exist in this package and they are
#: genuinely different: μ (an expectation, for the optimizer) and a realized
#: excess return (a measurement, for a report). A third copy is a third chance
#: for them to drift.
_ARITHMETIC_ANNUALIZER_SITES = {
    # μ itself — the single definition site the config, the app and the
    # scripts all route through.
    ("src/optimization_engine/data/covariance.py", "expected_returns_from_history"),
    # The arithmetic Sharpe's numerator, inside the single Sharpe definition.
    ("src/optimization_engine/analytics/performance.py", "sharpe_ratio"),
    # A rolling *realized* excess return, the information ratio's numerator.
    # A measurement of what happened, not a forecast, and never fed to a solve.
    ("src/optimization_engine/analytics/report.py", "rolling_relative"),
}


def _arithmetic_annualizer_sites() -> list[tuple[str, int, str]]:
    """Every ``<something>.mean() * <annualizing factor>`` in the scanned trees."""
    found: list[tuple[str, int, str]] = []
    for tree in _SCANNED:
        for path in sorted((ROOT / tree).rglob("*.py")):
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            enclosing: dict[int, str] = {}
            for node in ast.walk(module):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in ast.walk(node):
                        enclosing.setdefault(getattr(inner, "lineno", -1), node.name)
            for node in ast.walk(module):
                if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
                    continue
                has_mean = any(
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "mean"
                    for sub in ast.walk(node.left)
                )
                factor = ast.unparse(node.right)
                if not has_mean or not any(tok in factor for tok in _ANNUALIZERS):
                    continue
                found.append(
                    (
                        str(path.relative_to(ROOT)),
                        node.lineno,
                        enclosing.get(node.lineno, "<module>"),
                    )
                )
    return found


def test_the_arithmetic_annualizer_has_no_fourth_copy() -> None:
    """``r̄ · ppy`` may be written out only where it is the defining site.

    The deliberate complement to
    ``tests/test_expected_returns.py::test_no_other_mu_implementation``, which
    scans for the *compounding* shape ``(1+r).prod() ** (ppy/n)``. That scan
    cannot see this one: the whole point of Task 1.5 was that ``"mean"`` stopped
    compounding and started multiplying, so the shape the old scan hunts is no
    longer the shape μ has. Between them the two scans cover both conventions
    and neither can be reintroduced quietly.
    """
    sites = _arithmetic_annualizer_sites()
    where = {(path, function) for path, _, function in sites}
    unexpected = where - _ARITHMETIC_ANNUALIZER_SITES
    assert not unexpected, (
        "the arithmetic annualizer is written out somewhere new:\n  "
        + "\n  ".join(f"{p}:{n} in {f}()" for p, n, f in sites if (p, f) in unexpected)
        + "\nRoute μ through expected_returns_from_history() and a Sharpe "
        "through sharpe_ratio(); if this really is a fourth quantity, add it "
        "to _ARITHMETIC_ANNUALIZER_SITES with a note saying what it is."
    )
    stale = _ARITHMETIC_ANNUALIZER_SITES - where
    assert not stale, f"these sanctioned sites no longer exist: {sorted(stale)}"


# ---------------------------------------------------------------------------
# 3. Risk aversion: μ'w − λ·w'Σw, with no ½
# ---------------------------------------------------------------------------


def _reference_utility_solve(
    mu: pd.Series,
    sigma: pd.DataFrame,
    constraints: PortfolioConstraints,
    coefficient: float,
) -> np.ndarray:
    """Maximize ``μ'w − coefficient·w'Σw`` under ``constraints``, from scratch.

    Written out here on purpose. The convention under test is a property of an
    expression in ``mean_variance.py``, and the only way to check an expression
    is to state the alternative and show the code does not match it. The
    constraint set comes from the package's own builder so that the comparison
    is against the same feasible region, not a simplified one.
    """
    import cvxpy as cp

    from optimization_engine.optimizers._cvxpy_helpers import (
        build_constraints,
        solve_problem,
    )

    w = cp.Variable(len(mu))
    objective = cp.Maximize(
        mu.values @ w - coefficient * cp.quad_form(w, cp.psd_wrap(sigma.values))
    )
    solve_problem(
        cp.Problem(
            objective,
            build_constraints(w, list(mu.index), constraints, cov_matrix=sigma.values),
        )
    )
    return np.asarray(w.value, dtype=float)


@pytest.mark.parametrize("risk_aversion", [8.0, 25.0])
def test_risk_aversion_carries_no_half(small: pd.DataFrame, risk_aversion: float) -> None:
    """λ multiplies the variance whole: the utility is ``μ'w − λ·w'Σw``.

    Checked numerically rather than by reading the docstring, because the
    docstring is what a refactor leaves behind. The textbook writes the utility
    both ways — with and without the ½ — and the two are not a relabelling:
    at the same nominal λ they pick different portfolios, which is why one
    package's "risk aversion 5" is another's "risk aversion 10". Whichever
    convention a package takes, it has to take it everywhere, and this is
    where the choice is recorded.

    Both risk-aversion values are chosen inside the range where the solution is
    interior, so the two conventions genuinely disagree. At λ=1 this panel goes
    to a corner under either, and a corner solution would let the wrong
    convention pass.
    """
    mu = expected_returns_from_history(small, method="mean", periods_per_year=PPY)
    sigma = small.cov() * PPY
    constraints = PortfolioConstraints()

    from optimization_engine.optimizers.mean_variance import MeanVarianceOptimizer

    solved = (
        MeanVarianceOptimizer(
            expected_returns=mu,
            cov_matrix=sigma,
            constraints=constraints,
            risk_aversion=risk_aversion,
        )
        .optimize()
        .weights.reindex(small.columns)
        .to_numpy(dtype=float)
    )

    no_half = _reference_utility_solve(mu, sigma, constraints, risk_aversion)
    with_half = _reference_utility_solve(mu, sigma, constraints, risk_aversion / 2.0)

    np.testing.assert_allclose(
        solved,
        no_half,
        atol=1e-6,
        err_msg=(
            f"at risk_aversion={risk_aversion} the solve does not maximize "
            "μ'w − λ·w'Σw; the convention this package documents in "
            "OptimizerSpec.risk_aversion has moved"
        ),
    )
    # The alternative convention is a genuinely different portfolio here, so
    # the agreement above is evidence and not a coincidence of the panel.
    assert np.abs(solved - with_half).max() > 1e-3, (
        "the two conventions picked the same portfolio at "
        f"risk_aversion={risk_aversion}, so this test cannot tell them apart; "
        "choose a λ where the solution is interior"
    )


def test_black_litterman_hands_its_subsolve_delta_over_two(
    small: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """δ and λ are different coefficients, and the conversion is the ½.

    Black-Litterman's reverse optimization ``π = δΣw`` is the first-order
    condition of ``μ'w − (δ/2)·w'Σw`` — the textbook's *other* convention.
    Handing that δ straight to a sub-solve that reads it as λ in ``μ'w −
    λ·w'Σw`` doubles the effective aversion, and the model's defining property
    breaks: with no views the posterior is the prior, so the answer must be the
    market portfolio, and at λ=δ it lands halfway to minimum variance instead.

    Both halves are asserted: the coefficient literally passed, and the
    consequence that makes it the right one.
    """
    import optimization_engine.optimizers.black_litterman as bl_module

    sigma = small.cov() * PPY
    market = pd.Series(1.0 / small.shape[1], index=small.columns)
    delta = 7.0
    captured: dict[str, float] = {}

    class RecordingMeanVariance(bl_module.MeanVarianceOptimizer):
        """Records the λ handed down, then solves exactly as it would have."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["risk_aversion"] = float(kwargs["risk_aversion"])  # type: ignore[arg-type]
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(bl_module, "MeanVarianceOptimizer", RecordingMeanVariance)

    result = bl_module.BlackLittermanOptimizer(
        cov_matrix=sigma,
        market_weights=market,
        views={},
        risk_aversion=delta,
        # Shorting allowed so the recovery below is a genuine optimum and not a
        # long-only boundary that any coefficient would have produced.
        constraints=PortfolioConstraints(long_only=False),
    ).optimize()

    assert captured["risk_aversion"] == pytest.approx(delta / 2.0), (
        f"Black-Litterman passed λ={captured['risk_aversion']} for δ={delta}; "
        "π = δΣw is the first-order condition of a utility with a ½ in it, so "
        "the mean-variance solve — which has no ½ — needs δ/2"
    )
    assert result.extras["bl_risk_aversion"] == pytest.approx(delta), (
        "the reported bl_risk_aversion is δ, the reverse-optimization "
        "coefficient, not the λ the sub-solve received"
    )
    np.testing.assert_allclose(
        result.weights.reindex(small.columns).to_numpy(dtype=float),
        market.to_numpy(dtype=float),
        atol=1e-6,
        err_msg=(
            "with no views the posterior is the prior, so the answer must be "
            "the market portfolio; it is not, which means the sub-solve's "
            "effective risk aversion is not δ/2"
        ),
    )


# ---------------------------------------------------------------------------
# 4. CVaR's alpha is the tail probability
# ---------------------------------------------------------------------------

ALPHA = 0.05


@pytest.fixture(scope="module")
def cvar_result(small: pd.DataFrame):
    from optimization_engine.optimizers.cvar import CVaROptimizer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CVaROptimizer(
            returns=small, cov_matrix=small.cov() * PPY, alpha=ALPHA
        ).optimize()


def test_cvar_alpha_is_the_tail_not_the_confidence_level(
    small: pd.DataFrame, cvar_result
) -> None:
    """``alpha=0.05`` reports the worst 5%, not the best 95%.

    The two readings of ``alpha`` are both current in the literature — a tail
    probability of 0.05 and a confidence level of 0.05 name opposite ends of
    the same distribution — and a package that means one while its caller means
    the other reports a small loss as a catastrophe or the reverse. Checked
    against the sample the optimizer actually saw, so this measures the
    reported number rather than re-deriving it from the same code path.

    The tolerance is 3% relative because the reference below counts the tail by
    order statistic while the optimizer's own report uses an interpolated
    quantile, which puts a scenario or two either side of the boundary. That is
    a rounding difference. Reading ``alpha`` the other way is not: the mean of
    the best 95% of this sample is a *gain*, so the wrong convention does not
    land near the right answer, it lands on the opposite sign.
    """
    weights = cvar_result.weights.reindex(small.columns).to_numpy(dtype=float)
    portfolio = np.sort(small.to_numpy(dtype=float) @ weights)
    n_tail = int(round(ALPHA * len(portfolio)))

    worst_five_percent = float(-portfolio[:n_tail].mean())
    best_ninety_five_percent = float(-portfolio[n_tail:].mean())

    reported = float(cvar_result.extras["cvar_period"])
    assert cvar_result.extras["cvar_alpha"] == ALPHA
    assert reported == pytest.approx(worst_five_percent, rel=0.03), (
        f"reported per-period CVaR {reported:.6f} is not the mean loss of the "
        f"worst {ALPHA:.0%} ({worst_five_percent:.6f})"
    )
    assert best_ninety_five_percent < 0 < reported, (
        "the sample was chosen so the two readings of alpha have opposite "
        "signs; if this fails the test can no longer tell them apart"
    )

    # VaR is the threshold the tail sits beyond, so CVaR cannot be smaller.
    assert reported >= float(cvar_result.extras["var_period"])
    # And the solver's own ζ is that threshold, which is what its name claims.
    assert float(cvar_result.extras["cvar_solver_zeta"]) == pytest.approx(
        float(cvar_result.extras["var_period"]), abs=1e-6
    )


# ---------------------------------------------------------------------------
# 5. Every labelled quantity is what its label says
# ---------------------------------------------------------------------------


def test_sqrt_t_scaled_is_scaled_by_sqrt_t(cvar_result) -> None:
    """``cvar_sqrt_t_scaled`` names its own approximation, and performs it.

    The key used to be ``cvar_annualized``, which claimed something the number
    could not deliver: √T scaling of a tail statistic assumes i.i.d. returns,
    and tail risk is where that assumption fails hardest. Renaming it was the
    fix; the name now has to stay true, which means the arithmetic has to match
    the label exactly rather than approximately.
    """
    extras = cvar_result.extras
    scale = float(np.sqrt(PPY))
    assert float(extras["cvar_sqrt_t_scaled"]) == pytest.approx(
        float(extras["cvar_period"]) * scale, rel=0.0, abs=1e-12
    )
    assert float(extras["var_sqrt_t_scaled"]) == pytest.approx(
        float(extras["var_period"]) * scale, rel=0.0, abs=1e-12
    )


def test_cdar_zeta_is_a_threshold_and_the_objective_is_an_objective(
    small: pd.DataFrame,
) -> None:
    """Two different numbers, two different names — they used to share one.

    The solved ``ζ`` is the drawdown-at-risk threshold; the objective is
    ``ζ + Σz/(α·T)``, the CDaR the linear program minimized. Reporting ζ under
    the name ``cdar_solver_objective`` understated the quantity the solve
    optimized, by exactly the tail term.

    Both are recomputed here from the returned weights on the *uncompounded*
    equity curve — the curve the program is stated on — which is the only way
    to show the labels are attached to the right quantities and not merely to
    two distinct numbers.
    """
    from optimization_engine.optimizers.cdar import CDaROptimizer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CDaROptimizer(
            returns=small, cov_matrix=small.cov() * PPY, alpha=ALPHA
        ).optimize()

    extras = result.extras
    zeta = float(extras["cdar_solver_zeta"])
    objective = float(extras["cdar_solver_objective"])

    weights = result.weights.reindex(small.columns).to_numpy(dtype=float)
    equity = np.cumsum(small.to_numpy(dtype=float) @ weights)
    peak = np.maximum(np.maximum.accumulate(equity), 0.0)
    drawdown = peak - equity
    n = len(drawdown)

    # ζ is the threshold: the α-worst drawdown, in the order statistics.
    ranked = np.sort(drawdown)[::-1]
    assert zeta == pytest.approx(float(ranked[int(np.ceil(ALPHA * n)) - 1]), abs=1e-6), (
        f"cdar_solver_zeta={zeta:.8f} is not the drawdown-at-risk threshold; "
        "the name says threshold, so it must be one"
    )
    # The objective is ζ plus the averaged excess over it — the program's value.
    assert objective == pytest.approx(
        zeta + float(np.maximum(drawdown - zeta, 0.0).sum()) / (ALPHA * n), abs=1e-6
    )
    assert objective > zeta, (
        "an objective equal to its own threshold means the tail term vanished, "
        "which is what the old single key looked like"
    )
    assert extras["cdar_alpha"] == ALPHA


def test_a_reported_bounds_mode_is_one_the_registry_admits(small: pd.DataFrame) -> None:
    """The label on the result must be a label the method is allowed to wear.

    ``bounds_mode`` is the claim that decides whether an analyst believes the
    weights honour the mandate, and it appears twice: declared per method in
    ``requirements.py``, and reported per solve in ``extras``. A method that
    declares hard bounds and then projects is mislabelled in the one field
    written to prevent that.

    ``tests/test_optimizers.py::test_result_bounds_mode_matches_its_requirements``
    runs this across every registered optimizer under a binding mandate, and
    ``test_hard_bounds_solves_audit_clean`` checks the label is *true* and not
    merely permitted. Kept here in miniature because "the reported mode is one
    of the declared ones" is the convention itself, and this is where a reader
    looks it up.
    """
    from optimization_engine.optimizers.max_diversification import (
        MaxDiversificationOptimizer,
    )
    from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer
    from optimization_engine.optimizers.requirements import requirements_for

    #: The one-to-many reading of a declared mode. Only ``hard_or_projected``
    #: admits two answers, and only because the method takes the exact solve
    #: when it can and says so per solve when it could not.
    admits = {
        "hard": {"hard"},
        "constrained": {"constrained"},
        "soft_iterated": {"soft_iterated"},
        "hard_or_projected": {"hard", "soft_iterated"},
    }

    sigma = small.cov() * PPY
    for name, optimizer in (
        ("min_variance", MinVarianceOptimizer(cov_matrix=sigma)),
        ("max_diversification", MaxDiversificationOptimizer(cov_matrix=sigma)),
    ):
        declared = requirements_for(name).bounds_mode
        assert declared in admits, f"{name}: undeclared bounds mode {declared!r}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reported = optimizer.optimize().extras["bounds_mode"]
        assert reported in admits[declared], (
            f"{name} declares bounds_mode={declared!r} but its result reports "
            f"{reported!r}, which the registry does not admit for it"
        )


def test_a_fallback_projection_never_claims_optimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key on a fallback result may describe it as an optimum.

    The projection fallback runs an *unconstrained* solve and then projects
    onto the mandate. That inner solve genuinely returns ``optimal``, and
    merging its status wholesale is what used to stamp ``optimal`` on a book
    nobody optimized and which may breach the mandate besides.

    ``tests/test_max_diversification_honesty.py`` pins the individual keys —
    the status, the degraded bounds mode, the absent objective value. The
    convention this adds is the blanket one: whatever keys the fallback grows
    later, none of them may carry the word.
    """
    from optimization_engine.optimizers import max_diversification as max_div
    from optimization_engine.optimizers._cvxpy_helpers import SolverFailure

    assets = ["A", "B", "C", "D"]
    correlation = np.array(
        [
            [1.0, 0.8, 0.2, 0.7],
            [0.8, 1.0, 0.1, 0.6],
            [0.2, 0.1, 1.0, 0.3],
            [0.7, 0.6, 0.3, 1.0],
        ]
    )
    vol = np.array([0.20, 0.15, 0.25, 0.18])
    sigma = pd.DataFrame(np.outer(vol, vol) * correlation, index=assets, columns=assets)

    real_solve = max_div.solve_problem
    calls = {"n": 0}

    def fail_the_constrained_solve(problem, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SolverFailure("solver_error", ("CLARABEL", "SCS"))
        return real_solve(problem, *args, **kwargs)

    monkeypatch.setattr(max_div, "solve_problem", fail_the_constrained_solve)

    result = max_div.MaxDiversificationOptimizer(
        cov_matrix=sigma,
        constraints=PortfolioConstraints(
            bounds={a: (0.0, 0.30) for a in assets},
            benchmark_weights={"A": 1.0},
            max_tracking_error=0.20,
        ),
    ).optimize()

    assert result.extras["solver_status"] == "fallback_projection"
    claims_optimal = {
        key: value
        for key, value in result.extras.items()
        if isinstance(value, str) and "optimal" in value.lower()
    }
    assert not claims_optimal, (
        "a projected fallback reports itself as optimal under "
        f"{sorted(claims_optimal)}; the inner solve that was optimal solved a "
        "different problem from the one the caller asked about"
    )
    # And the bounds label degrades with it: the mandate is now imposed by
    # projection, which is a weaker promise than the hard solve would have made.
    assert result.extras["bounds_mode"] == "soft_iterated"
