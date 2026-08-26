"""Regression tests for the analytical-rigor fixes.

Each test here pins down a specific way the engine used to be wrong, so a
future refactor cannot quietly reintroduce it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.analytics.risk import (
    downside_deviation,
    drawdown_table,
    max_drawdown_duration,
    omega_ratio,
    tail_ratio,
    ulcer_index,
)
from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.covariance import (
    _semi,
    covariance_diagnostics,
    covariance_matrix,
    james_stein_shrinkage,
    nearest_psd,
)
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.diagnostics import (
    check_constraints,
    diversification_ratio,
    effective_n,
    herfindahl_index,
    risk_decomposition,
)
from optimization_engine.optimizers.feasibility import analyze_feasibility
from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 5, seed=7))


@pytest.fixture(scope="module")
def cov(returns: pd.DataFrame) -> pd.DataFrame:
    return covariance_matrix(returns, method="ledoit_wolf")


@pytest.fixture(scope="module")
def mu(returns: pd.DataFrame) -> pd.Series:
    return (1 + returns).prod() ** (252 / len(returns)) - 1


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


def test_semicovariance_does_not_recentre_deviations(returns):
    """Semicovariance is D'D/(T-1), not the covariance *of* D.

    Using DataFrame.cov() subtracts D's own (negative) mean, understating
    downside co-movement.
    """
    deviation = (returns - returns.mean()).clip(upper=0.0)
    expected = deviation.T.values @ deviation.values / (len(deviation) - 1)
    got = _semi(returns).values
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-14)
    # And it genuinely differs from the re-centred version.
    recentred = deviation.cov(ddof=1).values
    assert np.abs(got - recentred).max() > 1e-8


def test_semicovariance_respects_explicit_mar(returns):
    zero_mar = _semi(returns, mar=0.0)
    mean_mar = _semi(returns, mar=None)
    assert not np.allclose(zero_mar.values, mean_mar.values)


def test_covariance_output_is_psd(returns):
    for method in ("sample", "ledoit_wolf", "oas", "ewma", "semi"):
        c = covariance_matrix(returns, method=method)
        eig = np.linalg.eigvalsh(c.values)
        assert eig.min() >= -1e-12, f"{method} produced a non-PSD matrix"


def test_nearest_psd_repairs_indefinite_matrix():
    bad = pd.DataFrame([[1.0, 0.99], [0.99, 0.5]], index=["a", "b"], columns=["a", "b"])
    assert np.linalg.eigvalsh(bad.values).min() < 0
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed.values).min() >= -1e-12
    assert list(fixed.columns) == ["a", "b"]


def test_covariance_rejects_missing_values(returns):
    dirty = returns.copy()
    dirty.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        covariance_matrix(dirty)


def test_covariance_diagnostics_flag_thin_samples(returns):
    short = returns.iloc[:20]
    diag = covariance_diagnostics(
        covariance_matrix(short, method="ledoit_wolf"), n_observations=len(short)
    )
    assert not diag.is_reliable
    assert any("observations" in w for w in diag.warnings)


def test_james_stein_pulls_means_toward_the_centre(returns, cov, mu):
    shrunk, intensity = james_stein_shrinkage(mu, cov, len(returns))
    assert 0.0 <= intensity <= 1.0
    if intensity > 0:
        assert shrunk.std() < mu.std()


# ---------------------------------------------------------------------------
# Downside risk
# ---------------------------------------------------------------------------


def test_downside_deviation_averages_over_all_periods():
    r = pd.Series([0.02, -0.01, 0.03, -0.04, 0.01])
    expected = float(np.sqrt((np.minimum(r, 0.0) ** 2).mean()))
    assert downside_deviation(r) == pytest.approx(expected)
    # The old shortcut — std of the negative subset — is materially smaller.
    assert float(r[r < 0].std(ddof=0)) < expected


def test_downside_deviation_uses_the_mar():
    r = pd.Series([0.02, 0.01, 0.03, 0.015])
    assert downside_deviation(r, mar=0.0) == pytest.approx(0.0)
    assert downside_deviation(r, mar=0.05) > 0.0


def test_drawdown_table_reports_timing_and_recovery():
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    r = pd.Series([0.10, -0.20, -0.10, 0.05, 0.40, 0.01, -0.05, 0.02], index=idx)
    table = drawdown_table(r)
    assert not table.empty
    worst = table.iloc[0]
    assert worst["max_drawdown"] < 0
    assert worst["decline_periods"] >= 1
    assert max_drawdown_duration(r) >= worst["total_periods"]


def test_ulcer_index_separates_equal_max_drawdowns():
    """Same depth, different time underwater — Ulcer must tell them apart."""
    idx = pd.date_range("2020-01-01", periods=11, freq="D")
    # A peak first, so the drop is measured from somewhere.
    quick = pd.Series([0.05, -0.10, 0.12] + [0.0] * 8, index=idx)
    slow = pd.Series([0.05, -0.10] + [0.0] * 8 + [0.12], index=idx)
    assert ulcer_index(slow) > ulcer_index(quick)


def test_omega_and_tail_ratio_are_finite(returns):
    assert np.isfinite(omega_ratio(returns.iloc[:, 0]))
    assert np.isfinite(tail_ratio(returns.iloc[:, 0]))


# ---------------------------------------------------------------------------
# Optimizer correctness
# ---------------------------------------------------------------------------


def test_max_diversification_solves_the_bounded_problem(returns, cov):
    """Bounded max-DR must match the exact constrained optimum, not a projection."""
    import cvxpy as cp

    cons = PortfolioConstraints(bounds={a: (0.0, 0.15) for a in cov.columns})
    got = MaxDiversificationOptimizer(cov_matrix=cov, constraints=cons).optimize()

    std = np.sqrt(np.diag(cov.values))
    n = len(std)
    y = cp.Variable(n, nonneg=True)
    k = cp.Variable(nonneg=True)
    cp.Problem(
        cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov.values))),
        [std @ y == 1, cp.sum(y) == k, y <= 0.15 * k, k >= 1e-8],
    ).solve()
    reference = np.asarray(y.value) / float(k.value)
    ref_dr = (reference @ std) / np.sqrt(reference @ cov.values @ reference)

    assert diversification_ratio(got.weights, cov) == pytest.approx(ref_dr, rel=1e-3)
    assert got.is_compliant


def test_max_diversification_honours_group_bounds(returns, cov):
    groups = {a: ("Equity" if "Equity" in a else "Other") for a in cov.columns}
    cons = PortfolioConstraints(
        bounds={a: (0.0, 0.5) for a in cov.columns},
        groups=groups,
        group_bounds={"Equity": (0.0, 0.20)},
    )
    res = MaxDiversificationOptimizer(cov_matrix=cov, constraints=cons).optimize()
    equity = sum(v for a, v in res.weights.items() if groups[a] == "Equity")
    assert equity <= 0.20 + 1e-6
    assert res.is_compliant


def test_long_only_tightens_negative_bounds():
    cons = PortfolioConstraints(bounds={"A": (-0.5, 0.5)}, long_only=True)
    assert cons.get_bounds("A") == (0.0, 0.5)
    cons.long_only = False
    assert cons.get_bounds("A") == (-0.5, 0.5)


def test_clean_weights_never_leaves_the_box(returns, cov, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.02, 0.20] for a in cov.columns},
        optimizer=OptimizerSpec(name="min_variance"),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w >= 0.02 - 1e-6).all()
    assert (w <= 0.20 + 1e-6).all()
    assert run.result.is_compliant


def test_missing_expected_returns_warn_rather_than_silently_zero(cov):
    from optimization_engine.optimizers.mean_variance import MeanVarianceOptimizer

    partial = pd.Series({cov.columns[0]: 0.08})
    opt = MeanVarianceOptimizer(
        expected_returns=partial, cov_matrix=cov, risk_aversion=2.0
    )
    with pytest.warns(UserWarning, match="no expected return"):
        opt.optimize()


def test_turnover_budget_is_enforced_end_to_end(returns, mu):
    previous = {a: 1.0 / len(mu) for a in mu.index}
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 1.0] for a in mu.index},
        previous_weights=previous,
        turnover_limit=0.10,
        optimizer=OptimizerSpec(name="mean_variance", risk_aversion=1.0),
    )
    run = run_engine(returns, cfg)
    prev = pd.Series(previous).reindex(run.result.weights.index)
    turnover = float((run.result.weights - prev).abs().sum())
    assert turnover <= 0.10 + 1e-4
    assert run.result.is_compliant


def test_unconstrained_turnover_is_larger_than_the_budget(returns, mu):
    """The budget above is binding — otherwise the test proves nothing."""
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 1.0] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_aversion=1.0),
    )
    run = run_engine(returns, cfg)
    prev = pd.Series({a: 1.0 / len(mu) for a in mu.index}).reindex(
        run.result.weights.index
    )
    assert float((run.result.weights - prev).abs().sum()) > 0.10


def test_leverage_and_short_selling_round_trip(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [-0.3, 0.5] for a in mu.index},
        long_only=False,
        leverage=1.6,
        optimizer=OptimizerSpec(name="mean_variance", risk_aversion=1.0),
    )
    run = run_engine(returns, cfg)
    assert float(run.result.weights.abs().sum()) <= 1.6 + 1e-4
    assert run.result.weights.min() >= -0.3 - 1e-6


def test_risk_parity_reports_budget_error(returns, cov):
    from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

    res = RiskParityOptimizer(
        cov_matrix=cov, constraints=PortfolioConstraints()
    ).optimize()
    assert res.extras["risk_budget_max_error"] < 1e-3


def test_risk_parity_rejects_zero_budget(cov):
    from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

    budget = {a: 1.0 for a in cov.columns}
    budget[cov.columns[0]] = 0.0
    with pytest.raises(ValueError, match="zero"):
        RiskParityOptimizer(cov_matrix=cov, risk_budget=budget).optimize()


def test_cvar_reports_realized_tail_metrics(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.4] for a in mu.index},
        optimizer=OptimizerSpec(name="cvar", cvar_alpha=0.05),
    )
    run = run_engine(returns, cfg)
    assert run.result.extras["cvar_period"] > 0
    assert run.result.extras["tail_observations"] > 0
    assert run.result.extras["cvar_annualized"] > run.result.extras["cvar_period"]


def test_cvar_rejects_a_confidence_level_passed_as_alpha(returns):
    from optimization_engine.optimizers.cvar import CVaROptimizer

    with pytest.raises(ValueError, match="tail probability"):
        CVaROptimizer(returns=returns, alpha=0.95)


def test_hrp_warns_instead_of_silently_dropping_group_bounds(returns, cov):
    from optimization_engine.optimizers.hrp import HRPOptimizer

    cons = PortfolioConstraints(
        groups={a: "All" for a in cov.columns}, group_bounds={"All": (1.0, 1.0)}
    )
    with pytest.warns(UserWarning, match="group budgets"):
        HRPOptimizer(cov_matrix=cov, constraints=cons).optimize()


def test_max_sharpe_warns_that_turnover_cannot_bind(cov, mu):
    from optimization_engine.optimizers.mean_variance import MaxSharpeOptimizer

    cons = PortfolioConstraints(
        previous_weights={a: 1.0 / len(mu) for a in mu.index}, turnover_limit=0.05
    )
    with pytest.warns(UserWarning, match="turnover"):
        MaxSharpeOptimizer(
            expected_returns=mu, cov_matrix=cov, constraints=cons
        ).optimize()


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------


def test_relative_view_moves_the_spread_toward_the_view(cov):
    from optimization_engine.optimizers.black_litterman import (
        View,
        black_litterman_posterior,
        build_pick_matrix,
    )

    mkt = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    view = View({"US_Equity": 1.0, "EM_Equity": -1.0}, 0.03)
    prior, _ = black_litterman_posterior(cov, mkt, None)
    post, _ = black_litterman_posterior(cov, mkt, [view])
    P, _, _ = build_pick_matrix([view], list(cov.columns))
    prior_spread = float((P @ prior.values)[0])
    post_spread = float((P @ post.values)[0])
    assert prior_spread < post_spread <= 0.03


def test_absolute_view_mapping_still_supported(cov):
    from optimization_engine.optimizers.black_litterman import black_litterman_posterior

    mkt = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    prior, _ = black_litterman_posterior(cov, mkt, None)
    post, _ = black_litterman_posterior(cov, mkt, {"Gold": 0.10})
    assert post["Gold"] > prior["Gold"]


def test_he_litterman_omega_scales_with_the_view_portfolio(cov):
    from optimization_engine.optimizers.black_litterman import View, build_pick_matrix

    view = View({"US_Equity": 1.0, "EM_Equity": -1.0}, 0.03)
    P, _, _ = build_pick_matrix([view], list(cov.columns))
    omega = float(np.diag(P @ (0.05 * cov.values) @ P.T)[0])
    naive = 0.05 * float(cov.loc["US_Equity", "US_Equity"])
    # A spread view's uncertainty is not one leg's variance.
    assert abs(omega - naive) > 1e-8


def test_view_confidence_must_be_positive(cov):
    from optimization_engine.optimizers.black_litterman import View, black_litterman_posterior

    mkt = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    with pytest.raises(ValueError, match="positive"):
        black_litterman_posterior(
            cov, mkt, [View({"Gold": 1.0}, 0.10, confidence=0.0)]
        )


def test_implied_risk_aversion_matches_market_sharpe(cov):
    from optimization_engine.optimizers.black_litterman import implied_risk_aversion

    mkt = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    var = float(mkt.values @ cov.values @ mkt.values)
    delta = implied_risk_aversion(0.08, var, 0.02)
    assert delta == pytest.approx((0.08 - 0.02) / var)


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


def test_feasibility_catches_caps_below_the_budget(cov, mu):
    cons = PortfolioConstraints(bounds={a: (0.0, 0.05) for a in cov.columns})
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert not report.is_feasible
    assert any(i.code == "max_weights_below_budget" for i in report.issues)


def test_feasibility_catches_floors_above_the_budget(cov, mu):
    cons = PortfolioConstraints(bounds={a: (0.20, 1.0) for a in cov.columns})
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert any(i.code == "min_weights_exceed_budget" for i in report.issues)


def test_feasibility_catches_unreachable_group_minimum(cov, mu):
    groups = {a: ("Equity" if "Equity" in a else "Other") for a in cov.columns}
    cons = PortfolioConstraints(
        bounds={a: (0.0, 0.05) for a in cov.columns},
        groups=groups,
        group_bounds={"Equity": (0.60, 1.0)},
    )
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert any(i.code == "group_min_unreachable" for i in report.issues)


def test_feasibility_bounds_the_return_target(cov, mu):
    cons = PortfolioConstraints(
        bounds={a: (0.0, 0.15) for a in cov.columns}, target_return=5.0
    )
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert any(i.code == "target_return_too_high" for i in report.issues)
    assert report.max_return is not None and report.max_return < 5.0


def test_feasibility_flags_a_dominated_target_without_blocking(cov, mu):
    cons = PortfolioConstraints(bounds={a: (0.0, 0.5) for a in cov.columns})
    base = analyze_feasibility(list(cov.columns), cons, mu, cov)
    cons.target_return = base.min_variance_return - 0.01
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    assert report.is_feasible  # solvable, just not efficient
    assert any(i.code == "target_return_inefficient" for i in report.issues)


def test_feasibility_reports_are_actionable(cov, mu):
    cons = PortfolioConstraints(bounds={a: (0.0, 0.05) for a in cov.columns})
    report = analyze_feasibility(list(cov.columns), cons, mu, cov)
    for issue in report.issues:
        assert issue.suggestion, f"{issue.code} has no suggested fix"


def test_run_engine_can_raise_on_infeasible(returns, mu):
    from optimization_engine.optimizers.feasibility import InfeasibleConstraintsError

    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.05] for a in mu.index},
        optimizer=OptimizerSpec(name="min_variance"),
    )
    with pytest.raises(InfeasibleConstraintsError, match="cannot be satisfied"):
        run_engine(returns, cfg, raise_on_infeasible=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_effective_n_reflects_concentration():
    equal = pd.Series([0.25] * 4)
    concentrated = pd.Series([0.90, 0.05, 0.03, 0.02])
    assert effective_n(equal) == pytest.approx(4.0)
    assert effective_n(concentrated) < 1.5
    assert herfindahl_index(equal) == pytest.approx(0.25)


def test_risk_decomposition_sums_to_portfolio_volatility(cov):
    w = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    table = risk_decomposition(w, cov)
    port_vol = float(np.sqrt(w.values @ cov.values @ w.values))
    assert table["contribution"].sum() == pytest.approx(port_vol, rel=1e-9)
    assert table["share_of_risk"].sum() == pytest.approx(1.0, rel=1e-9)


def test_check_constraints_detects_every_breach_kind():
    w = pd.Series({"A": 0.9, "B": 0.3, "C": -0.1})  # sums to 1.1, not 1
    cons = PortfolioConstraints(
        bounds={"A": (0.0, 0.5), "B": (0.0, 0.5), "C": (0.0, 0.5)},
        groups={"A": "G", "B": "G", "C": "H"},
        group_bounds={"G": (0.0, 0.6)},
        long_only=True,
    )
    kinds = {v.kind for v in check_constraints(w, cons)}
    assert {"bound", "budget", "group"} <= kinds


def test_compliant_solution_reports_no_violations(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.25] for a in mu.index},
        optimizer=OptimizerSpec(name="max_sharpe", risk_free_rate=0.02),
    )
    run = run_engine(returns, cfg)
    assert run.result.is_compliant, run.result.violations
    assert run.diagnostics is not None
    assert run.diagnostics.effective_n > 1


# ---------------------------------------------------------------------------
# Frontier
# ---------------------------------------------------------------------------


def test_frontier_range_respects_binding_caps(returns, mu):
    """A 15% cap used to make more than half the frontier infeasible."""
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.15] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.02),
    )
    run = run_engine(returns, cfg, build_frontier=True, n_frontier_points=15)
    assert run.frontier.n_failed == 0, run.frontier.failures
    lo, hi = run.frontier.reachable_range
    assert lo < hi < float(mu.max())


def test_frontier_marks_the_anchor_portfolios(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.30] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.02),
    )
    fr = run_engine(returns, cfg, build_frontier=True, n_frontier_points=12).frontier
    assert fr.min_variance is not None and fr.tangency is not None
    # Nothing on the frontier can have a lower volatility than the GMV.
    assert fr.efficient["expected_volatility"].min() >= (
        fr.min_variance["expected_volatility"] - 1e-6
    )
    # And nothing can beat the tangency portfolio's Sharpe ratio.
    assert fr.efficient["sharpe_ratio"].max() <= fr.tangency["sharpe_ratio"] + 1e-6


def test_frontier_excludes_the_dominated_lower_branch(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.30] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance"),
    )
    fr = run_engine(returns, cfg, build_frontier=True, n_frontier_points=10).frontier
    gmv_return = float(fr.min_variance["expected_return"])
    assert (fr.efficient["expected_return"] >= gmv_return - 1e-3).all()
    # Volatility rises monotonically along the efficient branch.
    vols = fr.efficient["expected_volatility"].values
    assert (np.diff(vols) >= -1e-6).all()


def test_max_sharpe_index_is_nan_safe():
    from optimization_engine.frontier import FrontierResult

    summary = pd.DataFrame(
        {
            "target": [0.1, 0.2],
            "expected_return": [np.nan, np.nan],
            "expected_volatility": [np.nan, np.nan],
            "sharpe_ratio": [np.nan, np.nan],
            "is_efficient": [False, False],
            "status": ["failed: x", "failed: y"],
        }
    )
    fr = FrontierResult(summary=summary, weights=pd.DataFrame())
    assert fr.max_sharpe_index == -1
    assert fr.plot_frame().empty


def test_plot_frame_index_matches_the_highlighted_row(returns, mu):
    cfg = EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.30] for a in mu.index},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.02),
    )
    fr = run_engine(returns, cfg, build_frontier=True, n_frontier_points=10).frontier
    frame = fr.plot_frame()
    best = frame["sharpe_ratio"].idxmax()
    assert frame.loc[best, "sharpe_ratio"] == frame["sharpe_ratio"].max()
