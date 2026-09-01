"""Smoke tests for the optimization engine."""

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

from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.covariance import covariance_matrix
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.optimizers import ConfigurationError
from optimization_engine.optimizers.factory import available_optimizers


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    prices = sample_dataset(n_periods=252 * 4, seed=7)
    return prices_to_returns(prices)


@pytest.fixture(scope="module")
def baseline_config(returns: pd.DataFrame) -> EngineConfig:
    expected = (1 + returns).prod() ** (252 / len(returns)) - 1
    return EngineConfig(
        expected_returns=expected.to_dict(),
        bounds={a: [0.0, 0.5] for a in returns.columns},
        groups={a: "All" for a in returns.columns},
        group_bounds={"All": [1.0, 1.0]},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.03),
    )


def test_lists_optimizers_includes_core_methods():
    names = available_optimizers()
    for required in (
        "mean_variance",
        "min_variance",
        "max_sharpe",
        "risk_parity",
        "hrp",
        "black_litterman",
        "cvar",
        "max_diversification",
        "equal_weight",
        "inverse_vol",
    ):
        assert required in names


def test_covariance_methods(returns: pd.DataFrame):
    for method in ["sample", "ledoit_wolf", "ewma", "semi"]:
        cov = covariance_matrix(returns, method=method)
        assert cov.shape == (returns.shape[1], returns.shape[1])
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals > -1e-8), f"{method}: cov has negative eigenvalues"


@pytest.mark.parametrize(
    "method",
    [
        "min_variance",
        "max_sharpe",
        "mean_variance",
        "risk_parity",
        "hrp",
        "max_diversification",
        "inverse_vol",
        "equal_weight",
    ],
)
def test_optimizer_runs(returns: pd.DataFrame, baseline_config: EngineConfig, method: str):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=OptimizerSpec(name=method, risk_free_rate=0.03),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0
    assert (w >= -1e-6).all(), "Weights must be non-negative"
    assert (w <= 0.5 + 1e-6).all(), "Weights must respect upper bound"


def test_target_return(returns: pd.DataFrame, baseline_config: EngineConfig):
    target = 0.06
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=OptimizerSpec(name="mean_variance", target_return=target),
    )
    run = run_engine(returns, cfg)
    assert run.result.expected_return == pytest.approx(target, abs=1e-3)


def test_cvar_optimizer(returns: pd.DataFrame, baseline_config: EngineConfig):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=OptimizerSpec(name="cvar", cvar_alpha=0.05),
    )
    run = run_engine(returns, cfg)
    assert pytest.approx(run.result.weights.sum(), abs=1e-3) == 1.0


def test_efficient_frontier(returns: pd.DataFrame, baseline_config: EngineConfig):
    run = run_engine(returns, baseline_config, build_frontier=True, n_frontier_points=10)
    assert run.frontier is not None
    summary = run.frontier.summary.dropna(subset=["expected_return"])
    assert (summary["expected_return"].diff().dropna() >= -1e-6).all()


def test_black_litterman(returns: pd.DataFrame, baseline_config: EngineConfig):
    spec = OptimizerSpec(
        name="black_litterman",
        bl_views={returns.columns[0]: 0.12},
        bl_view_confidences={returns.columns[0]: 0.0001},
        risk_free_rate=0.03,
    )
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=spec,
    )
    run = run_engine(returns, cfg)
    assert run.result.weights.sum() == pytest.approx(1.0, abs=1e-3)


def test_risk_parity_equal_contributions(returns: pd.DataFrame):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 1.0] for a in returns.columns},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    run = run_engine(returns, cfg)
    rc = run.risk_contributions().values
    rc = rc / rc.sum()
    target = np.ones_like(rc) / len(rc)
    assert np.max(np.abs(rc - target)) < 0.05  # ERC: roughly equal


def test_factory_raises_when_required_mu_missing(returns):
    # Test the factory directly: empty config + no override -> ConfigurationError.
    # (run_engine has a historical-mean fallback that fills mu in that case;
    # this validation matters for direct factory use.)
    from optimization_engine.data.covariance import covariance_matrix
    from optimization_engine.optimizers.factory import optimizer_factory

    cfg = EngineConfig(
        expected_returns={},
        bounds={a: [0.0, 1.0] for a in returns.columns},
        optimizer=OptimizerSpec(name="mean_variance"),
    )
    cov = covariance_matrix(returns, method="ledoit_wolf")
    with pytest.raises(ConfigurationError, match="expected_returns"):
        optimizer_factory(cfg, cov, expected_returns=None, returns=returns)


def test_factory_raises_when_returns_missing_for_cvar(returns):
    # CVaR needs the returns DataFrame; we exercise the factory directly
    # because the engine always supplies returns.
    from optimization_engine.data.covariance import covariance_matrix
    from optimization_engine.optimizers.factory import optimizer_factory

    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 1.0] for a in returns.columns},
        optimizer=OptimizerSpec(name="cvar"),
    )
    cov = covariance_matrix(returns, method="ledoit_wolf")
    with pytest.raises(ConfigurationError, match="returns"):
        optimizer_factory(cfg, cov, expected_returns=None, returns=None)


def test_factory_warns_on_incompatible_target_return(returns, baseline_config, caplog):
    import logging
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        optimizer=OptimizerSpec(name="hrp", target_return=0.05),
    )
    with caplog.at_level(logging.WARNING):
        run_engine(returns, cfg)
    assert any("target_return" in r.message for r in caplog.records)


@pytest.mark.parametrize("linkage", ["single", "average", "complete", "ward"])
def test_hrp_linkage_methods(returns, baseline_config, linkage):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        optimizer=OptimizerSpec(name="hrp", hrp_linkage=linkage),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert pytest.approx(w.sum(), abs=1e-3) == 1.0
    assert (w >= -1e-6).all()
    assert (w <= 0.5 + 1e-6).all()


def test_max_diversification_respects_tight_bounds(returns):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 0.3] for a in returns.columns},
        optimizer=OptimizerSpec(name="max_diversification"),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w <= 0.3 + 1e-6).all(), w[w > 0.3].to_dict()
    assert (w >= -1e-6).all()
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0


@pytest.mark.parametrize("method", ["equal_weight", "inverse_vol"])
def test_naive_methods_respect_tight_bounds(returns, method):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 0.2] for a in returns.columns},
        optimizer=OptimizerSpec(name=method),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w <= 0.2 + 1e-6).all()
    assert (w >= -1e-6).all()
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0


def test_constrained_risk_parity_respects_bounds(returns):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.05, 0.25] for a in returns.columns},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w >= 0.05 - 1e-5).all(), w[w < 0.05].to_dict()
    assert (w <= 0.25 + 1e-5).all(), w[w > 0.25].to_dict()
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0


def test_risk_parity_with_group_bounds(returns):
    cols = list(returns.columns)
    half = len(cols) // 2
    groups = {a: ("A" if i < half else "B") for i, a in enumerate(cols)}
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in cols},
        bounds={a: [0.0, 1.0] for a in cols},
        groups=groups,
        group_bounds={"A": [0.45, 0.55], "B": [0.45, 0.55]},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    run = run_engine(returns, cfg)
    g = run.result.weights.groupby(groups).sum()
    assert 0.45 - 1e-3 <= g["A"] <= 0.55 + 1e-3
    assert 0.45 - 1e-3 <= g["B"] <= 0.55 + 1e-3


def test_engine_uses_ema_expected_returns_when_specified(returns):
    cfg = EngineConfig(
        expected_returns={},  # empty -> engine seeds from history
        bounds={a: [0.0, 1.0] for a in returns.columns},
        expected_returns_method="ema",
        ema_span=120,
        optimizer=OptimizerSpec(name="min_variance"),  # min_variance ignores mu but engine still computes it
    )
    run = run_engine(returns, cfg)
    # Sanity: μ vector populated with finite values.
    assert run.expected_returns.notna().all()
    assert run.expected_returns.shape[0] == returns.shape[1]


def test_engine_uses_capm_expected_returns_when_specified(returns):
    cols = list(returns.columns)
    cfg = EngineConfig(
        expected_returns={},
        bounds={a: [0.0, 1.0] for a in cols},
        expected_returns_method="capm",
        market_weights={a: 1.0 / len(cols) for a in cols},
        market_return=0.08,
        optimizer=OptimizerSpec(name="min_variance", risk_free_rate=0.03),
    )
    run = run_engine(returns, cfg)
    assert run.expected_returns.notna().all()


def test_engine_default_method_unchanged(returns, baseline_config):
    # Existing test_optimizer_runs already covers this; just ensure default
    # historical_mean still works when method left at default.
    cfg = EngineConfig(
        expected_returns={},
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        optimizer=OptimizerSpec(name="min_variance"),
    )
    run = run_engine(returns, cfg)
    assert run.expected_returns.notna().all()


def test_cvar_with_target_return(returns, baseline_config):
    target = 0.05
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        optimizer=OptimizerSpec(
            name="cvar", cvar_alpha=0.05, target_return=target,
        ),
    )
    run = run_engine(returns, cfg)
    assert run.result.expected_return >= target - 1e-3


def test_black_litterman_no_views_runs(returns, baseline_config):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=OptimizerSpec(name="black_litterman", risk_aversion=2.5),
    )
    run = run_engine(returns, cfg)
    assert pytest.approx(run.result.weights.sum(), abs=1e-3) == 1.0
    assert (run.result.weights >= -1e-6).all()


@pytest.mark.parametrize("method", [
    "mean_variance", "min_variance", "max_sharpe", "cvar", "black_litterman",
])
def test_group_bounds_enforced_for_hard_methods(returns, method):
    cols = list(returns.columns)
    half = len(cols) // 2
    groups = {a: ("A" if i < half else "B") for i, a in enumerate(cols)}
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in cols},
        bounds={a: [0.0, 1.0] for a in cols},
        groups=groups,
        group_bounds={"A": [0.4, 0.6], "B": [0.4, 0.6]},
        optimizer=OptimizerSpec(name=method, risk_free_rate=0.0),
    )
    run = run_engine(returns, cfg)
    g = run.result.weights.groupby(groups).sum()
    assert 0.4 - 2e-3 <= g["A"] <= 0.6 + 2e-3, g.to_dict()
    assert 0.4 - 2e-3 <= g["B"] <= 0.6 + 2e-3, g.to_dict()


def test_infeasible_target_raises_clearly(returns, baseline_config):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        optimizer=OptimizerSpec(name="mean_variance", target_return=10.0),
    )
    with pytest.raises(RuntimeError, match=r"infeasible|status|Solver"):
        run_engine(returns, cfg)


# ---------------------------------------------------------------------------
# Black-Litterman convention and idempotence; ray-space constraints
# (review fixes O1, O2, O3)
# ---------------------------------------------------------------------------


def _synthetic_cov(n: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    names = [f"X{i}" for i in range(n)]
    return pd.DataFrame(a @ a.T / 50 + np.eye(n) * 0.01, index=names, columns=names)


def test_black_litterman_with_no_views_returns_the_market_portfolio():
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.black_litterman import BlackLittermanOptimizer

    cov = _synthetic_cov()
    market = pd.Series([0.4, 0.25, 0.15, 0.12, 0.08], index=cov.index)
    for rf in (0.0, 0.03):
        bl = BlackLittermanOptimizer(
            cov_matrix=cov,
            market_weights=market,
            constraints=PortfolioConstraints(
                long_only=True, fully_invested=True, bounds={a: (0.0, 1.0) for a in cov.index}
            ),
            risk_free_rate=rf,
            risk_aversion=2.5,
        )
        weights = bl.optimize().weights
        # The defining check of the model: no views, no move. With the
        # risk-aversion conventions of the prior and the sub-solve out of step
        # this landed halfway to the minimum-variance portfolio.
        np.testing.assert_allclose(weights.to_numpy(), market.to_numpy(), atol=1e-3)


def test_black_litterman_solves_are_idempotent():
    from optimization_engine.optimizers.black_litterman import BlackLittermanOptimizer

    cov = _synthetic_cov()
    prior = cov.copy()
    bl = BlackLittermanOptimizer(
        cov_matrix=cov,
        market_weights=pd.Series(0.2, index=cov.index),
        views={"X0": 0.10},
        view_confidences={"X0": 0.0004},
    )
    first = bl.optimize()
    second = bl.optimize()
    third = bl.optimize()
    pd.testing.assert_series_equal(first.weights, second.weights)
    pd.testing.assert_series_equal(first.weights, third.weights)
    assert first.expected_return == pytest.approx(third.expected_return)
    pd.testing.assert_frame_equal(bl.cov_matrix, prior)  # the input is not overwritten
    pd.testing.assert_series_equal(
        first.extras["bl_prior_returns"], third.extras["bl_prior_returns"]
    )


def _levered_tangency_inputs():
    names = ["A", "B", "C", "D"]
    mu = pd.Series([0.10, 0.02, 0.08, -0.03], index=names)
    corr = np.array(
        [[1.0, 0.8, 0.2, 0.7], [0.8, 1.0, 0.1, 0.6], [0.2, 0.1, 1.0, 0.3], [0.7, 0.6, 0.3, 1.0]]
    )
    vol = np.array([0.20, 0.15, 0.25, 0.18])
    cov = pd.DataFrame(np.outer(vol, vol) * corr, index=names, columns=names)
    return mu, cov


def test_max_sharpe_honours_a_leverage_cap():
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.mean_variance import MaxSharpeOptimizer

    mu, cov = _levered_tangency_inputs()
    bounds = {a: (-1.0, 1.0) for a in cov.index}
    free = MaxSharpeOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        constraints=PortfolioConstraints(long_only=False, fully_invested=True, bounds=bounds),
    ).optimize()
    assert free.weights.abs().sum() > 1.3  # the unconstrained tangency wants leverage

    capped = MaxSharpeOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        constraints=PortfolioConstraints(
            long_only=False, fully_invested=True, bounds=bounds, leverage=1.2
        ),
    ).optimize()
    assert capped.weights.abs().sum() <= 1.2 + 1e-6
    assert capped.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert capped.extras["violations"] == []
    assert "ignored_constraints" not in capped.extras


def test_max_diversification_honours_a_leverage_cap():
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer

    _, cov = _levered_tangency_inputs()
    bounds = {a: (-1.0, 1.0) for a in cov.index}
    capped = MaxDiversificationOptimizer(
        cov_matrix=cov,
        constraints=PortfolioConstraints(
            long_only=False, fully_invested=True, bounds=bounds, leverage=1.1
        ),
    ).optimize()
    assert capped.weights.abs().sum() <= 1.1 + 1e-6
    assert capped.extras["violations"] == []


@pytest.mark.parametrize("name", ["max_sharpe", "max_diversification"])
def test_ray_space_solves_report_an_open_budget_as_ignored(name):
    from optimization_engine.optimizers.base import PortfolioConstraints
    from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer
    from optimization_engine.optimizers.mean_variance import MaxSharpeOptimizer

    mu, cov = _levered_tangency_inputs()
    constraints = PortfolioConstraints(
        long_only=True, fully_invested=False, bounds={a: (0.0, 1.0) for a in cov.index}
    )
    if name == "max_sharpe":
        optimizer = MaxSharpeOptimizer(expected_returns=mu, cov_matrix=cov, constraints=constraints)
    else:
        optimizer = MaxDiversificationOptimizer(cov_matrix=cov, constraints=constraints)
    with pytest.warns(UserWarning, match="fully_invested=False"):
        result = optimizer.optimize()
    assert "fully_invested" in result.extras["ignored_constraints"]
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
