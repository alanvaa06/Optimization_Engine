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
from optimization_engine.optimizers.factory import available_optimizers
from optimization_engine.optimizers import ConfigurationError


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
