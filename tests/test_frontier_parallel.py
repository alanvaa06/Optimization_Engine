"""Frontier parallelization correctness."""

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
from optimization_engine.frontier import efficient_frontier


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=11))


@pytest.fixture(scope="module")
def baseline_config(returns: pd.DataFrame) -> EngineConfig:
    expected = (1 + returns).prod() ** (252 / len(returns)) - 1
    return EngineConfig(
        expected_returns=expected.to_dict(),
        bounds={a: [0.0, 0.5] for a in returns.columns},
        groups={a: "All" for a in returns.columns},
        group_bounds={"All": [1.0, 1.0]},
        optimizer=OptimizerSpec(name="mean_variance"),
    )


def test_parallel_matches_serial(returns, baseline_config):
    cov = covariance_matrix(returns, method="ledoit_wolf")
    mu = pd.Series(baseline_config.expected_returns)
    f1 = efficient_frontier(baseline_config, cov, mu, n_points=12, n_workers=1)
    f4 = efficient_frontier(baseline_config, cov, mu, n_points=12, n_workers=4)
    pd.testing.assert_frame_equal(
        f1.summary.drop(columns="status"),
        f4.summary.drop(columns="status"),
        check_exact=False, atol=1e-6,
    )


def test_risk_aversion_sweep_monotone(returns, baseline_config):
    cov = covariance_matrix(returns, method="ledoit_wolf")
    mu = pd.Series(baseline_config.expected_returns)
    fr = efficient_frontier(
        baseline_config, cov, mu,
        n_points=10, sweep="risk_aversion",
    )
    vols = fr.summary["expected_volatility"].dropna().values
    assert (np.diff(vols) <= 1e-6).all(), vols.tolist()


def test_infeasible_targets_surface_as_failed_rows(returns, baseline_config):
    cov = covariance_matrix(returns, method="ledoit_wolf")
    mu = pd.Series(baseline_config.expected_returns)
    fr = efficient_frontier(
        baseline_config, cov, mu,
        n_points=5, return_range=(10.0, 20.0),
    )
    assert fr.summary["status"].str.startswith("failed").any()
