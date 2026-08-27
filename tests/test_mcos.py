"""Monte Carlo Optimization Selection, and the config plumbing it needs."""

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

from optimization_engine.config import EngineConfig, OptimizerSpec, load_config, save_config
from optimization_engine.data.covariance import covariance_from_config
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.resampling import monte_carlo_optimization_selection


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 3, seed=5))


@pytest.fixture(scope="module")
def config(returns: pd.DataFrame) -> EngineConfig:
    return EngineConfig(
        bounds={a: [0.0, 0.5] for a in returns.columns},
        optimizer=OptimizerSpec(name="mean_variance", risk_aversion=2.0),
    )


def test_mcos_ranks_the_clustered_methods_above_a_direct_solve(
    returns: pd.DataFrame, config: EngineConfig
):
    """López de Prado's headline result, reproduced on this engine's own code.

    A direct mean-variance solve inverts one ill-conditioned matrix and moves
    a long way between samples; NCO never inverts anything larger than a
    cluster. The gap should be visible in a handful of simulations.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = monte_carlo_optimization_selection(
            returns,
            config,
            methods=("mean_variance", "min_variance", "hrp", "nco"),
            n_simulations=6,
            seed=3,
        )

    rmse = result.weight_rmse
    assert rmse["nco"] < rmse["mean_variance"]
    assert rmse["hrp"] < rmse["mean_variance"]
    assert result.n_simulations == 6
    assert set(result.ranking().columns) == {
        "weight_rmse",
        "max_weight_drift",
        "volatility_error",
        "return_error",
    }
    assert result.ranking().index[0] == rmse.idxmin()
    assert "recovered the true allocation" in result.describe()


def test_mcos_reports_every_method_it_was_asked_about(
    returns: pd.DataFrame, config: EngineConfig
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = monte_carlo_optimization_selection(
            returns, config, methods=("equal_weight", "inverse_vol"), n_simulations=3
        )
    assert set(result.weight_rmse.index) == {"equal_weight", "inverse_vol"}
    # 1/N estimates nothing, so it cannot be wrong about anything.
    assert result.weight_rmse["equal_weight"] == pytest.approx(0.0, abs=1e-9)


def test_mcos_rejects_a_degenerate_experiment(
    returns: pd.DataFrame, config: EngineConfig
):
    with pytest.raises(ValueError, match="at least 2 simulations"):
        monte_carlo_optimization_selection(returns, config, n_simulations=1)
    with pytest.raises(ValueError, match="at least one optimizer"):
        monte_carlo_optimization_selection(returns, config, methods=(), n_simulations=3)


def test_mcos_names_a_method_that_cannot_solve_the_universe(
    returns: pd.DataFrame, config: EngineConfig
):
    with pytest.raises(ValueError, match="could not be solved"):
        monte_carlo_optimization_selection(
            returns, config, methods=("not_an_optimizer",), n_simulations=3
        )


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_denoise_settings_round_trip_through_yaml(tmp_path: Path):
    original = EngineConfig(
        covariance_method="denoised",
        denoise=True,
        denoise_method="targeted_shrinkage",
        denoise_alpha=0.25,
        detone=1,
        optimizer=OptimizerSpec(
            name="nco",
            nco_objective="max_sharpe",
            cluster_linkage="average",
            n_clusters=4,
            herc_risk_measure="cdar",
            cdar_alpha=0.02,
        ),
    )
    path = tmp_path / "config.yaml"
    save_config(original, path)
    loaded = load_config(path)

    assert loaded.denoise is True
    assert loaded.denoise_method == "targeted_shrinkage"
    assert loaded.denoise_alpha == 0.25
    assert loaded.detone == 1
    assert loaded.optimizer.nco_objective == "max_sharpe"
    assert loaded.optimizer.cluster_linkage == "average"
    assert loaded.optimizer.n_clusters == 4
    assert loaded.optimizer.herc_risk_measure == "cdar"
    assert loaded.optimizer.cdar_alpha == 0.02


def test_covariance_from_config_honours_the_denoise_flags(returns: pd.DataFrame):
    plain = EngineConfig(covariance_method="sample")
    filtered = EngineConfig(covariance_method="sample", denoise=True)

    assert "denoise_report" not in covariance_from_config(returns, plain).attrs
    denoised = covariance_from_config(returns, filtered)
    report = denoised.attrs["denoise_report"]
    assert report.n_signal_eigenvalues >= 1
    # Volatilities are a property of each asset, not of the co-movement, so
    # the filter must leave the diagonal alone.
    assert np.allclose(
        np.diag(denoised.values),
        np.diag(covariance_from_config(returns, plain).values),
    )
