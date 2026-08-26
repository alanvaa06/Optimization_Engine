"""Frontier uncertainty and Michaud resampling."""

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
from optimization_engine.resampling import (
    bootstrap_frontier,
    resample_returns,
    resampled_efficient_frontier,
)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 5, seed=13))


@pytest.fixture(scope="module")
def config(returns: pd.DataFrame) -> EngineConfig:
    mu = (1 + returns).prod() ** (252 / len(returns)) - 1
    return EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.4] for a in returns.columns},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.03),
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["iid", "block", "parametric"])
def test_resample_preserves_shape(returns, method):
    drawn = resample_returns(returns, method=method, rng=np.random.default_rng(0))
    assert drawn.shape == returns.shape
    assert list(drawn.columns) == list(returns.columns)


def test_resampling_is_reproducible(returns):
    a = resample_returns(returns, "block", rng=np.random.default_rng(7))
    b = resample_returns(returns, "block", rng=np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)


def test_block_bootstrap_preserves_more_autocorrelation_than_iid():
    """That is the whole reason to prefer blocks for financial returns."""
    rng = np.random.default_rng(0)
    n = 2000
    noise = rng.normal(0, 0.01, n)
    series = np.zeros(n)
    for i in range(1, n):
        series[i] = 0.6 * series[i - 1] + noise[i]
    frame = pd.DataFrame({"x": series})

    block_rho, iid_rho = [], []
    for seed in range(12):
        block_rho.append(
            abs(resample_returns(frame, "block", block_size=60,
                                 rng=np.random.default_rng(seed))["x"].autocorr(1))
        )
        iid_rho.append(
            abs(resample_returns(frame, "iid",
                                 rng=np.random.default_rng(seed))["x"].autocorr(1))
        )
    assert np.mean(block_rho) > np.mean(iid_rho) + 0.1


def test_unknown_resampling_method_is_rejected(returns):
    with pytest.raises(ValueError, match="Unknown resampling method"):
        resample_returns(returns, method="jackknife")


# ---------------------------------------------------------------------------
# Frontier uncertainty
# ---------------------------------------------------------------------------


def test_bootstrap_frontier_produces_an_ordered_band(returns, config):
    u = bootstrap_frontier(returns, config, n_draws=12, n_points=8, seed=3)
    assert u.n_draws >= 8
    q = u.quantiles
    # Quantiles must not cross.
    assert (q["q05"] <= q["q50"] + 1e-9).all()
    assert (q["q50"] <= q["q95"] + 1e-9).all()
    width = u.band_width()
    assert (width >= -1e-12).all()
    assert width.notna().any()


def test_band_is_wider_on_a_shorter_sample(returns, config):
    """Less data means more estimation error, and the band should say so."""
    long_run = bootstrap_frontier(returns, config, n_draws=12, n_points=8, seed=5)
    short = returns.iloc[-250:]
    short_config = EngineConfig(
        expected_returns=config.expected_returns,
        bounds=config.bounds,
        optimizer=config.optimizer,
    )
    short_run = bootstrap_frontier(short, short_config, n_draws=12, n_points=8, seed=5)
    assert float(short_run.band_width().median()) > float(
        long_run.band_width().median()
    )


def test_summary_names_the_band(returns, config):
    u = bootstrap_frontier(returns, config, n_draws=8, n_points=6, seed=1)
    text = u.summary()
    assert "resampled histories" in text and "%" in text


def test_weight_dispersion_flags_the_unstable_positions(returns, config):
    u = bootstrap_frontier(returns, config, n_draws=12, n_points=6, seed=2)
    assert not u.weight_dispersion.empty
    assert (u.weight_dispersion >= 0).all()
    assert u.weight_dispersion.is_monotonic_decreasing


def test_bootstrap_requires_enough_draws(returns, config):
    with pytest.raises(ValueError, match="at least 2 draws"):
        bootstrap_frontier(returns, config, n_draws=1)


def test_point_estimate_is_carried_alongside_the_band(returns, config):
    u = bootstrap_frontier(returns, config, n_draws=8, n_points=6, seed=4)
    assert u.point_estimate.n_failed == 0
    assert not u.point_estimate.plot_frame().empty


# ---------------------------------------------------------------------------
# Michaud resampling
# ---------------------------------------------------------------------------


def test_resampled_frontier_weights_are_valid(returns, config):
    averaged = resampled_efficient_frontier(
        returns, config, n_draws=10, n_points=6, seed=1
    )
    assert not averaged.empty
    np.testing.assert_allclose(averaged.sum().values, 1.0, atol=1e-6)
    assert (averaged.values >= -1e-9).all()
    assert (averaged.values <= 0.4 + 1e-6).all()


def test_resampled_frontier_is_more_diversified_than_the_point_estimate(
    returns, config
):
    """Averaging weights across draws is the point: it stops the optimizer
    acting on differences the sample cannot resolve."""
    averaged = resampled_efficient_frontier(
        returns, config, n_draws=12, n_points=8, seed=1
    )
    point = efficient_frontier(
        config,
        covariance_matrix(returns),
        pd.Series(config.expected_returns),
        n_points=8,
    ).weights

    def effective_n(frame):
        return float(np.mean([1.0 / (frame[c] ** 2).sum() for c in frame.columns]))

    assert effective_n(averaged) > effective_n(point)


def test_resampled_frontier_is_reproducible(returns, config):
    a = resampled_efficient_frontier(returns, config, n_draws=6, n_points=5, seed=9)
    b = resampled_efficient_frontier(returns, config, n_draws=6, n_points=5, seed=9)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def test_point_estimate_is_clipped_to_the_band_range(returns, config):
    """A band that stops mid-plot beside a full-length line reads as broken."""
    from optimization_engine.reporting.plots import plot_frontier_uncertainty

    u = bootstrap_frontier(returns, config, n_draws=10, n_points=8, seed=8)
    fig = plot_frontier_uncertainty(u)
    point = next(t for t in fig.data if str(t.name).startswith("Point"))
    assert min(point.x) >= u.volatility.min() - 1e-9
    assert max(point.x) <= u.volatility.max() + 1e-9


def test_uncertainty_plots_render(returns, config):
    from optimization_engine.reporting.plots import (
        plot_frontier_uncertainty,
        plot_weight_dispersion,
    )

    u = bootstrap_frontier(returns, config, n_draws=8, n_points=6, seed=6)
    fig = plot_frontier_uncertainty(u)
    names = {t.name for t in fig.data}
    assert "Point estimate (observed sample)" in names
    assert any("percentile" in n for n in names)
    assert plot_weight_dispersion(u.weight_dispersion).data
