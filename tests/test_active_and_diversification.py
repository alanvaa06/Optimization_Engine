"""Grinold-Kahn active analytics and Meucci's effective number of bets."""

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

from optimization_engine.analytics.active import (
    active_risk_decomposition,
    fundamental_law,
    grinold_alpha,
    implied_breadth,
    information_coefficient,
    optimal_active_risk,
    risk_aversion_from_information_ratio,
    transfer_coefficient,
    value_added,
)
from optimization_engine.analytics.diversification import (
    compare_diversification,
    diversification_distribution,
    minimum_torsion,
    pca_torsion,
)
from optimization_engine.data.covariance import covariance_matrix
from optimization_engine.data.loader import prices_to_returns, sample_dataset


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=19))


@pytest.fixture(scope="module")
def cov(returns: pd.DataFrame) -> pd.DataFrame:
    return covariance_matrix(returns, method="ledoit_wolf")


def _equal_weights(cov: pd.DataFrame) -> pd.Series:
    return pd.Series(np.ones(len(cov.columns)) / len(cov.columns), index=cov.columns)


# ---------------------------------------------------------------------------
# Information coefficient
# ---------------------------------------------------------------------------


def test_a_perfect_forecast_scores_an_ic_of_one():
    rng = np.random.default_rng(0)
    realized = pd.DataFrame(
        rng.normal(size=(60, 8)), columns=[f"a{i}" for i in range(8)]
    )
    ic = information_coefficient(realized, realized)
    assert ic.mean == pytest.approx(1.0)
    assert ic.hit_rate == 1.0


def test_a_useless_forecast_scores_an_ic_near_zero():
    rng = np.random.default_rng(1)
    columns = [f"a{i}" for i in range(20)]
    forecasts = pd.DataFrame(rng.normal(size=(400, 20)), columns=columns)
    realized = pd.DataFrame(rng.normal(size=(400, 20)), columns=columns)
    ic = information_coefficient(forecasts, realized)
    assert abs(ic.mean) < 0.05
    assert abs(ic.t_statistic) < 2.5


def test_a_reversed_forecast_scores_a_negative_ic():
    rng = np.random.default_rng(2)
    realized = pd.DataFrame(
        rng.normal(size=(50, 10)), columns=[f"a{i}" for i in range(10)]
    )
    ic = information_coefficient(-realized, realized)
    assert ic.mean == pytest.approx(-1.0)


def test_information_coefficient_aligns_on_the_shared_universe():
    rng = np.random.default_rng(3)
    realized = pd.DataFrame(rng.normal(size=(40, 6)), columns=list("abcdef"))
    forecasts = realized[["a", "b", "c", "d"]]
    ic = information_coefficient(forecasts, realized)
    assert ic.n_assets == 4
    assert ic.mean == pytest.approx(1.0)


def test_information_coefficient_needs_a_shared_universe():
    frame = pd.DataFrame(np.zeros((5, 2)), columns=["a", "b"])
    other = pd.DataFrame(np.zeros((5, 2)), columns=["x", "y"])
    with pytest.raises(ValueError, match="share no dates or no assets"):
        information_coefficient(frame, other)


# ---------------------------------------------------------------------------
# Transfer coefficient
# ---------------------------------------------------------------------------


def test_the_unconstrained_optimal_book_transfers_everything(cov: pd.DataFrame):
    alphas = pd.Series(
        np.linspace(-0.02, 0.02, len(cov.columns)), index=cov.columns
    )
    optimal = pd.Series(np.linalg.pinv(cov.values) @ alphas.values, index=cov.columns)
    assert transfer_coefficient(alphas, optimal, cov) == pytest.approx(1.0, abs=1e-6)


def test_scaling_the_active_book_does_not_change_the_transfer(cov: pd.DataFrame):
    """TC is a correlation, so gearing up the same bets must not move it."""
    alphas = pd.Series(
        np.linspace(-0.02, 0.02, len(cov.columns)), index=cov.columns
    )
    active = pd.Series(
        np.linspace(-0.05, 0.05, len(cov.columns)), index=cov.columns
    )
    assert transfer_coefficient(alphas, active, cov) == pytest.approx(
        transfer_coefficient(alphas, 3.0 * active, cov)
    )


def test_a_book_that_fights_its_alphas_transfers_negatively(cov: pd.DataFrame):
    alphas = pd.Series(
        np.linspace(-0.02, 0.02, len(cov.columns)), index=cov.columns
    )
    optimal = pd.Series(np.linalg.pinv(cov.values) @ alphas.values, index=cov.columns)
    assert transfer_coefficient(alphas, -optimal, cov) == pytest.approx(-1.0, abs=1e-6)


def test_the_risk_adjusted_transfer_works_on_a_singular_covariance(
    returns: pd.DataFrame,
):
    """A detoned covariance has no inverse, and this definition needs none."""
    singular = covariance_matrix(returns, method="denoised", detone=1)
    alphas = pd.Series(
        np.linspace(-0.02, 0.02, len(singular.columns)), index=singular.columns
    )
    active = pd.Series(
        np.linspace(-0.05, 0.05, len(singular.columns)), index=singular.columns
    )
    value = transfer_coefficient(alphas, active, singular, method="risk_adjusted")
    assert -1.0 <= value <= 1.0


def test_transfer_coefficient_rejects_an_unknown_method(cov: pd.DataFrame):
    series = pd.Series(0.01, index=cov.columns)
    with pytest.raises(ValueError, match="Unknown transfer-coefficient method"):
        transfer_coefficient(series, series, cov, method="hopeful")


def test_holding_the_benchmark_leaves_nothing_to_transfer(cov: pd.DataFrame):
    alphas = pd.Series(
        np.linspace(-0.02, 0.02, len(cov.columns)), index=cov.columns
    )
    flat = pd.Series(0.0, index=cov.columns)
    with pytest.raises(ValueError, match="no active positions"):
        transfer_coefficient(alphas, flat, cov)


def test_a_flat_alpha_vector_leaves_nothing_to_transfer(cov: pd.DataFrame):
    """A constant expected-return vector carries no cross-sectional view.

    Returning NaN here would surface as a confusing failure inside
    ``fundamental_law`` several calls later; the modelling mistake is worth
    naming where it happens.
    """
    flat_alphas = pd.Series(0.05, index=cov.columns)
    active = pd.Series(
        np.linspace(-0.05, 0.05, len(cov.columns)), index=cov.columns
    )
    with pytest.raises(ValueError, match="same alpha"):
        transfer_coefficient(flat_alphas, active, cov)


# ---------------------------------------------------------------------------
# The fundamental law
# ---------------------------------------------------------------------------


def test_the_law_multiplies_out_as_advertised():
    report = fundamental_law(0.05, breadth=100, transfer_coefficient=0.4)
    assert report.unconstrained_information_ratio == pytest.approx(0.5)
    assert report.information_ratio == pytest.approx(0.2)
    assert report.constraint_cost == pytest.approx(0.3)


def test_breadth_and_implied_breadth_are_inverses():
    report = fundamental_law(0.04, breadth=625, transfer_coefficient=0.5)
    assert implied_breadth(
        report.information_ratio, 0.04, 0.5
    ) == pytest.approx(625.0)


def test_an_active_risk_budget_turns_the_ratio_into_a_return():
    report = fundamental_law(0.05, breadth=100, transfer_coefficient=1.0, active_risk=0.04)
    assert report.expected_active_return == pytest.approx(0.5 * 0.04)
    assert "active return" in report.describe()


def test_the_law_rejects_impossible_inputs():
    with pytest.raises(ValueError, match="at least 1"):
        fundamental_law(0.05, breadth=0.5)
    with pytest.raises(ValueError, match="correlation"):
        fundamental_law(0.05, breadth=100, transfer_coefficient=1.4)
    with pytest.raises(ValueError, match="zero skill"):
        implied_breadth(1.0, 0.0)


# ---------------------------------------------------------------------------
# Grinold's alpha and the risk-aversion calibration
# ---------------------------------------------------------------------------


def test_grinold_alpha_scales_scores_by_skill_and_volatility():
    scores = pd.Series([2.0, 0.0, -1.0], index=list("abc"))
    volatility = pd.Series([0.20, 0.20, 0.10], index=list("abc"))
    alpha = grinold_alpha(scores, volatility, 0.05)
    assert alpha["a"] == pytest.approx(0.05 * 0.20 * 2.0)
    assert alpha["b"] == pytest.approx(0.0)
    assert alpha["c"] == pytest.approx(-0.05 * 0.10)


def test_grinold_alpha_is_far_smaller_than_the_raw_score_suggests():
    """The discipline the formula exists to impose."""
    scores = pd.Series([2.0], index=["a"])
    alpha = grinold_alpha(scores, pd.Series([0.20], index=["a"]), 0.05)
    assert float(alpha.iloc[0]) == pytest.approx(0.02)


def test_risk_aversion_and_optimal_active_risk_round_trip():
    lam = risk_aversion_from_information_ratio(0.5, 0.04)
    assert lam == pytest.approx(6.25)
    assert optimal_active_risk(0.5, lam) == pytest.approx(0.04)


def test_value_added_peaks_at_the_optimal_active_risk():
    ir, lam = 0.5, 6.25
    best = optimal_active_risk(ir, lam)
    peak = value_added(ir, best, lam)
    for offset in (-0.01, 0.01):
        assert value_added(ir, best + offset, lam) < peak


def test_risk_aversion_rejects_a_zero_budget():
    with pytest.raises(ValueError, match="must be positive"):
        risk_aversion_from_information_ratio(0.5, 0.0)


# ---------------------------------------------------------------------------
# Active risk decomposition
# ---------------------------------------------------------------------------


def test_active_contributions_sum_to_the_tracking_error(cov: pd.DataFrame):
    benchmark = _equal_weights(cov)
    weights = benchmark.copy()
    weights.iloc[0] += 0.08
    weights.iloc[-1] -= 0.08

    frame = active_risk_decomposition(weights, benchmark, cov)
    active = (weights - benchmark).values
    tracking_error = float(np.sqrt(active @ cov.values @ active))
    assert frame["contribution"].sum() == pytest.approx(tracking_error)
    assert frame["share_of_tracking_error"].sum() == pytest.approx(1.0)


def test_holding_the_benchmark_produces_no_active_risk(cov: pd.DataFrame):
    benchmark = _equal_weights(cov)
    frame = active_risk_decomposition(benchmark, benchmark, cov)
    assert frame["contribution"].abs().sum() == pytest.approx(0.0)


def test_a_position_matching_the_benchmark_contributes_no_tracking_error(
    cov: pd.DataFrame,
):
    """The distinction between absolute and active risk, in one assertion."""
    benchmark = _equal_weights(cov)
    weights = benchmark.copy()
    weights.iloc[1] += 0.05
    weights.iloc[2] -= 0.05

    frame = active_risk_decomposition(weights, benchmark, cov)
    untouched = frame.index[0]
    assert frame.loc[untouched, "active_weight"] == pytest.approx(0.0)
    assert frame.loc[untouched, "contribution"] == pytest.approx(0.0)
    # ...even though it carries plenty of absolute risk.
    assert frame.loc[untouched, "weight"] > 0


# ---------------------------------------------------------------------------
# Effective number of bets
# ---------------------------------------------------------------------------


def test_torsion_factors_are_genuinely_uncorrelated(cov: pd.DataFrame):
    t = minimum_torsion(cov).values
    factor_cov = t @ cov.values @ t.T
    off_diagonal = factor_cov - np.diag(np.diag(factor_cov))
    scale = float(np.abs(np.diag(factor_cov)).max())
    assert float(np.abs(off_diagonal).max()) < 1e-10 * scale


def test_torsion_stays_close_to_the_original_assets(cov: pd.DataFrame):
    """That closeness is the whole point of *minimum* torsion."""
    t = minimum_torsion(cov)
    pca = pca_torsion(cov)
    identity = np.eye(len(cov.columns))
    assert np.linalg.norm(t.values - identity) < np.linalg.norm(pca.values - identity)


def test_uncorrelated_equal_bets_give_the_maximum_number_of_bets():
    assets = list("abcde")
    cov = pd.DataFrame(np.eye(5) * 0.04, index=assets, columns=assets)
    weights = pd.Series(0.2, index=assets)
    for model in ("minimum_torsion", "pca"):
        report = diversification_distribution(weights, cov, model=model)
        assert report.effective_number_of_bets == pytest.approx(5.0)
        assert report.concentration == pytest.approx(1.0)


def test_a_single_position_is_one_bet():
    assets = list("abcde")
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 5))
    cov = pd.DataFrame(np.cov(base, rowvar=False), index=assets, columns=assets)
    weights = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=assets)
    report = diversification_distribution(weights, cov, model="pca")
    assert report.effective_number_of_bets < 3.0


def test_pca_collapses_when_one_factor_dominates():
    assets = list("abcde")
    cov = pd.DataFrame(
        np.full((5, 5), 0.0396) + np.eye(5) * 0.0004, index=assets, columns=assets
    )
    weights = pd.Series(0.2, index=assets)
    pca = diversification_distribution(weights, cov, model="pca")
    assert pca.effective_number_of_bets == pytest.approx(1.0, abs=0.05)


def test_the_distribution_is_a_probability_distribution(cov: pd.DataFrame):
    weights = _equal_weights(cov)
    for model in ("minimum_torsion", "pca"):
        report = diversification_distribution(weights, cov, model=model)
        assert report.distribution.sum() == pytest.approx(1.0)
        assert (report.distribution >= 0).all()
        assert 1.0 <= report.effective_number_of_bets <= len(cov.columns) + 1e-9


def test_comparing_the_two_rotations_reports_both(cov: pd.DataFrame):
    frame = compare_diversification(_equal_weights(cov), cov)
    assert set(frame.index) == {"minimum_torsion", "pca"}
    assert (frame["effective_bets"] >= 1.0).all()


def test_diversification_rejects_an_unknown_rotation(cov: pd.DataFrame):
    with pytest.raises(ValueError, match="Unknown torsion model"):
        diversification_distribution(_equal_weights(cov), cov, model="magic")
