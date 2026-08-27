"""Deflated Sharpe, minimum track record length, and CSCV overfitting.

The central test here is the one that matters in practice: a family of
strategies with *no* skill at all, from which the best is selected. The
ordinary probabilistic Sharpe ratio declares the winner significant; the
deflated one must not.
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

from optimization_engine.analytics.performance import sharpe_ratio
from optimization_engine.analytics.selection import (
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
)


@pytest.fixture(scope="module")
def skill_free_trials() -> pd.DataFrame:
    """50 strategies drawn from the same zero-mean distribution."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0, 0.01, size=(1000, 50)),
        columns=[f"trial_{i}" for i in range(50)],
    )


# ---------------------------------------------------------------------------
# Expected maximum
# ---------------------------------------------------------------------------


def test_expected_maximum_grows_with_the_number_of_trials():
    values = [expected_maximum_sharpe(n, 1.0) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)


def test_a_single_trial_has_nothing_to_deflate():
    assert expected_maximum_sharpe(1, 1.0) == 0.0


def test_expected_maximum_scales_with_the_dispersion_of_trials():
    wide = expected_maximum_sharpe(100, 4.0)
    narrow = expected_maximum_sharpe(100, 1.0)
    assert wide == pytest.approx(2.0 * narrow)


def test_expected_maximum_matches_a_simulated_maximum():
    """The closed form should track a brute-force maximum of N normals."""
    rng = np.random.default_rng(4)
    n_trials = 200
    draws = rng.normal(0.0, 1.0, size=(5000, n_trials))
    simulated = float(draws.max(axis=1).mean())
    assert expected_maximum_sharpe(n_trials, 1.0) == pytest.approx(
        simulated, abs=0.05
    )


def test_expected_maximum_rejects_a_negative_variance():
    with pytest.raises(ValueError, match="non-negative"):
        expected_maximum_sharpe(10, -1.0)


# ---------------------------------------------------------------------------
# The deflated Sharpe ratio
# ---------------------------------------------------------------------------


def test_best_of_many_skill_free_trials_does_not_survive_deflation(
    skill_free_trials: pd.DataFrame,
):
    annual = skill_free_trials.apply(sharpe_ratio, periods_per_year=252)
    winner = skill_free_trials[annual.idxmax()]

    report = deflated_sharpe_ratio(
        winner, n_trials=skill_free_trials.shape[1], trial_sharpes=annual
    )
    # Undeflated, the winner looks convincing...
    assert report.probabilistic > 0.95
    # ...and once the search is accounted for, it is not.
    assert not report.is_significant
    assert report.deflated < report.probabilistic
    assert report.benchmark_sharpe > 0


def test_one_trial_deflates_to_the_plain_probabilistic_sharpe(
    skill_free_trials: pd.DataFrame,
):
    series = skill_free_trials.iloc[:, 0]
    report = deflated_sharpe_ratio(series, n_trials=1)
    assert report.deflated == pytest.approx(report.probabilistic)


def test_more_trials_never_raise_the_deflated_sharpe(
    skill_free_trials: pd.DataFrame,
):
    series = skill_free_trials.iloc[:, 0]
    values = [
        deflated_sharpe_ratio(series, n_trials=n).deflated
        for n in (1, 10, 100, 1000)
    ]
    assert values == sorted(values, reverse=True)


def test_a_genuinely_strong_strategy_still_survives_deflation():
    rng = np.random.default_rng(2)
    strong = pd.Series(rng.normal(0.0015, 0.005, size=2000))
    report = deflated_sharpe_ratio(strong, n_trials=50)
    assert report.is_significant


def test_deflation_needs_a_usable_sample():
    with pytest.raises(ValueError, match="at least 3 observations"):
        deflated_sharpe_ratio(pd.Series([0.01, -0.01]), n_trials=5)


def test_trial_sharpes_need_a_dispersion_to_be_useful():
    series = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    with pytest.raises(ValueError, match="at least 2 entries"):
        deflated_sharpe_ratio(series, n_trials=5, trial_sharpes=pd.Series([1.0]))


# ---------------------------------------------------------------------------
# Minimum track record length
# ---------------------------------------------------------------------------


def test_minimum_track_record_falls_as_the_sharpe_rises():
    rng = np.random.default_rng(6)
    weak = pd.Series(rng.normal(0.0002, 0.01, size=2000))
    strong = pd.Series(rng.normal(0.0020, 0.01, size=2000))
    assert minimum_track_record_length(strong) < minimum_track_record_length(weak)


def test_minimum_track_record_is_infinite_below_the_benchmark():
    rng = np.random.default_rng(7)
    series = pd.Series(rng.normal(0.0001, 0.01, size=500))
    assert minimum_track_record_length(series, benchmark_sharpe=3.0) == float("inf")


def test_minimum_track_record_rejects_an_impossible_confidence():
    series = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 500))
    with pytest.raises(ValueError, match="confidence"):
        minimum_track_record_length(series, confidence=1.5)


def test_negative_skew_lengthens_the_required_track_record():
    """Two series with the same Sharpe but different shapes are not equal.

    The one with the long left tail needs more history before its Sharpe can
    be believed — which is exactly the correction the PSR family exists for.
    """
    rng = np.random.default_rng(8)
    symmetric = rng.normal(0.0, 1.0, size=4000)
    skewed = -np.abs(rng.normal(0.0, 1.0, size=4000)) ** 1.5
    skewed = skewed - skewed.mean()

    def rescale(x: np.ndarray) -> pd.Series:
        return pd.Series(0.001 + 0.01 * (x - x.mean()) / x.std())

    plain = minimum_track_record_length(rescale(symmetric))
    long_tail = minimum_track_record_length(rescale(skewed))
    assert long_tail > plain


# ---------------------------------------------------------------------------
# Probability of backtest overfitting
# ---------------------------------------------------------------------------


def test_selecting_among_skill_free_strategies_is_close_to_a_coin_flip(
    skill_free_trials: pd.DataFrame,
):
    report = probability_of_backtest_overfitting(
        skill_free_trials.iloc[:, :12], n_partitions=8
    )
    assert report.n_splits == 70  # C(8, 4)
    assert 0.2 < report.pbo < 0.8
    # And in-sample ranking should carry no useful information.
    assert report.performance_degradation < 0.5


def test_a_genuinely_superior_strategy_is_selected_reliably():
    rng = np.random.default_rng(12)
    noise = rng.normal(0.0, 0.01, size=(1200, 9))
    winner = rng.normal(0.0025, 0.01, size=(1200, 1))
    frame = pd.DataFrame(
        np.hstack([winner, noise]),
        columns=["real"] + [f"noise_{i}" for i in range(9)],
    )
    report = probability_of_backtest_overfitting(frame, n_partitions=8)
    assert report.pbo < 0.1
    assert report.probability_of_loss < 0.2


def test_cscv_requires_an_even_partition_count(skill_free_trials: pd.DataFrame):
    with pytest.raises(ValueError, match="even number"):
        probability_of_backtest_overfitting(skill_free_trials, n_partitions=7)


def test_cscv_requires_something_to_choose_between(skill_free_trials: pd.DataFrame):
    with pytest.raises(ValueError, match="at least 2 columns"):
        probability_of_backtest_overfitting(
            skill_free_trials.iloc[:, :1], n_partitions=8
        )


def test_cscv_requires_enough_history_to_cut(skill_free_trials: pd.DataFrame):
    with pytest.raises(ValueError, match="observations"):
        probability_of_backtest_overfitting(
            skill_free_trials.iloc[:10], n_partitions=8
        )
