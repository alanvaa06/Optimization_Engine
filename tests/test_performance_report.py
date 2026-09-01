"""The one performance object: absolute and relative, on one aligned sample."""

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

from optimization_engine.analytics.performance import (
    gain_to_pain_ratio,
    martin_ratio,
    summary_stats,
    time_under_water,
    win_loss_ratio,
)
from optimization_engine.analytics.relative import (
    appraisal_ratio,
    batting_average,
    m_squared,
    relative_drawdown,
    summary_relative,
    treynor_ratio,
    up_down_number_ratio,
)
from optimization_engine.analytics.report import (
    BENCHMARK,
    EXCESS,
    PORTFOLIO,
    compare_performance,
    performance_report,
    period_returns,
    rolling_relative,
)
from optimization_engine.benchmark import BenchmarkSpec
from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 5, seed=17))


@pytest.fixture(scope="module")
def portfolio(returns) -> pd.Series:
    return returns.iloc[:, :4].mean(axis=1)


@pytest.fixture(scope="module")
def benchmark(returns) -> pd.Series:
    return returns.mean(axis=1)


@pytest.fixture(scope="module")
def report(portfolio, benchmark):
    return performance_report(
        portfolio, benchmark, riskfree_rate=0.03, benchmark_label="Equal weight"
    )


# ---------------------------------------------------------------------------
# New absolute metrics
# ---------------------------------------------------------------------------


def test_gain_to_pain_is_one_when_gains_match_losses():
    series = pd.Series([0.10, -0.05, -0.05, 0.10, -0.10])
    assert gain_to_pain_ratio(series) == pytest.approx(0.0)
    winner = pd.Series([0.10, -0.05])
    assert winner.sum() / 0.05 == pytest.approx(gain_to_pain_ratio(winner))


def test_gain_to_pain_is_infinite_without_losses():
    assert gain_to_pain_ratio(pd.Series([0.01, 0.02])) == float("inf")


def test_win_loss_ratio_compares_average_sizes():
    series = pd.Series([0.04, 0.02, -0.01, -0.03])
    assert win_loss_ratio(series) == pytest.approx(0.03 / 0.02)


def test_win_loss_ratio_is_zero_when_nothing_wins():
    assert win_loss_ratio(pd.Series([-0.01, -0.02])) == 0.0


def test_time_under_water_is_zero_for_a_monotone_series():
    assert time_under_water(pd.Series([0.01] * 10)) == pytest.approx(0.0)


def test_time_under_water_counts_the_periods_below_the_peak():
    # Up, down, then recovering: three of the four periods sit below the peak.
    series = pd.Series([0.10, -0.05, 0.01, 0.01])
    assert 0.0 < time_under_water(series) <= 1.0


def test_martin_ratio_prices_the_duration_of_the_pain():
    # Both rise, fall by the same amount, and recover to the same place. The
    # slow one spends far longer below its high-water mark, so its Ulcer
    # index is larger and its Martin ratio lower — which is the whole point
    # of preferring it to Calmar, where the two would score identically.
    quick = pd.Series([0.05] * 6 + [-0.10] + [0.115] + [0.0] * 12)
    slow = pd.Series([0.05] * 6 + [-0.10] + [0.0] * 11 + [0.115])
    assert quick.sum() == pytest.approx(slow.sum())
    assert martin_ratio(quick, periods_per_year=12) > martin_ratio(
        slow, periods_per_year=12
    )
    assert time_under_water(quick) < time_under_water(slow)


def test_extended_summary_carries_the_new_columns(portfolio):
    stats = summary_stats(portfolio.to_frame("p"), extended=True)
    for column in (
        "Martin Ratio", "Gain-to-Pain", "Win/Loss Ratio",
        "Time Under Water", "Best Period", "Worst Period",
    ):
        assert column in stats.columns
    assert stats.loc["p", "Best Period"] == pytest.approx(portfolio.max())
    assert stats.loc["p", "Worst Period"] == pytest.approx(portfolio.min())


# ---------------------------------------------------------------------------
# New relative metrics
# ---------------------------------------------------------------------------


def test_batting_average_of_a_portfolio_against_itself_is_zero(portfolio):
    value = float(batting_average(portfolio.to_frame("p"), portfolio).iloc[0])
    assert value == pytest.approx(0.0)


def test_batting_average_counts_the_winning_periods(portfolio, benchmark):
    value = float(batting_average(portfolio.to_frame("p"), benchmark).iloc[0])
    expected = float(((portfolio - benchmark) > 0).mean())
    assert value == pytest.approx(expected)


def test_m_squared_ranks_like_the_sharpe_ratio(returns, benchmark):
    frame = returns.iloc[:, :3]
    m2 = m_squared(frame, benchmark, riskfree_rate=0.02)
    sharpe = summary_stats(frame, riskfree_rate=0.02)["Sharpe Ratio"]
    assert list(m2.sort_values().index) == list(sharpe.sort_values().index)


def test_treynor_uses_systematic_risk(portfolio, benchmark):
    value = float(treynor_ratio(portfolio.to_frame("p"), benchmark).iloc[0])
    assert np.isfinite(value)


def test_appraisal_ratio_is_alpha_over_residual_risk(portfolio, benchmark):
    stats = summary_relative(
        portfolio.to_frame("p"), benchmark, extended=True
    )
    ratio = float(appraisal_ratio(portfolio.to_frame("p"), benchmark).iloc[0])
    expected = (
        stats.loc["p", "Alpha (annualized)"] / stats.loc["p", "Residual Vol"]
    )
    assert ratio == pytest.approx(float(expected))


def test_relative_drawdown_is_zero_against_itself(portfolio):
    dd = relative_drawdown(portfolio.to_frame("p"), portfolio)
    assert float(dd["p"].abs().max()) == pytest.approx(0.0, abs=1e-9)


def test_relative_drawdown_is_never_positive(portfolio, benchmark):
    dd = relative_drawdown(portfolio.to_frame("p"), benchmark)
    assert float(dd["p"].max()) <= 1e-12
    assert float(dd["p"].min()) < 0.0


def test_up_down_number_ratios_are_shares(portfolio, benchmark):
    frame = up_down_number_ratio(portfolio.to_frame("p"), benchmark)
    for column in ("Up Number Ratio", "Down Number Ratio"):
        assert 0.0 <= float(frame.loc["p", column]) <= 1.0


def test_extended_relative_summary_carries_the_second_tier(portfolio, benchmark):
    stats = summary_relative(portfolio.to_frame("p"), benchmark, extended=True)
    for column in (
        "Batting Average", "Treynor Ratio", "M-squared", "Appraisal Ratio",
        "Correlation", "Downside T.E.", "Worst Relative Drawdown",
        "Prob. Excess > 0", "Up Number Ratio", "Down Number Ratio",
    ):
        assert column in stats.columns


# ---------------------------------------------------------------------------
# Calendar periods and rolling frames
# ---------------------------------------------------------------------------


def test_period_returns_compound_within_the_period(portfolio):
    yearly = period_returns(portfolio.to_frame(PORTFOLIO), "yearly")
    first_year = portfolio[portfolio.index.year == portfolio.index[0].year]
    expected = float((1 + first_year).prod() - 1)
    assert float(yearly.iloc[0, 0]) == pytest.approx(expected)


def test_period_excess_is_the_difference_of_the_period_returns(report):
    periods = report.periods
    difference = periods[PORTFOLIO] - periods[BENCHMARK]
    pd.testing.assert_series_equal(
        periods[EXCESS], difference, check_names=False
    )


@pytest.mark.parametrize("freq", ["yearly", "quarterly", "monthly"])
def test_every_frequency_produces_a_table(portfolio, freq):
    table = period_returns(portfolio, freq)
    assert not table.empty


def test_unknown_frequency_is_rejected(portfolio):
    with pytest.raises(ValueError, match="Unknown period"):
        period_returns(portfolio, "fortnightly")


def test_period_returns_need_dates():
    series = pd.Series([0.01, 0.02, 0.03])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        period_returns(series)


def test_rolling_relative_against_itself_is_flat(portfolio):
    frame = rolling_relative(portfolio, portfolio, window=60)
    frame = frame.dropna(subset=["rolling_excess", "rolling_beta"])
    assert not frame.empty
    assert float(frame["rolling_excess"].abs().max()) == pytest.approx(0.0, abs=1e-12)
    assert float(frame["rolling_beta"].max()) == pytest.approx(1.0, abs=1e-8)
    # Zero tracking error leaves the information ratio genuinely undefined
    # rather than infinite, and the frame says so instead of guessing.
    assert frame["rolling_information_ratio"].isna().all()


def test_rolling_window_must_be_usable(portfolio, benchmark):
    with pytest.raises(ValueError, match="at least 2"):
        rolling_relative(portfolio, benchmark, window=1)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_report_holds_both_halves(report):
    assert report.has_benchmark
    assert PORTFOLIO in report.absolute.index
    assert BENCHMARK in report.absolute.index
    assert list(report.relative.index) == [PORTFOLIO]
    assert EXCESS in report.returns.columns


def test_absolute_and_relative_cover_the_same_sample(portfolio, benchmark):
    # A benchmark that starts late must shorten the absolute numbers too,
    # otherwise the two halves describe different periods.
    late = benchmark.iloc[200:]
    report = performance_report(portfolio, late)
    assert len(report.returns) == len(late)
    assert report.returns.index.min() == late.index.min()


def test_report_without_a_benchmark_is_absolute_only(portfolio):
    report = performance_report(portfolio)
    assert not report.has_benchmark
    assert report.relative is None
    assert report.rolling_relative_frame is None
    assert report.relative_drawdown_series is None
    assert "performance_relative" not in report.to_frames()
    assert "returned" in report.describe()


def test_headline_carries_absolute_and_relative(report):
    head = report.headline()
    assert head["sharpe_ratio"] == pytest.approx(
        float(report.absolute.loc[PORTFOLIO, "Sharpe Ratio"])
    )
    assert head["information_ratio"] == pytest.approx(
        float(report.relative.loc[PORTFOLIO, "Information Ratio"])
    )


def test_active_share_appears_only_with_both_weight_vectors(portfolio, benchmark, returns):
    weights = pd.Series(0.25, index=returns.columns[:4])
    bench_weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    with_weights = performance_report(
        portfolio,
        benchmark,
        portfolio_weights=weights,
        benchmark_weights=bench_weights,
    )
    assert with_weights.active_share is not None
    assert "active_share" in with_weights.headline()
    without = performance_report(portfolio, benchmark)
    assert without.active_share is None
    assert "active_share" not in without.headline()


def test_metrics_are_tidy_and_complete(report):
    tidy = report.metrics()
    assert list(tidy.columns) == ["block", "series", "metric", "value"]
    assert set(tidy["block"]) == {"Absolute", "Relative"}
    absolute_rows = tidy[
        (tidy.block == "Absolute") & (tidy.series == PORTFOLIO)
    ]
    assert len(absolute_rows) == report.absolute.shape[1]


def test_frames_are_named_for_a_workbook(report):
    frames = report.to_frames()
    assert "performance_absolute" in frames
    assert "performance_relative" in frames
    assert "performance_periods" in frames
    # Excel refuses sheet names past 31 characters.
    assert all(len(name) <= 31 for name in frames)
    assert all(isinstance(f, pd.DataFrame) for f in frames.values())


def test_describe_hedges_an_insignificant_alpha(portfolio, benchmark):
    text = performance_report(
        portfolio, benchmark, benchmark_label="EW"
    ).describe()
    assert "EW" in text
    assert "t-statistic" in text or "significant" in text


def test_an_empty_portfolio_is_rejected():
    with pytest.raises(ValueError, match="empty portfolio"):
        performance_report(pd.Series(dtype=float))


def test_a_disjoint_benchmark_is_rejected(portfolio):
    shifted = pd.Series(
        portfolio.values, index=portfolio.index + pd.DateOffset(years=30)
    )
    with pytest.raises(ValueError, match="share no dates"):
        performance_report(portfolio, shifted)


def test_compare_performance_lines_up_several_streams(returns, benchmark):
    frame = compare_performance(
        {"a": returns.iloc[:, 0], "b": returns.iloc[:, 1]}, benchmark
    )
    assert list(frame.index) == ["a", "b"]
    assert "Sharpe Ratio" in frame.columns
    assert "Information Ratio" in frame.columns


def test_compare_performance_needs_something_to_compare():
    with pytest.raises(ValueError, match="at least one"):
        compare_performance({})


# ---------------------------------------------------------------------------
# Wired to a run
# ---------------------------------------------------------------------------


def test_a_run_reports_against_its_own_benchmark(returns):
    cfg = EngineConfig(
        optimizer=OptimizerSpec(name="mean_variance", risk_aversion=3.0),
        benchmark=BenchmarkSpec(kind="equal_weight"),
    )
    run = run_engine(returns, cfg)
    report = run.performance(riskfree_rate=0.02)
    assert report.has_benchmark
    assert report.benchmark_label == "Equal weight (1/N)"
    assert report.active_share is not None
    assert report.metadata["rebalancing"] == "monthly"
    assert report.metadata["out_of_sample"] is False


def test_a_run_without_a_benchmark_still_reports(returns):
    cfg = EngineConfig(optimizer=OptimizerSpec(name="min_variance"))
    run = run_engine(returns, cfg)
    report = run.performance()
    assert not report.has_benchmark
    assert np.isfinite(report.headline()["sharpe_ratio"])


def test_an_override_stream_marks_the_report_out_of_sample(returns):
    cfg = EngineConfig(
        optimizer=OptimizerSpec(name="min_variance"),
        benchmark=BenchmarkSpec(kind="equal_weight"),
    )
    run = run_engine(returns, cfg)
    oos = returns.iloc[-250:].mean(axis=1)
    report = run.performance(returns_override=oos)
    assert report.metadata["out_of_sample"] is True
    assert len(report.returns) == len(oos)


# ---------------------------------------------------------------------------
# Streams of unequal length (review fix B4)
# ---------------------------------------------------------------------------


def test_comparisons_score_streams_over_their_common_window(returns):
    from optimization_engine.analytics.backtest import compare_in_and_out_of_sample
    from optimization_engine.analytics.performance import summary_stats

    fitted = returns.iloc[:, :5].mean(axis=1)
    walk_forward = returns.iloc[504:, :3].mean(axis=1) - 0.0004  # starts later, does worse

    table = compare_in_and_out_of_sample(fitted, walk_forward, 252, 0.03)
    assert not table.isna().any().any(), table[table.isna().any(axis=1)]

    alone = summary_stats(walk_forward.to_frame("x"), periods_per_year=252, riskfree_rate=0.03).T["x"]
    pd.testing.assert_series_equal(
        table["Out-of-sample (walk-forward)"], alone, check_names=False
    )
    fitted_on_window = summary_stats(
        fitted.reindex(walk_forward.index).to_frame("x"), periods_per_year=252, riskfree_rate=0.03
    ).T["x"]
    pd.testing.assert_series_equal(table["In-sample (fitted)"], fitted_on_window, check_names=False)

    side_by_side = compare_performance(
        {"fitted": fitted, "walk-forward": walk_forward}, returns.mean(axis=1), riskfree_rate=0.03
    )
    assert not side_by_side.isna().any().any()
    assert side_by_side.loc["walk-forward", "Annualized Return"] == pytest.approx(
        alone["Annualized Return"]
    )


def test_a_nan_period_is_neither_a_return_nor_a_loss(returns):
    from optimization_engine.analytics.performance import annualize_returns, hit_rate
    from optimization_engine.analytics.risk import cvar_historic, var_historic

    stream = returns.iloc[:, 0]
    padded = pd.concat([pd.Series(np.nan, index=returns.index[:300]), stream.iloc[300:]])
    alone = stream.iloc[300:]

    assert annualize_returns(padded, 252) == pytest.approx(annualize_returns(alone, 252))
    assert hit_rate(padded) == pytest.approx(hit_rate(alone))
    assert var_historic(padded) == pytest.approx(var_historic(alone))
    assert cvar_historic(padded) == pytest.approx(cvar_historic(alone))
    frame = pd.DataFrame({"padded": padded, "alone": alone})
    assert annualize_returns(frame, 252)["padded"] == pytest.approx(annualize_returns(alone, 252))
