"""Backtesting: drift, rebalancing, costs, and out-of-sample walk-forward."""

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

from optimization_engine.analytics.backtest import (
    backtest_weights,
    compare_in_and_out_of_sample,
    rebalance_dates,
    walk_forward_backtest,
)
from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.data.quality import align_panel, analyze_prices
from optimization_engine.engine import run_engine


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 6, seed=3))


def test_rebalance_dates_always_include_inception(returns):
    for freq in ("none", "monthly", "quarterly", "annual"):
        dates = rebalance_dates(returns.index, freq)
        assert dates[0] == returns.index[0]
    assert len(rebalance_dates(returns.index, "none")) == 1
    assert len(rebalance_dates(returns.index, "monthly")) > 12


def test_unknown_frequency_is_rejected(returns):
    with pytest.raises(ValueError, match="Unknown rebalance frequency"):
        rebalance_dates(returns.index, "fortnightly")


def test_buy_and_hold_lets_weights_drift(returns):
    w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    held = backtest_weights(returns, w, frequency="none")
    final = held.weights.iloc[-1]
    assert float(final.std()) > 1e-4, "buy-and-hold weights should have drifted"
    # Only the initial purchase is a trade.
    assert len(held.turnover) == 1


def test_daily_rebalancing_pins_weights_to_target(returns):
    w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    pinned = backtest_weights(returns, w, frequency="daily")
    np.testing.assert_allclose(
        pinned.weights.iloc[-1].values, w.values, atol=1e-12
    )


def test_transaction_costs_reduce_returns_and_are_accounted(returns):
    w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    free = backtest_weights(returns, w, frequency="monthly", transaction_cost_bps=0)
    costed = backtest_weights(returns, w, frequency="monthly", transaction_cost_bps=25)
    assert costed.returns.sum() < free.returns.sum()
    assert costed.total_cost > 0
    assert costed.cost_drag() > 0
    # Cost equals traded notional times the rate, to the basis point.
    assert costed.total_cost == pytest.approx(
        costed.total_turnover * 25 / 10_000, rel=1e-9
    )


def test_more_frequent_rebalancing_trades_more(returns):
    w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    annual = backtest_weights(returns, w, frequency="annual")
    monthly = backtest_weights(returns, w, frequency="monthly")
    assert monthly.total_turnover > annual.total_turnover


def test_gross_and_net_agree_without_costs(returns):
    w = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    bt = backtest_weights(returns, w, frequency="monthly", transaction_cost_bps=0)
    pd.testing.assert_series_equal(
        bt.returns.rename("x"), bt.gross_returns.rename("x")
    )


def test_backtest_rejects_empty_returns():
    with pytest.raises(ValueError, match="empty returns"):
        backtest_weights(pd.DataFrame(), pd.Series(dtype=float))


def _solver(bounds_cap: float = 0.30):
    def solve(window: pd.DataFrame) -> pd.Series:
        cfg = EngineConfig(
            bounds={a: [0.0, bounds_cap] for a in window.columns},
            optimizer=OptimizerSpec(name="min_variance"),
        )
        return run_engine(window, cfg, check_feasibility=False).result.weights

    return solve


def test_walk_forward_never_uses_future_data(returns):
    """Each solve must see only returns strictly before its decision date."""
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def solve(window: pd.DataFrame) -> pd.Series:
        seen.append((window.index[0], window.index[-1]))
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    wf = walk_forward_backtest(
        returns, solve, lookback=252, rebalance_every=63
    )
    for (_, window_end), decision in zip(seen, wf.weights_history.index):
        assert window_end < decision


def test_walk_forward_is_flagged_out_of_sample(returns):
    wf = walk_forward_backtest(
        returns, _solver(), lookback=504, rebalance_every=63
    )
    assert wf.backtest.is_out_of_sample
    assert wf.n_rebalances > 5
    assert not wf.failures


def test_walk_forward_degrades_versus_in_sample(returns):
    """The in-sample Sharpe of a fitted portfolio should beat its own OOS run."""
    cfg = EngineConfig(
        bounds={a: [0.0, 0.30] for a in returns.columns},
        optimizer=OptimizerSpec(name="max_sharpe", risk_free_rate=0.02),
    )
    fitted = run_engine(returns, cfg)

    def solve(window: pd.DataFrame) -> pd.Series:
        return run_engine(window, cfg, check_feasibility=False).result.weights

    wf = walk_forward_backtest(returns, solve, lookback=504, rebalance_every=63)
    comparison = compare_in_and_out_of_sample(
        fitted.backtest_returns()["portfolio"].reindex(wf.returns.index),
        wf.returns,
        periods_per_year=252,
    )
    assert "Degradation" in comparison.columns
    assert comparison.loc["Sharpe Ratio", "Degradation"] > 0


def test_walk_forward_carries_weights_forward_when_a_solve_fails(returns):
    calls = {"n": 0}

    def flaky(window: pd.DataFrame) -> pd.Series:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("solver blew up")
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    wf = walk_forward_backtest(
        returns, flaky, lookback=252, rebalance_every=126
    )
    assert len(wf.failures) == 1
    assert "solver blew up" in wf.failures[0]
    # The failed date still holds a book — the previous one.
    assert wf.n_rebalances == len(wf.windows)


def test_walk_forward_requires_enough_history(returns):
    with pytest.raises(ValueError, match="out of sample"):
        walk_forward_backtest(
            returns.iloc[:100], _solver(), lookback=252, rebalance_every=21
        )


def test_walk_forward_rejects_degenerate_parameters(returns):
    with pytest.raises(ValueError, match="lookback"):
        walk_forward_backtest(returns, _solver(), lookback=1, rebalance_every=21)
    with pytest.raises(ValueError, match="rebalance_every"):
        walk_forward_backtest(returns, _solver(), lookback=252, rebalance_every=0)


def test_weight_stability_detects_a_churning_optimizer(returns):
    rng = np.random.default_rng(0)

    def churn(window: pd.DataFrame) -> pd.Series:
        w = rng.random(window.shape[1])
        return pd.Series(w / w.sum(), index=window.columns)

    steady = walk_forward_backtest(
        returns,
        lambda w: pd.Series(1.0 / w.shape[1], index=w.columns),
        lookback=252, rebalance_every=63,
    )
    noisy = walk_forward_backtest(
        returns, churn, lookback=252, rebalance_every=63
    )
    assert noisy.weight_stability().mean() > steady.weight_stability().mean()


def test_engine_walk_forward_helper_round_trips(returns):
    cfg = EngineConfig(
        bounds={a: [0.0, 0.30] for a in returns.columns},
        optimizer=OptimizerSpec(name="min_variance"),
    )
    run = run_engine(returns, cfg)
    wf = run.walk_forward(lookback=504, rebalance_every=126, transaction_cost_bps=10)
    assert wf.backtest.is_out_of_sample
    table = run.in_vs_out_of_sample(wf)
    assert {"In-sample (fitted)", "Out-of-sample (walk-forward)"} <= set(table.columns)


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------


def test_clean_panel_reports_no_errors():
    report = analyze_prices(sample_dataset(252 * 3))
    assert report.is_usable
    assert not report.errors


def test_interior_gaps_are_flagged():
    prices = sample_dataset(252 * 2).copy()
    prices.iloc[50:60, 0] = np.nan
    report = analyze_prices(prices)
    assert any(i.code == "interior_gaps" for i in report.issues)


def test_stale_prices_are_flagged():
    prices = sample_dataset(252 * 2).copy()
    prices.iloc[30:45, 1] = prices.iloc[29, 1]
    report = analyze_prices(prices)
    assert any(i.code == "stale_prices" for i in report.issues)


def test_short_common_history_is_flagged():
    prices = sample_dataset(252 * 3).copy()
    prices.iloc[:600, 2] = np.nan
    report = analyze_prices(prices)
    assert any(i.code == "short_common_history" for i in report.issues)
    assert report.n_common_periods < len(prices)


def test_thin_sample_is_flagged():
    prices = sample_dataset(60)
    report = analyze_prices(prices, min_observations_per_asset=10)
    assert any(i.code in ("thin_sample", "singular_sample") for i in report.issues)


def test_non_positive_prices_are_an_error():
    prices = sample_dataset(252).copy()
    prices.iloc[10, 0] = 0.0
    report = analyze_prices(prices)
    assert any(i.code == "non_positive_price" for i in report.errors)
    assert not report.is_usable


def test_align_panel_explains_what_it_did():
    prices = sample_dataset(252 * 2).copy()
    prices.iloc[50:53, 0] = np.nan
    aligned, actions = align_panel(prices, method="ffill")
    assert not aligned.isna().any().any()
    assert actions and any("Forward-filled" in a for a in actions)


def test_align_panel_can_drop_short_assets():
    prices = sample_dataset(252 * 2).copy()
    prices.iloc[:-10, 3] = np.nan
    aligned, actions = align_panel(prices, method="drop_assets", min_observations=100)
    assert prices.columns[3] not in aligned.columns
    assert any("Dropped" in a for a in actions)


def test_overlap_matrix_is_symmetric_with_counts_on_the_diagonal():
    prices = sample_dataset(252).copy()
    prices.iloc[:100, 0] = np.nan
    report = analyze_prices(prices)
    assert (report.overlap.values == report.overlap.values.T).all()
    assert report.overlap.iloc[0, 0] == 152
