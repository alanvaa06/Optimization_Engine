"""The simulation stack: spec, costs, execution, provenance, and the layers on top.

These cover the pieces that the compact ``analytics.backtest`` API cannot
express — execution lag, a cost model with market impact, the trade and cost
frames, the grid runner and its trial count, and the holdout audit log.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.backtest import (
    BacktestSpec,
    CostSpec,
    LinearCost,
    MarketContext,
    SquareRootImpactCost,
    SweepSpec,
    assert_within_holdout,
    build_cost_model,
    build_tearsheet,
    compute_position_stats,
    compute_tca,
    cost_by_asset,
    final_holdout_run,
    gate_returns,
    holdout_segment,
    read_audit_log,
    run_backtest,
    run_sweep,
    trailing_volatilities,
    walk_forward_run,
)
from optimization_engine.backtest.holdout import (
    REPEATED,
    SHIFTED_HOLDOUT,
    HoldoutViolationError,
)
from optimization_engine.backtest.spec import SpecValidationError
from optimization_engine.backtest.sweep import (
    HARD_CELL_CAP,
    SweepValidationError,
    expand_grid,
)
from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.loader import prices_to_returns, sample_dataset


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=11))


@pytest.fixture(scope="module")
def equal_weights(returns) -> pd.Series:
    return pd.Series(1.0 / returns.shape[1], index=returns.columns)


# -- the spec ---------------------------------------------------------------


def test_spec_rejects_what_it_cannot_simulate():
    with pytest.raises(SpecValidationError, match="Unknown rebalance frequency"):
        BacktestSpec(frequency="fortnightly")
    with pytest.raises(SpecValidationError, match="cannot be negative"):
        BacktestSpec(execution_lag=-1)
    with pytest.raises(SpecValidationError, match="initial_capital"):
        BacktestSpec(initial_capital=0.0)
    with pytest.raises(SpecValidationError, match="must be non-negative"):
        CostSpec(commission_bps=-1.0)
    with pytest.raises(SpecValidationError, match="impact_participation"):
        CostSpec(impact_coefficient=0.1, impact_participation=0.0)


def test_spec_hash_tracks_meaning_not_labels():
    base = BacktestSpec(frequency="monthly", costs=CostSpec(commission_bps=10))
    renamed = base.with_(name="a different label entirely")
    assert renamed.spec_hash == base.spec_hash

    costlier = base.with_(costs=CostSpec(commission_bps=11))
    assert costlier.spec_hash != base.spec_hash
    assert BacktestSpec.from_dict(base.to_dict()).spec_hash == base.spec_hash


# -- cost models ------------------------------------------------------------


def test_linear_cost_is_proportional_to_size():
    model = LinearCost(commission_bps=10.0, slippage_bps=5.0)
    small = model.charge(asset="A", traded_weight=0.01, context=MarketContext())
    large = model.charge(asset="A", traded_weight=0.10, context=MarketContext())
    assert large.total == pytest.approx(10 * small.total)
    assert small.commission == pytest.approx(0.01 * 10 / 10_000)
    assert small.slippage == pytest.approx(0.01 * 5 / 10_000)


def test_impact_makes_a_big_trade_cost_more_per_unit():
    model = SquareRootImpactCost(eta=1.0, participation=0.05, slippage_bps=0.0)
    context = MarketContext(volatility=0.01)
    small = model.charge(asset="A", traded_weight=0.01, context=context)
    large = model.charge(asset="A", traded_weight=0.09, context=context)
    per_unit_small = small.total / 0.01
    per_unit_large = large.total / 0.09
    assert per_unit_large > per_unit_small
    # Cost per unit scales as sqrt(size): nine times the trade, three times the rate.
    assert per_unit_large / per_unit_small == pytest.approx(3.0, rel=1e-9)


def test_impact_degrades_loudly_when_volatility_is_unknown():
    model = SquareRootImpactCost(eta=1.0, participation=0.05, commission_bps=5.0)
    quote = model.charge(asset="AAA", traded_weight=0.1, context=MarketContext(None))
    assert quote.slippage == 0.0
    assert quote.commission > 0.0
    assert "AAA" in quote.degraded_reason
    assert "volatility" in quote.degraded_reason


def test_trailing_volatility_never_sees_its_own_period(returns):
    vol = trailing_volatilities(returns, lookback=20, min_observations=20)
    assert vol.iloc[:20].isna().all().all(), "no estimate before the window fills"
    asset = returns.columns[0]
    manual = returns[asset].iloc[:20].std(ddof=1)
    assert vol[asset].iloc[20] == pytest.approx(manual)


def test_build_cost_model_picks_the_cheapest_representation():
    from optimization_engine.backtest.costs import ZeroCost

    assert isinstance(build_cost_model(CostSpec()), ZeroCost)
    assert isinstance(build_cost_model(CostSpec(commission_bps=5)), LinearCost)
    assert isinstance(
        build_cost_model(CostSpec(impact_coefficient=0.5)), SquareRootImpactCost
    )


# -- the runner -------------------------------------------------------------


def test_run_is_reproducible_and_stamped(returns, equal_weights):
    spec = BacktestSpec(frequency="monthly", costs=CostSpec(commission_bps=10))
    first = run_backtest(returns, equal_weights, spec)
    second = run_backtest(returns, equal_weights, spec)
    assert first.meta.result_hash == second.meta.result_hash
    assert first.meta.spec_hash == spec.spec_hash
    assert first.meta.n_periods == len(returns)
    assert first.meta.n_assets == returns.shape[1]

    costlier = run_backtest(
        returns, equal_weights, spec.with_(costs=CostSpec(commission_bps=25))
    )
    assert costlier.meta.result_hash != first.meta.result_hash


def test_execution_lag_delays_the_fill(returns, equal_weights):
    immediate = run_backtest(
        returns, equal_weights, BacktestSpec(frequency="none", execution_lag=0)
    )
    delayed = run_backtest(
        returns, equal_weights, BacktestSpec(frequency="none", execution_lag=3)
    )
    # The book is in cash until the order fills, so the first periods earn nothing.
    assert (delayed.weights.iloc[:3] == 0.0).all().all()
    assert delayed.returns.iloc[:3].eq(0.0).all()
    assert immediate.returns.iloc[0] != 0.0
    assert delayed.rebalance_dates[0] == returns.index[3]


def test_a_lag_past_the_end_of_the_sample_never_fills(returns, equal_weights):
    spec = BacktestSpec(frequency="none", execution_lag=len(returns) + 5)
    run = run_backtest(returns, equal_weights, spec)
    assert run.trades.empty
    assert run.total_turnover == 0.0
    assert run.returns.eq(0.0).all()


def test_trades_frame_splits_the_cost_and_sides(returns, equal_weights):
    spec = BacktestSpec(
        frequency="quarterly", costs=CostSpec(commission_bps=8, slippage_bps=4)
    )
    run = run_backtest(returns, equal_weights, spec)
    assert set(run.trades["side"]) <= {"buy", "sell"}
    assert (run.trades["commission"] >= 0).all()
    # The commission and slippage rates are in the ratio they were configured in.
    assert run.trades["slippage"].sum() / run.trades["commission"].sum() == pytest.approx(
        4 / 8, rel=1e-9
    )
    # Per-date totals agree with the per-trade rows.
    assert run.costs["total"].sum() == pytest.approx(run.trades["cost"].sum())
    assert run.total_cost == pytest.approx(run.trades["cost"].sum())


def test_nav_follows_the_net_return_stream(returns, equal_weights):
    spec = BacktestSpec(
        frequency="monthly", costs=CostSpec(commission_bps=20), initial_capital=1_000_000
    )
    run = run_backtest(returns, equal_weights, spec)
    expected = 1_000_000 * (1 + run.returns).cumprod()
    pd.testing.assert_series_equal(run.nav, expected, check_names=False)
    # Notional is quoted against the NAV the trade actually met.
    first = run.trades.iloc[0]
    assert first["notional"] == pytest.approx(abs(first["traded_weight"]) * 1_000_000)


def test_impact_costs_scale_with_the_size_of_the_book(returns, equal_weights):
    thin = BacktestSpec(
        frequency="monthly",
        costs=CostSpec(impact_coefficient=0.5, impact_participation=0.005),
    )
    deep = thin.with_(
        costs=CostSpec(impact_coefficient=0.5, impact_participation=0.20)
    )
    expensive = run_backtest(returns, equal_weights, thin)
    cheap = run_backtest(returns, equal_weights, deep)
    assert expensive.total_cost > cheap.total_cost
    assert expensive.total_turnover == pytest.approx(cheap.total_turnover)


def test_early_impact_trades_degrade_and_say_so(returns, equal_weights):
    spec = BacktestSpec(
        frequency="none",
        costs=CostSpec(
            impact_coefficient=0.5, impact_volatility_lookback=63, min_impact_observations=21
        ),
    )
    run = run_backtest(returns, equal_weights, spec)
    # The very first trade has no history behind it, so impact cannot be priced.
    assert run.meta.degradations
    assert "volatility" in run.meta.degradations[0]


# -- walk-forward -----------------------------------------------------------


def _mean_variance_config(returns: pd.DataFrame) -> EngineConfig:
    return EngineConfig(
        expected_returns={asset: 0.0 for asset in returns.columns},
        optimizer=OptimizerSpec(name="min_variance"),
        periods_per_year=252,
    )


def test_walk_forward_bundle_carries_windows_and_stability(returns):
    from optimization_engine.engine import run_engine

    config = _mean_variance_config(returns)
    config.expected_returns = {}

    def solve(window: pd.DataFrame) -> pd.Series:
        return run_engine(window, config, check_feasibility=False).result.weights

    walk = walk_forward_run(
        returns,
        solve,
        lookback=252,
        rebalance_every=63,
        spec=BacktestSpec(costs=CostSpec(commission_bps=10)),
    )
    assert walk.run.meta.is_out_of_sample
    assert walk.n_rebalances == len(walk.windows)
    assert (walk.windows["window_end"] < walk.windows["decision_date"]).all()
    assert walk.weight_stability().gt(0).any()
    assert walk.run.meta.notes["lookback"] == 252


def test_walk_forward_records_failures_as_rows(returns):
    calls = {"n": 0}

    def flaky(window: pd.DataFrame) -> pd.Series:
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("solver gave up")
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    walk = walk_forward_run(returns, flaky, lookback=252, rebalance_every=63)
    assert walk.n_failures > 0
    assert (walk.windows["status"] == "ok").sum() + walk.n_failures == len(walk.windows)
    assert "solver gave up" in walk.failures[0]


# -- transaction-cost analysis ---------------------------------------------


def test_tca_normalizes_cost_by_what_was_traded(returns, equal_weights):
    spec = BacktestSpec(
        frequency="monthly", costs=CostSpec(commission_bps=15, slippage_bps=5)
    )
    panel = compute_tca(run_backtest(returns, equal_weights, spec))
    assert panel.cost_bps_of_notional == pytest.approx(20.0, rel=1e-6)
    assert panel.commission_share == pytest.approx(0.75, rel=1e-6)
    assert panel.annualized_turnover > 0
    assert panel.total_cost == pytest.approx(panel.commission + panel.slippage)
    assert not panel.reasons
    assert "bps of notional" in panel.describe()


def test_tca_says_why_a_ratio_is_missing_rather_than_reporting_zero(returns, equal_weights):
    spec = BacktestSpec(frequency="none", execution_lag=len(returns) + 1)
    panel = compute_tca(run_backtest(returns, equal_weights, spec))
    assert panel.cost_bps_of_notional is None
    assert "no traded notional" in panel.reasons["cost_bps_of_notional"]
    assert panel.commission_share is None
    assert panel.total_cost == 0.0
    assert "No trades" in panel.describe()


def test_cost_by_asset_accounts_for_every_penny(returns, equal_weights):
    spec = BacktestSpec(frequency="monthly", costs=CostSpec(commission_bps=12))
    run = run_backtest(returns, equal_weights, spec)
    by_asset = cost_by_asset(run)
    assert by_asset["cost"].sum() == pytest.approx(run.total_cost)
    assert list(by_asset.index) == sorted(by_asset.index, key=lambda a: -by_asset.loc[a, "cost"])


# -- position statistics ----------------------------------------------------


def test_position_episodes_close_when_a_name_leaves_the_book():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    weights = pd.DataFrame(
        {"A": [0.5, 0.5, 0.0, 0.0, 0.5, 0.5], "B": [0.5, 0.5, 1.0, 1.0, 0.5, 0.5]},
        index=index,
    )
    asset_returns = pd.DataFrame(
        {"A": [0.01] * 6, "B": [-0.01] * 6}, index=index
    )
    from optimization_engine.backtest.positions import position_episodes

    episodes = position_episodes(weights, asset_returns)
    a_episodes = [e for e in episodes if e.asset == "A"]
    assert len(a_episodes) == 2
    assert a_episodes[0].closed and a_episodes[0].periods == 2
    assert not a_episodes[1].closed, "still held on the final bar"
    assert a_episodes[0].contribution == pytest.approx(0.01)


def test_position_stats_report_nothing_when_nothing_closed(returns, equal_weights):
    run = run_backtest(returns, equal_weights, BacktestSpec(frequency="none"))
    stats = compute_position_stats(run, returns)
    # A fully invested buy-and-hold book never closes a position.
    assert stats.n_positions is None
    assert stats.n_open_at_end == returns.shape[1]
    assert "no closed positions" in stats.reasons["win_rate"]


def test_position_stats_summarize_closed_round_trips():
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    weights = pd.DataFrame(
        {
            "winner": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "loser": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    asset_returns = pd.DataFrame(
        {"winner": [0.02] * 8, "loser": [-0.03] * 8}, index=index
    )
    run = run_backtest(asset_returns, weights, BacktestSpec(frequency="none"))
    stats = compute_position_stats(run, asset_returns)
    assert stats.n_positions == 2
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.avg_win > 0 and stats.avg_loss < 0
    assert stats.profit_factor == pytest.approx(0.04 / 0.06, rel=1e-9)
    assert stats.avg_holding_periods == pytest.approx(2.0)


# -- sweeps -----------------------------------------------------------------


def test_grid_expands_deterministically_and_validates_paths(returns):
    config = _mean_variance_config(returns)
    sweep = SweepSpec(
        params={
            "optimizer.name": ["min_variance", "equal_weight"],
            "covariance_method": ["sample", "ledoit_wolf"],
        }
    )
    cells = expand_grid(config, sweep)
    assert len(cells) == 4
    # Paths sorted lexicographically, the last one varying fastest — so the
    # same grid expands to the same order every time.
    assert [c.params["covariance_method"] for c in cells] == [
        "sample", "sample", "ledoit_wolf", "ledoit_wolf"
    ]
    assert [c.params["optimizer.name"] for c in cells] == [
        "min_variance", "equal_weight", "min_variance", "equal_weight"
    ]
    assert [c.cell_id for c in cells] == [0, 1, 2, 3]
    assert all(c.config is not None for c in cells)

    with pytest.raises(SweepValidationError, match="does not exist"):
        expand_grid(config, SweepSpec(params={"nonsense.path": [1]}))


def test_optional_optimizer_fields_are_sweepable(returns):
    config = _mean_variance_config(returns)
    cells = expand_grid(
        config, SweepSpec(params={"optimizer.target_volatility": [0.05, 0.10]})
    )
    assert [c.config.optimizer.target_volatility for c in cells] == [0.05, 0.10]


def test_grid_size_is_capped(returns):
    config = _mean_variance_config(returns)
    with pytest.raises(SweepValidationError, match="over max_cells"):
        expand_grid(
            config,
            SweepSpec(params={"optimizer.risk_aversion": [float(i) for i in range(11)]},
                      max_cells=10),
        )
    with pytest.raises(SweepValidationError, match="hard cap"):
        SweepSpec(params={"optimizer.risk_aversion": [1.0]}, max_cells=HARD_CELL_CAP + 1)
    with pytest.raises(SweepValidationError, match="empty value list"):
        SweepSpec(params={"optimizer.risk_aversion": []})


def test_a_failed_cell_is_a_row_never_a_drop(returns):
    config = _mean_variance_config(returns)
    sweep = SweepSpec(params={"optimizer.name": ["min_variance", "no_such_optimizer"]})

    def evaluate(cell_config: EngineConfig) -> pd.Series:
        if cell_config.optimizer.name == "no_such_optimizer":
            raise KeyError("unknown optimizer")
        return pd.Series(
            np.full(len(returns), 0.001), index=returns.index, name="cell"
        )

    results = run_sweep(config, sweep, evaluate)
    assert len(results.frame) == results.n_cells == 2
    assert results.n_ok == 1 and results.n_failed == 1
    assert results.frame.loc[1, "status"] == "run_error"
    assert "unknown optimizer" in results.frame.loc[1, "error"]
    assert np.isnan(results.frame.loc[1, "sharpe"])


def test_sweep_counts_its_own_trials_for_the_deflation(returns):
    rng = np.random.default_rng(7)
    config = _mean_variance_config(returns)
    sweep = SweepSpec(
        params={"optimizer.risk_aversion": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    streams = {
        aversion: pd.Series(
            rng.normal(0.0004, 0.01, len(returns)), index=returns.index
        )
        for aversion in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    }

    def evaluate(cell_config: EngineConfig) -> pd.Series:
        return streams[cell_config.optimizer.risk_aversion]

    results = run_sweep(config, sweep, evaluate)
    assert results.n_ok == 6
    matrix = results.return_matrix()
    assert matrix.shape == (len(returns), 6)

    deflated = results.deflated_sharpe(0)
    assert 0.0 <= deflated.deflated <= 1.0
    assert deflated.n_trials == 6
    # Deflating against six trials raises the bar above zero, so the deflated
    # probability can only be below the undeflated one.
    assert deflated.benchmark_sharpe > 0.0
    assert deflated.deflated <= deflated.probabilistic

    report = results.overfitting_report(n_partitions=6)
    assert 0.0 <= report.pbo <= 1.0
    assert "no ranking" not in results.describe()
    assert not hasattr(results, "best")


def test_sweep_needs_two_cells_to_diagnose_a_selection(returns):
    config = _mean_variance_config(returns)
    sweep = SweepSpec(params={"optimizer.name": ["min_variance"]})
    results = run_sweep(
        config,
        sweep,
        lambda cfg: pd.Series(np.full(len(returns), 0.001), index=returns.index),
    )
    with pytest.raises(ValueError, match="at least two successful"):
        results.overfitting_report()


# -- holdout ----------------------------------------------------------------


def test_gating_truncates_rather_than_flags(returns):
    boundary = returns.index[len(returns) // 2]
    gated = gate_returns(returns, boundary)
    assert gated.index.max() == boundary
    assert len(gated) + len(holdout_segment(returns, boundary)) == len(returns)
    assert_within_holdout(gated, boundary)
    with pytest.raises(HoldoutViolationError, match="past the holdout boundary"):
        assert_within_holdout(returns, boundary)


def test_holdout_needs_a_segment_to_evaluate(returns):
    with pytest.raises(ValueError, match="No held-out segment"):
        final_holdout_run(
            returns, returns.index[-1], lambda seg: seg.iloc[:, 0], audit_path="unused"
        )


def test_the_audit_log_flags_a_second_look(returns, tmp_path):
    audit = tmp_path / "holdout_audit.jsonl"
    boundary = returns.index[len(returns) // 2]
    strategy = {"optimizer": "min_variance", "lookback": 252}

    def evaluate(segment: pd.DataFrame) -> pd.Series:
        return segment.mean(axis=1).rename("holdout")

    first = final_holdout_run(
        returns, boundary, evaluate, strategy=strategy, audit_path=audit
    )
    assert first.is_first_look
    assert "First look" in first.describe()
    assert not first.summary.empty

    second = final_holdout_run(
        returns, boundary, evaluate, strategy=strategy, audit_path=audit
    )
    assert REPEATED in second.flags

    moved = final_holdout_run(
        returns,
        returns.index[len(returns) // 2 + 10],
        evaluate,
        strategy=strategy,
        audit_path=audit,
    )
    assert SHIFTED_HOLDOUT in moved.flags
    assert REPEATED not in moved.flags

    rows = read_audit_log(audit)
    assert len(rows) == 3
    assert [json.loads(line)["label"] for line in audit.read_text().splitlines()] == [
        "final", "final", "final"
    ]


def test_a_different_strategy_is_not_a_second_look(returns, tmp_path):
    audit = tmp_path / "audit.jsonl"
    boundary = returns.index[len(returns) // 2]

    def evaluate(segment: pd.DataFrame) -> pd.Series:
        return segment.mean(axis=1)

    final_holdout_run(
        returns, boundary, evaluate, strategy={"optimizer": "a"}, audit_path=audit
    )
    other = final_holdout_run(
        returns, boundary, evaluate, strategy={"optimizer": "b"}, audit_path=audit
    )
    assert other.flags == ()


def test_the_audit_clock_is_injectable(returns, tmp_path):
    audit = tmp_path / "audit.jsonl"
    stamped = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
    outcome = final_holdout_run(
        returns,
        returns.index[len(returns) // 2],
        lambda segment: segment.mean(axis=1),
        audit_path=audit,
        clock=lambda: stamped,
    )
    assert outcome.audit_row["timestamp"] == stamped.isoformat()


# -- tearsheet --------------------------------------------------------------


def test_tearsheet_states_what_the_run_did_not_model(returns, equal_weights):
    run = run_backtest(returns, equal_weights, BacktestSpec(frequency="monthly"))
    sheet = build_tearsheet(run, returns)
    joined = " ".join(sheet.caveats)
    assert "chosen knowing these returns" in joined
    assert "modelled as free" in joined
    assert "fill on the close" in joined
    assert "undeflated" in joined
    assert sheet.deflated_sharpe is None


def test_tearsheet_drops_the_caveats_the_run_answered(returns, equal_weights):
    spec = BacktestSpec(
        frequency="monthly",
        costs=CostSpec(commission_bps=10, slippage_bps=5),
        execution_lag=1,
        is_out_of_sample=True,
    )
    run = run_backtest(returns, equal_weights, spec)
    sheet = build_tearsheet(run, returns, n_trials=12)
    joined = " ".join(sheet.caveats)
    assert "chosen knowing these returns" not in joined
    assert "modelled as free" not in joined
    assert "fill on the close" not in joined
    assert sheet.deflated_sharpe is not None
    assert "performance" in sheet.to_frames()
    assert sheet.metadata["result_hash"] == run.meta.result_hash


def test_engine_exposes_the_whole_stack(returns):
    from optimization_engine.engine import run_engine

    config = _mean_variance_config(returns)
    run = run_engine(returns.iloc[:400], config)
    bundle = run.simulate(BacktestSpec(costs=CostSpec(commission_bps=10)))
    assert bundle.meta.spec_hash
    sheet = run.tearsheet(bundle)
    assert not sheet.performance.empty
    assert sheet.tca.total_cost > 0


def test_a_walk_forward_prices_its_first_trades_off_the_whole_history(returns):
    """Impact must not degrade for want of data that is sitting right there.

    The evaluation window starts partway into the sample, so estimating
    trailing volatility from it alone would leave the first trades unpriced
    even though years of returns precede them.
    """
    spec = BacktestSpec(
        costs=CostSpec(impact_coefficient=0.5, impact_volatility_lookback=63)
    )

    def solve(window: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    walk = walk_forward_run(returns, solve, lookback=252, rebalance_every=126, spec=spec)
    assert not walk.run.meta.degradations, walk.run.meta.degradations

    # Replaying the same weights without that context degrades, which is what
    # makes the line above a result rather than a coincidence.
    blind = run_backtest(
        returns.loc[walk.weights_history.index[0]:], walk.weights_history, spec
    )
    assert blind.meta.degradations
    assert blind.total_cost < walk.run.total_cost
