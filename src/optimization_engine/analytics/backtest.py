"""Backtesting: weight drift, rebalancing, transaction costs, and walk-forward.

The default "backtest" in most optimizer demos applies the solved weights to
the same return history the optimizer was fitted on, and holds those weights
constant forever. Both halves of that are wrong in ways that flatter the
result:

* **Look-ahead.** The covariance and expected returns were estimated on the
  very returns being replayed. The optimizer already knew which assets won.
* **Costless continuous rebalancing.** Holding constant weights without
  saying so assumes the book is rebalanced every period, for free. Real
  weights drift with performance, and pulling them back costs money.

This module provides both an honest in-sample replay (:func:`backtest_weights`,
with explicit drift, rebalancing and costs) and a genuinely out-of-sample
:func:`walk_forward_backtest` that re-estimates and re-solves on a rolling
window and only ever holds positions forward in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd

RebalanceFrequency = Literal[
    "none", "daily", "weekly", "monthly", "quarterly", "annual"
]

#: Pandas offset aliases for each supported rebalance cadence.
_FREQ_ALIAS: dict[str, str | None] = {
    "none": None,
    "daily": None,  # every period; handled without resampling
    "weekly": "W",
    "monthly": "ME",
    "quarterly": "QE",
    "annual": "YE",
}

REBALANCE_DESCRIPTIONS: dict[str, str] = {
    "none": (
        "Buy and hold. Weights drift with performance; winners compound into "
        "a larger share of the book."
    ),
    "daily": "Rebalance every period. Zero drift, maximum turnover and cost.",
    "weekly": "Rebalance weekly.",
    "monthly": "Rebalance monthly — the common institutional default.",
    "quarterly": "Rebalance quarterly.",
    "annual": "Rebalance annually. Low cost, large intra-year drift.",
}


def rebalance_dates(
    index: pd.DatetimeIndex, frequency: RebalanceFrequency
) -> pd.DatetimeIndex:
    """Dates on which the book is traded back to target weights.

    The first date is always included: that is the initial purchase.
    """
    if len(index) == 0:
        return pd.DatetimeIndex([])
    if frequency == "none":
        return pd.DatetimeIndex([index[0]])
    if frequency == "daily":
        return pd.DatetimeIndex(index)
    alias = _FREQ_ALIAS.get(frequency)
    if alias is None:
        raise ValueError(
            f"Unknown rebalance frequency {frequency!r}. "
            f"Available: {sorted(_FREQ_ALIAS)}"
        )
    marks = pd.Series(index, index=index).resample(alias).last().dropna()
    dates = pd.DatetimeIndex(marks.values)
    if index[0] not in dates:
        dates = pd.DatetimeIndex([index[0]]).append(dates)
    return dates.unique().sort_values()


@dataclass
class BacktestResult:
    """Outcome of replaying a weight schedule over a return history.

    Attributes:
        returns: Portfolio returns, net of transaction costs.
        gross_returns: The same series before costs.
        weights: Actual held weights at the start of each period, after drift.
        turnover: One-way turnover traded on each rebalance date.
        costs: Transaction cost charged on each rebalance date.
        rebalance_dates: When the book was traded.
        is_out_of_sample: Whether the weights were chosen without seeing the
            returns they are evaluated on. The single most important caveat
            attached to any backtest number.
        metadata: Free-form run description (lookback, frequency, costs …).
    """

    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    rebalance_dates: pd.DatetimeIndex
    is_out_of_sample: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def total_turnover(self) -> float:
        return float(self.turnover.sum())

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())

    @property
    def annualized_turnover(self) -> float:
        periods_per_year = int(self.metadata.get("periods_per_year", 252))
        years = len(self.returns) / periods_per_year
        return float(self.turnover.sum() / years) if years > 0 else float("nan")

    def wealth(self, starting: float = 1.0) -> pd.Series:
        return starting * (1 + self.returns).cumprod()

    def summary(self, periods_per_year: int | None = None, riskfree_rate: float = 0.0):
        """Standard performance summary of the net-of-cost return stream."""
        from optimization_engine.analytics.performance import summary_stats

        ppy = periods_per_year or int(self.metadata.get("periods_per_year", 252))
        return summary_stats(
            self.returns.to_frame("portfolio"),
            periods_per_year=ppy,
            riskfree_rate=riskfree_rate,
            extended=True,
        )

    def cost_drag(self, periods_per_year: int | None = None) -> float:
        """Annualized return given up to transaction costs."""
        from optimization_engine.analytics.performance import annualize_returns

        ppy = periods_per_year or int(self.metadata.get("periods_per_year", 252))
        gross = annualize_returns(self.gross_returns, ppy)
        net = annualize_returns(self.returns, ppy)
        return float(gross - net)


def backtest_weights(
    returns: pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
    frequency: RebalanceFrequency = "monthly",
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
    is_out_of_sample: bool = False,
) -> BacktestResult:
    """Replay a weight schedule, letting positions drift between rebalances.

    Args:
        returns: Periodic asset returns.
        weights: Either one target vector held throughout, or a frame of
            target weights indexed by the date they become effective (used by
            the walk-forward runner).
        frequency: How often the book is traded back to target.
        transaction_cost_bps: One-way cost in basis points of traded notional.
            25 bps on 100% turnover costs 25 bps of NAV.
        periods_per_year: Observations per year, for annualizing turnover.
        is_out_of_sample: Tag the result; see :class:`BacktestResult`.

    Raises:
        ValueError: If ``returns`` is empty or the weights cover none of it.
    """
    if returns is None or returns.empty:
        raise ValueError("Cannot backtest on empty returns.")

    assets = list(returns.columns)
    if isinstance(weights, pd.Series):
        schedule = pd.DataFrame(
            [weights.reindex(assets).fillna(0.0).values],
            index=[returns.index[0]],
            columns=assets,
        )
    else:
        schedule = weights.reindex(columns=assets).fillna(0.0)
        schedule = schedule.sort_index()
        if schedule.empty:
            raise ValueError("Weight schedule is empty.")

    marks = rebalance_dates(returns.index, frequency)
    mark_set = set(marks)

    held = np.zeros(len(assets))
    current_target = schedule.iloc[0].values.astype(float)
    held_rows: list[np.ndarray] = []
    gross: list[float] = []
    net: list[float] = []
    turnover_by_date: dict[pd.Timestamp, float] = {}
    cost_by_date: dict[pd.Timestamp, float] = {}
    cost_rate = transaction_cost_bps / 10_000.0
    first = True

    for date in returns.index:
        # A new target from the schedule always forces a trade: the walk-forward
        # runner only emits one when the optimizer has actually re-solved.
        if date in schedule.index:
            current_target = schedule.loc[date].values.astype(float)
            trade_now = True
        else:
            trade_now = date in mark_set

        if first or trade_now:
            traded = float(np.abs(current_target - held).sum())
            if traded > 1e-12:
                turnover_by_date[date] = traded
                cost_by_date[date] = traded * cost_rate
            held = current_target.copy()
            first = False

        held_rows.append(held.copy())
        period = returns.loc[date].values.astype(float)
        gross_ret = float(held @ period)
        gross.append(gross_ret)
        net.append(gross_ret - cost_by_date.get(date, 0.0))

        # Drift: positions grow with their own return, then renormalize to the
        # new portfolio value so the weights still sum to the invested total.
        grown = held * (1.0 + period)
        total = grown.sum()
        held = grown / total if abs(total) > 1e-12 else grown

    index = returns.index
    return BacktestResult(
        returns=pd.Series(net, index=index, name="portfolio"),
        gross_returns=pd.Series(gross, index=index, name="gross"),
        weights=pd.DataFrame(held_rows, index=index, columns=assets),
        turnover=pd.Series(turnover_by_date, dtype=float).reindex(
            pd.DatetimeIndex(sorted(turnover_by_date))
        ),
        costs=pd.Series(cost_by_date, dtype=float).reindex(
            pd.DatetimeIndex(sorted(cost_by_date))
        ),
        rebalance_dates=marks,
        is_out_of_sample=is_out_of_sample,
        metadata={
            "frequency": frequency,
            "transaction_cost_bps": transaction_cost_bps,
            "periods_per_year": periods_per_year,
        },
    )


@dataclass
class WalkForwardResult:
    """Out-of-sample walk-forward evaluation of an optimization process."""

    backtest: BacktestResult
    weights_history: pd.DataFrame
    windows: pd.DataFrame
    failures: tuple[str, ...] = ()

    @property
    def returns(self) -> pd.Series:
        return self.backtest.returns

    @property
    def n_rebalances(self) -> int:
        return len(self.weights_history)

    def weight_stability(self) -> pd.Series:
        """Average absolute change in each asset's weight between rebalances.

        High values mean the optimizer is chasing estimation noise: the
        allocation is being rewritten every period on data that barely moved.
        """
        if len(self.weights_history) < 2:
            return pd.Series(dtype=float)
        return self.weights_history.diff().abs().mean()


def walk_forward_backtest(
    returns: pd.DataFrame,
    solve: Callable[[pd.DataFrame], pd.Series],
    lookback: int,
    rebalance_every: int,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
    min_lookback: int | None = None,
    expanding: bool = False,
) -> WalkForwardResult:
    """Re-estimate and re-solve on a rolling window, holding results forward.

    At each rebalance the optimizer sees only returns strictly *before* the
    decision date; the resulting weights are then held over the following
    ``rebalance_every`` periods. No information from the evaluation window
    reaches the estimate, which is what makes the resulting track record
    something other than a description of the past.

    Args:
        returns: Full periodic return history.
        solve: Callable taking a returns window and returning target weights.
            Typically wraps ``run_engine`` with a fixed config.
        lookback: Estimation window length in periods.
        rebalance_every: Periods between re-solves.
        transaction_cost_bps: One-way cost on traded notional.
        periods_per_year: Observations per year.
        min_lookback: Minimum window before the first solve. Defaults to
            ``lookback``.
        expanding: Use a growing window anchored at the start instead of a
            fixed-length rolling one.

    Raises:
        ValueError: If the history is too short to produce a single
            out-of-sample period, or if every solve fails.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be at least 2 periods; got {lookback}.")
    if rebalance_every < 1:
        raise ValueError(
            f"rebalance_every must be at least 1 period; got {rebalance_every}."
        )
    min_lookback = min_lookback or lookback
    n = len(returns)
    if n <= min_lookback:
        raise ValueError(
            f"Need more than {min_lookback} observations to evaluate anything "
            f"out of sample; got {n}. Shorten the lookback or load more history."
        )

    decision_points = list(range(min_lookback, n, rebalance_every))
    schedule: dict[pd.Timestamp, pd.Series] = {}
    window_rows: list[dict[str, object]] = []
    failures: list[str] = []
    last_weights: pd.Series | None = None

    for pos in decision_points:
        start = 0 if expanding else max(0, pos - lookback)
        window = returns.iloc[start:pos]
        decision_date = returns.index[pos]
        try:
            w = solve(window)
            w = w.reindex(returns.columns).fillna(0.0)
            schedule[decision_date] = w
            last_weights = w
            status = "ok"
        except Exception as exc:
            # Carrying the previous book forward is what a desk would actually
            # do when a re-solve fails; skipping the period silently would
            # remove a real cost from the track record.
            failures.append(f"{decision_date.date()}: {exc}")
            status = f"failed: {exc}"
            if last_weights is not None:
                schedule[decision_date] = last_weights
        window_rows.append(
            {
                "decision_date": decision_date,
                "window_start": returns.index[start],
                "window_end": returns.index[pos - 1],
                "window_length": pos - start,
                "status": status,
            }
        )

    if not schedule:
        raise ValueError(
            "Every walk-forward solve failed; there is nothing to evaluate. "
            f"First error: {failures[0] if failures else 'unknown'}"
        )

    weights_history = pd.DataFrame(schedule).T.sort_index()
    first_date = weights_history.index[0]
    evaluation = returns.loc[first_date:]

    backtest = backtest_weights(
        evaluation,
        weights_history,
        frequency="none",  # trades happen on schedule dates only
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        is_out_of_sample=True,
    )
    backtest.metadata.update(
        {
            "lookback": lookback,
            "rebalance_every": rebalance_every,
            "expanding": expanding,
            "n_rebalances": len(weights_history),
            "n_failed_solves": len(failures),
        }
    )

    return WalkForwardResult(
        backtest=backtest,
        weights_history=weights_history,
        windows=pd.DataFrame(window_rows),
        failures=tuple(failures),
    )


def compare_in_and_out_of_sample(
    in_sample: pd.Series,
    out_of_sample: pd.Series,
    periods_per_year: int = 252,
    riskfree_rate: float = 0.0,
) -> pd.DataFrame:
    """Put the fitted and the walk-forward track records side by side.

    The gap between the two columns is the honest measure of how much of the
    backtest was optimization and how much was hindsight.
    """
    from optimization_engine.analytics.performance import summary_stats

    frame = pd.concat(
        [
            in_sample.rename("In-sample (fitted)"),
            out_of_sample.rename("Out-of-sample (walk-forward)"),
        ],
        axis=1,
    )
    stats = summary_stats(
        frame, periods_per_year=periods_per_year, riskfree_rate=riskfree_rate
    ).T
    stats["Degradation"] = stats["In-sample (fitted)"] - stats["Out-of-sample (walk-forward)"]
    return stats
