"""Walk-forward evaluation: re-estimate, re-solve, hold forward, repeat.

An in-sample replay tells you how a weight vector would have done on the
returns it was fitted to. That is a description of the past, not a track
record. A walk-forward run evaluates the *process*: at each decision the
optimizer sees only returns strictly before the decision date, and the
weights it produces are held over the periods that follow. Nothing from the
evaluation window reaches the estimate.

The runner is deliberately honest about failure. When a solve raises — an
infeasible constraint set on one window, a covariance that will not invert —
the previous book is carried forward, which is what a desk would actually do,
and the failure is recorded. Silently skipping the period would remove a real
cost from the track record and make a fragile process look robust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from optimization_engine.backtest.results import RunResult
from optimization_engine.backtest.runner import run_backtest
from optimization_engine.backtest.spec import BacktestSpec


@dataclass
class WalkForwardRun:
    """The out-of-sample track record of an optimization process.

    Attributes:
        run: The replayed result, tagged out-of-sample.
        weights_history: Target weights by decision date.
        windows: One row per decision — window bounds, length, and status.
        failures: Human-readable reasons for each failed solve.
    """

    run: RunResult
    weights_history: pd.DataFrame
    windows: pd.DataFrame
    failures: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.run.returns

    @property
    def n_rebalances(self) -> int:
        return len(self.weights_history)

    @property
    def n_failures(self) -> int:
        return len(self.failures)

    def weight_stability(self) -> pd.Series:
        """Average absolute change in each asset's weight between decisions.

        High values mean the optimizer is chasing estimation noise: the
        allocation is being rewritten every period on data that barely moved.
        """
        if len(self.weights_history) < 2:
            return pd.Series(dtype=float)
        return self.weights_history.diff().abs().mean()


def walk_forward_run(
    returns: pd.DataFrame,
    solve: Callable[[pd.DataFrame], pd.Series],
    *,
    lookback: int,
    rebalance_every: int,
    spec: BacktestSpec | None = None,
    min_lookback: int | None = None,
    expanding: bool = False,
) -> WalkForwardRun:
    """Roll an estimation window forward, re-solving as it goes.

    Args:
        returns: Full periodic return history.
        solve: Callable taking a returns window and returning target weights.
        lookback: Estimation window length in periods.
        rebalance_every: Periods between re-solves.
        spec: Run description. Its ``frequency`` is forced to ``"none"``,
            since the decision schedule already says when the book trades,
            and ``is_out_of_sample`` is forced on.
        min_lookback: Minimum window before the first solve. Defaults to
            ``lookback``.
        expanding: Grow the window from the start instead of rolling it.

    Raises:
        ValueError: If the parameters are degenerate, the history is too
            short to produce a single out-of-sample period, or every solve
            failed.
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

    base = spec or BacktestSpec()
    spec = base.with_(frequency="none", is_out_of_sample=True)

    schedule: dict[pd.Timestamp, pd.Series] = {}
    window_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    last_weights: pd.Series | None = None

    for position in range(min_lookback, n, rebalance_every):
        start = 0 if expanding else max(0, position - lookback)
        window = returns.iloc[start:position]
        decision_date = returns.index[position]
        try:
            solved = solve(window).reindex(returns.columns).fillna(0.0)
            schedule[decision_date] = solved
            last_weights = solved
            status = "ok"
        except Exception as exc:  # noqa: BLE001 — a failed solve is a row, never a drop
            failures.append(f"{pd.Timestamp(decision_date).date()}: {exc}")
            status = f"failed: {exc}"
            if last_weights is not None:
                schedule[decision_date] = last_weights
        window_rows.append(
            {
                "decision_date": decision_date,
                "window_start": returns.index[start],
                "window_end": returns.index[position - 1],
                "window_length": position - start,
                "status": status,
            }
        )

    if not schedule:
        raise ValueError(
            "Every walk-forward solve failed; there is nothing to evaluate. "
            f"First error: {failures[0] if failures else 'unknown'}"
        )

    weights_history = pd.DataFrame(schedule).T.sort_index()
    evaluation = returns.loc[weights_history.index[0]:]
    notes = {
        "lookback": int(lookback),
        "rebalance_every": int(rebalance_every),
        "expanding": bool(expanding),
        "n_rebalances": int(len(weights_history)),
        "n_failed_solves": int(len(failures)),
    }
    run = run_backtest(evaluation, weights_history, spec, notes=notes)

    return WalkForwardRun(
        run=run,
        weights_history=weights_history,
        windows=pd.DataFrame(window_rows),
        failures=tuple(failures),
        metadata=dict(notes),
    )


__all__ = ["WalkForwardRun", "walk_forward_run"]
