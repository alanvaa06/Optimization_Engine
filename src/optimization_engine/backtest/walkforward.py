"""Walk-forward evaluation: re-estimate, re-solve, hold forward, repeat.

An in-sample replay tells you how a weight vector would have done on the
returns it was fitted to. That is a description of the past, not a track
record. A walk-forward run evaluates the *process*: at each decision the
optimizer sees only returns strictly before the decision date, and the
weights it produces are held over the periods that follow. Nothing from the
evaluation window reaches the estimate.

**Two cadences, and they answer different questions.** ``rebalance_every`` is
how often the optimizer *re-solves* — how stale the desk is willing to let its
view get. ``rebalance_frequency`` is how often the book is *traded back* to
whichever target is current — how far the desk is willing to let the weights
drift away from that view. Real mandates separate them: a quarterly investment
committee running a monthly rebalancing discipline re-solves four times a year
and trades twelve. Collapsing the two ties the drift tolerance to the research
calendar for no reason other than that it was easier to implement, and it
understates the turnover of every policy that rebalances more often than it
re-optimizes.

The default is ``"none"``: trade only when a new target is solved. That is the
minimal reading of "hold the solution forward", and it is what this runner has
always done — a policy has to ask for the extra trading before it pays for it.

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
from optimization_engine.backtest.spec import BacktestSpec, RebalanceFrequency


@dataclass
class WalkForwardRun:
    """The out-of-sample track record of an optimization process.

    Attributes:
        run: The replayed result, tagged out-of-sample.
        weights_history: Target weights by decision date — one row per
            re-solve, not per trade. With a trading cadence finer than the
            re-solve cadence the book trades more often than this frame has
            rows, and ``run.rebalance_dates`` is the record of that.
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
    def n_resolves(self) -> int:
        """How many times the optimizer actually re-solved."""
        return len(self.weights_history)

    @property
    def n_rebalances(self) -> int:
        """Deprecated spelling of :attr:`n_resolves`.

        Kept because callers and saved notebooks use it, but the name is now
        ambiguous: with the two cadences separated, the book can rebalance
        without the optimizer re-solving. Count trades with
        :attr:`n_trade_dates`.
        """
        return self.n_resolves

    @property
    def n_trade_dates(self) -> int:
        """How many dates the book actually traded on.

        Equal to :attr:`n_resolves` when the trading cadence is ``"none"``,
        and larger when a calendar rebalance sits between re-solves.
        """
        return len(self.run.rebalance_dates)

    @property
    def n_failures(self) -> int:
        return len(self.failures)

    @property
    def rebalance_frequency(self) -> str:
        """The trading calendar this run was actually evaluated on."""
        return str(self.metadata.get("rebalance_frequency", "none"))

    def weight_stability(self) -> pd.Series:
        """Average absolute change in each asset's weight between re-solves.

        High values mean the optimizer is chasing estimation noise: the
        allocation is being rewritten every window on data that barely moved.
        This reads the solved targets, so it measures the optimizer, not the
        drift that the trading cadence corrects for.
        """
        if len(self.weights_history) < 2:
            return pd.Series(dtype=float)
        return self.weights_history.diff().abs().mean()

    def describe(self) -> str:
        trading = (
            "trading only on re-solves"
            if self.rebalance_frequency == "none"
            else f"{self.rebalance_frequency} rebalancing between re-solves"
        )
        return (
            f"{self.n_resolves} re-solves, {self.n_trade_dates} trade dates "
            f"({trading}); {self.n_failures} failed solve(s)."
        )


def _resolve_trading_cadence(
    spec: BacktestSpec | None, rebalance_frequency: RebalanceFrequency | None
) -> RebalanceFrequency:
    """Decide which calendar the book trades on, without overriding anyone.

    The old runner forced ``frequency="none"`` onto whatever spec it was
    handed, so a caller who asked for monthly rebalancing got buy-and-hold
    between re-solves and no signal that their request had been dropped. The
    rules here are the smallest set that never does that:

    * an explicit ``rebalance_frequency`` always wins;
    * otherwise a caller-supplied spec means what it says;
    * otherwise — no spec, no argument — trade only on re-solves.
    """
    if rebalance_frequency is not None:
        return rebalance_frequency
    if spec is not None:
        return spec.frequency
    return "none"


def walk_forward_run(
    returns: pd.DataFrame,
    solve: Callable[[pd.DataFrame], pd.Series],
    *,
    lookback: int,
    rebalance_every: int,
    spec: BacktestSpec | None = None,
    min_lookback: int | None = None,
    expanding: bool = False,
    rebalance_frequency: RebalanceFrequency | None = None,
    prices: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
) -> WalkForwardRun:
    """Roll an estimation window forward, re-solving as it goes.

    Args:
        returns: Full periodic return history.
        solve: Callable taking a returns window and returning target weights.
        lookback: Estimation window length in periods.
        rebalance_every: Periods between **re-solves** — the re-optimization
            cadence. This is the one that decides how often the optimizer
            sees new data; it does not, on its own, decide how often the book
            trades.
        spec: Run description. ``is_out_of_sample`` is forced on, because it
            is. Its ``frequency`` is read as the trading cadence unless
            ``rebalance_frequency`` overrides it.
        min_lookback: Minimum window before the first solve. Defaults to
            ``lookback``.
        expanding: Grow the window from the start instead of rolling it.
        rebalance_frequency: How often the book is **traded back** to the
            most recent solved target between re-solves — the rebalancing
            cadence. ``"none"`` (the default when neither this nor a spec is
            given) trades only when a new target is solved and lets the
            weights drift in between. Any other cadence adds calendar trades
            on top of the re-solves; the target they trade to is always the
            freshest one available on that date, never a future one.
        prices: Close prices, needed only when the cost model prices capacity
            from traded volume.
        volumes: Traded volume per asset and period. Optional: without one the
            impact model uses its fixed participation rate, which is the only
            thing it can do for an index universe.

    Returns:
        The bundle. ``n_resolves`` counts optimizations, ``n_trade_dates``
        counts trades, and the two differ exactly when the cadences do.

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

    trading = _resolve_trading_cadence(spec, rebalance_frequency)
    base = spec or BacktestSpec()
    spec = base.with_(frequency=trading, is_out_of_sample=True)

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
        "rebalance_frequency": str(trading),
        "expanding": bool(expanding),
        "n_resolves": int(len(weights_history)),
        # The old key, kept so saved runs and reports keep reading.
        "n_rebalances": int(len(weights_history)),
        "n_failed_solves": int(len(failures)),
    }
    # The cost models see the whole history, not just the evaluated slice:
    # a decision made at the first evaluated date has years of returns behind
    # it, and pricing its impact off the slice alone would throw them away.
    run = run_backtest(
        evaluation,
        weights_history,
        spec,
        notes=notes,
        context_returns=returns,
        prices=prices,
        volumes=volumes,
    )

    return WalkForwardRun(
        run=run,
        weights_history=weights_history,
        windows=pd.DataFrame(window_rows),
        failures=tuple(failures),
        metadata=dict(notes),
    )


__all__ = ["WalkForwardRun", "walk_forward_run"]
