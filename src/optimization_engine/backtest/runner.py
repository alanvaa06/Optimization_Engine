"""The simulation core: weights in, a :class:`RunResult` out.

This is the only place in the library that advances a portfolio through time,
and it is deliberately dull. It reads no files, holds no state between calls,
and consults nothing but its arguments, so the same spec and the same returns
always produce the same result hash. Everything interesting — cost models,
robustness diagnostics, sweeps — is layered around it rather than inside it.

What the loop models, in order, on each period:

1. **Execution.** If a decision reaches its execution date, the book is
   traded to the target. Costs are charged on the traded notional and taken
   out of that period's return.
2. **Return.** The held weights earn the period's asset returns.
3. **Drift.** Positions grow with their own return and are renormalized to
   the new portfolio value, so the weights still describe the book. This is
   what makes a "buy and hold" backtest different from a costlessly
   rebalanced one, and the difference compounds.

The one thing the loop refuses to do is trade on information it does not yet
have. A decision taken on date ``t`` executes at ``t + execution_lag``, and
the volatility that prices its impact is estimated on returns strictly
before ``t``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from optimization_engine.backtest.calendar import execution_positions, rebalance_dates
from optimization_engine.backtest.costs import (
    CostModel,
    MarketContext,
    build_cost_model,
    context_request,
    trailing_dollar_volume,
    trailing_volatilities,
)
from optimization_engine.backtest.results import (
    COST_COLUMNS,
    TRADE_COLUMNS,
    RunResult,
    build_meta,
    empty_costs,
    empty_trades,
)
from optimization_engine.backtest.spec import MIN_ADV_CAPITAL, BacktestSpec

#: Traded fractions below this are float residue, not orders.
_TRADE_EPS = 1e-12

#: Cap on distinct degradation reasons carried on the meta. The reasons are a
#: diagnostic, not a log file; a run that degrades on every trade needs to say
#: so once, not sixty thousand times.
_MAX_DEGRADATIONS = 20


def _as_schedule(
    weights: pd.Series | pd.DataFrame, assets: list[str], first_date: pd.Timestamp
) -> pd.DataFrame:
    """Normalize either weight form into a frame of targets by effective date."""
    if isinstance(weights, pd.Series):
        return pd.DataFrame(
            [weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)],
            index=[first_date],
            columns=assets,
        )
    schedule = weights.reindex(columns=assets).fillna(0.0).sort_index()
    if schedule.empty:
        raise ValueError("Weight schedule is empty.")
    return schedule.astype(float)


def run_backtest(
    returns: pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
    spec: BacktestSpec | None = None,
    *,
    cost_model: CostModel | None = None,
    notes: dict[str, Any] | None = None,
    context_returns: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
) -> RunResult:
    """Replay a weight schedule over a return history.

    Args:
        returns: Periodic asset returns, one column per asset. This is the
            evaluation window: every period here is replayed.
        weights: Either one target vector held throughout, or a frame of
            target weights indexed by the date they become effective — which
            is what the walk-forward runner produces.
        spec: The run description. Defaults to a monthly, costless,
            same-period-fill, in-sample replay.
        cost_model: Override the model built from ``spec.costs``. Useful for
            testing and for cost models this library does not ship.
        notes: Free-form annotations to carry on the result metadata.
        context_returns: A longer history the cost models may estimate from.
            A walk-forward evaluation starts partway into the sample, so
            estimating trailing volatility from the evaluation window alone
            would degrade its first trades for want of data that exists —
            just outside the slice. Only rows strictly before each decision
            date are ever read, so this widens what can be estimated without
            widening what can be seen.
        prices: Close prices for the same assets, needed only to turn share
            volume into traded notional. Ignored unless the cost model prices
            capacity from ADV.
        volumes: Traded volume per asset and period. Optional throughout: with
            no volume panel — an index universe, a fund NAV series, any
            provider that does not publish it — the impact model prices from
            its fixed participation rate and records that it did so. Supplying
            one only matters when
            ``spec.costs.impact_participation_source == "adv"``.

    Returns:
        The full result bundle. See :class:`~optimization_engine.backtest.results.RunResult`.

    Raises:
        ValueError: If ``returns`` is empty, the weight schedule is, or the
            cost model prices impact from volume while ``spec.initial_capital``
            is too small for that to mean anything.

    """
    if returns is None or returns.empty:
        raise ValueError("Cannot backtest on empty returns.")

    spec = spec or BacktestSpec()
    assets = list(returns.columns)
    index = pd.DatetimeIndex(returns.index)
    schedule = _as_schedule(weights, assets, index[0])
    model = cost_model if cost_model is not None else build_cost_model(spec.costs)

    # Every number that parameterizes the charge comes from the model, never
    # from the spec. The spec's cost fields have already done their job by
    # building the model; reading them again here would mean a model handed in
    # through ``cost_model=`` — which never passes through
    # ``build_cost_model`` — quietly ran on somebody else's numbers.
    wanted = context_request(model)

    volatility = None
    if wanted.volatility_lookback > 0:
        history = returns if context_returns is None else context_returns
        volatility = trailing_volatilities(
            history, wanted.volatility_lookback, wanted.volatility_min_observations
        ).reindex(index=index, columns=assets)

    if wanted.participation_lookback > 0 and float(spec.initial_capital) < MIN_ADV_CAPITAL:
        # Fund size is genuinely the run's, not the model's, so it does stay on
        # the spec — and this is the one place every route converges on. Left
        # through, an ADV charge against a one-currency-unit book rounds to
        # zero and reads as proof the strategy has no capacity limit.
        raise ValueError(
            "This cost model prices impact from traded volume, which needs a "
            f"real fund size: initial_capital is {spec.initial_capital:g}. "
            "Set it to the capital being deployed."
        )

    # Traded notional is computed only when a model actually asks for it, so a
    # universe with no volume — the common case for indices — costs nothing
    # and behaves identically to one that was never offered any.
    adv_notional = None
    if wanted.participation_lookback > 0 and volumes is not None and prices is not None:
        adv_notional = trailing_dollar_volume(
            prices,
            volumes,
            wanted.participation_lookback,
            wanted.participation_min_observations,
        ).reindex(index=index, columns=assets)
    adv_share = wanted.adv_share

    marks = rebalance_dates(index, spec.frequency)
    # A schedule date is always a decision: the walk-forward runner only emits
    # one when the optimizer has genuinely re-solved, so ignoring it would
    # silently discard the re-solve.
    decisions = pd.DatetimeIndex(
        sorted(set(marks).union(d for d in schedule.index if d in set(index)))
    )
    execution = execution_positions(index, decisions, spec.execution_lag)

    # Resolve, once, which target each decision is actually asking for: the
    # most recent schedule row effective on or before the decision date.
    schedule_positions = np.searchsorted(
        schedule.index.to_numpy(), index.to_numpy(), side="right"
    )
    targets_at_execution: dict[int, np.ndarray] = {}
    for decision_pos in sorted(execution):
        row = int(schedule_positions[decision_pos]) - 1
        if row < 0:
            # A calendar mark before the first target exists: nothing to trade to.
            continue
        # A later decision landing on the same execution date supersedes an
        # earlier one — the desk trades the freshest target it holds.
        targets_at_execution[execution[decision_pos]] = schedule.iloc[row].to_numpy(
            dtype=float
        )

    returns_matrix = returns.to_numpy(dtype=float)
    held = np.zeros(len(assets))
    held_rows: list[np.ndarray] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    nav_path: list[float] = []
    trade_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    degradations: list[str] = []
    seen_degradations: set[str] = set()
    traded_dates: list[pd.Timestamp] = []
    nav = float(spec.initial_capital)

    for position, date in enumerate(index):
        period_cost = 0.0
        target = targets_at_execution.get(position)
        if target is not None:
            deltas = target - held
            turnover = float(np.abs(deltas).sum())
            if turnover > _TRADE_EPS:
                commission_total = 0.0
                slippage_total = 0.0
                # Accumulated over the frame's own column order: a fixed
                # summation order is what makes the result hash reproducible.
                for asset_position, asset in enumerate(assets):
                    delta = float(deltas[asset_position])
                    if abs(delta) <= _TRADE_EPS:
                        continue
                    sigma = None
                    if volatility is not None:
                        raw = volatility.iat[position, asset_position]
                        sigma = None if pd.isna(raw) else float(raw)
                    participation = None
                    if adv_notional is not None and nav > 0.0:
                        capacity = adv_notional.iat[position, asset_position]
                        if not pd.isna(capacity):
                            # Capacity is a currency amount; what the impact law
                            # needs is its share of *this* book, so a fund that
                            # has grown sees the same name as less liquid.
                            participation = float(capacity) * adv_share / nav
                    quote = model.charge(
                        asset=asset,
                        traded_weight=delta,
                        context=MarketContext(
                            volatility=sigma, participation=participation
                        ),
                    )
                    if quote.degraded_reason and quote.degraded_reason not in seen_degradations:
                        seen_degradations.add(quote.degraded_reason)
                        if len(degradations) < _MAX_DEGRADATIONS:
                            degradations.append(quote.degraded_reason)
                    commission_total += quote.commission
                    slippage_total += quote.slippage
                    trade_rows.append(
                        {
                            "date": date,
                            "asset": asset,
                            "side": "buy" if delta > 0 else "sell",
                            "traded_weight": delta,
                            "notional": abs(delta) * nav,
                            "commission": quote.commission,
                            "slippage": quote.slippage,
                            "cost": quote.total,
                        }
                    )
                period_cost = commission_total + slippage_total
                cost_rows.append(
                    {
                        "date": date,
                        "commission": commission_total,
                        "slippage": slippage_total,
                        "total": period_cost,
                        "turnover": turnover,
                    }
                )
                traded_dates.append(date)
            held = target.copy()

        held_rows.append(held.copy())
        period = returns_matrix[position]
        gross = float(held @ period)
        gross_returns.append(gross)
        net = gross - period_cost
        net_returns.append(net)
        nav = nav * (1.0 + net)
        nav_path.append(nav)

        grown = held * (1.0 + period)
        total = grown.sum()
        held = grown / total if abs(total) > _TRADE_EPS else grown

    nav_series = pd.Series(nav_path, index=index, name="nav")
    trades = (
        pd.DataFrame(trade_rows, columns=list(TRADE_COLUMNS))
        if trade_rows
        else empty_trades()
    )
    costs = (
        pd.DataFrame(cost_rows, columns=list(COST_COLUMNS))
        if cost_rows
        else empty_costs()
    )
    held_frame = pd.DataFrame(held_rows, index=index, columns=assets)

    return RunResult(
        returns=pd.Series(net_returns, index=index, name="portfolio"),
        gross_returns=pd.Series(gross_returns, index=index, name="gross"),
        nav=nav_series,
        weights=held_frame,
        targets=schedule,
        trades=trades,
        costs=costs,
        rebalance_dates=pd.DatetimeIndex(traded_dates),
        meta=build_meta(
            spec,
            nav=nav_series,
            trades=trades,
            costs=costs,
            weights=held_frame,
            degradations=tuple(degradations),
            notes=notes,
        ),
    )


__all__ = ["run_backtest"]
