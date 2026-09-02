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

A target dated on a day the market was shut is traded on the next bar, not
quietly deferred to the next calendar rebalance. Every such move is named in
``meta.notes`` — see :func:`_place_schedule_on_bars`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from optimization_engine.backtest.calendar import execution_positions, rebalance_dates
from optimization_engine.backtest.costs import (
    CostModel,
    MarketContext,
    build_cost_model,
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

_LOG = logging.getLogger(__name__)

#: Traded fractions below this are float residue, not orders.
_TRADE_EPS = 1e-12

def _no_lookback() -> int:
    """Default for cost models written before ADV pricing existed."""
    return 0


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


def _place_schedule_on_bars(
    schedule_index: pd.Index, index: pd.DatetimeIndex
) -> tuple[list[pd.Timestamp], dict[str, Any]]:
    """Put every schedule date on a bar, and say what that cost.

    A schedule date that is not itself a bar — a Sunday-stamped weekly
    schedule replayed on a business-day index — used to be no decision at
    all. The target was not lost: the resolver picks the most recent schedule
    row on or before each decision, so it still reached the book at whatever
    calendar mark came next. What was lost was the **timing** the caller
    asked for, silently. Under ``frequency="none"`` there is exactly one mark
    — inception — so a whole weekly schedule collapsed into buy-and-hold.

    Every date is therefore executed on the first bar at or after it. The
    search is ``side="left"``, so a date that is already a bar maps to itself:
    nothing that works today can move. Dates past the last bar are dropped,
    because there is no bar left to fill them on.

    Args:
        schedule_index: The dates the schedule's targets become effective,
            sorted.
        index: The simulation's date index.

    Returns:
        The bar dates that become decisions, and notes naming every date that
        did not land where it was written: ``schedule_dates_moved``
        (``{original: executed}``), ``schedule_dates_dropped`` (past the last
        bar) and ``schedule_dates_collapsed`` (``{bar: [originals]}`` for bars
        that received more than one row — the last one wins, which is what the
        resolver has always done, but silently).
    """
    if not index.is_monotonic_increasing:
        # Nothing can be placed on an unordered calendar. Fall back to exact
        # matches, which is what the runner did before, rather than inventing
        # an ordering the caller did not ask for.
        return [date for date in schedule_index if date in set(index)], {}

    positions = np.searchsorted(
        index.to_numpy(), pd.DatetimeIndex(schedule_index).to_numpy(), side="left"
    )
    decisions: list[pd.Timestamp] = []
    moved: dict[str, str] = {}
    dropped: list[str] = []
    landed: dict[int, list[str]] = {}
    for original, raw_position in zip(schedule_index, positions):
        label = pd.Timestamp(original).isoformat()
        position = int(raw_position)
        if position >= len(index):
            dropped.append(label)
            continue
        executed = index[position]
        landed.setdefault(position, []).append(label)
        if executed != original:
            moved[label] = pd.Timestamp(executed).isoformat()
        decisions.append(executed)

    collapsed = {
        pd.Timestamp(index[position]).isoformat(): originals
        for position, originals in sorted(landed.items())
        if len(originals) > 1
    }
    notes: dict[str, Any] = {}
    if moved:
        notes["schedule_dates_moved"] = moved
    if dropped:
        notes["schedule_dates_dropped"] = dropped
    if collapsed:
        notes["schedule_dates_collapsed"] = collapsed
    if notes:
        _LOG.warning(
            "Weight schedule does not line up with the return index: "
            "%d date(s) moved to the next bar, %d dropped past the last bar, "
            "%d bar(s) received more than one target (the latest wins).",
            len(moved),
            len(dropped),
            len(collapsed),
        )
    return decisions, notes


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
        notes: Free-form annotations to carry on the result metadata. The
            runner adds entries of its own. ``"missing_returns"`` records how
            many periods had a gap somewhere, and — the part worth reading —
            which *held* assets had one, and how often; a missing return is
            replayed as a flat period for that asset alone and never touches
            the rest of the book. ``"schedule_dates_moved"``,
            ``"schedule_dates_dropped"`` and ``"schedule_dates_collapsed"``
            record every target date that did not fall on a bar. None of
            these are hashed: they describe the run, they are not part of
            what it computed.
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

    lookback = int(model.volatility_lookback())
    volatility = None
    if lookback > 0:
        history = returns if context_returns is None else context_returns
        volatility = trailing_volatilities(
            history, lookback, spec.costs.min_impact_observations
        ).reindex(index=index, columns=assets)

    # Traded notional is computed only when a model actually asks for it, so a
    # universe with no volume — the common case for indices — costs nothing
    # and behaves identically to one that was never offered any.
    adv_lookback = int(getattr(model, "participation_lookback", _no_lookback)())
    if adv_lookback > 0 and float(spec.initial_capital) < MIN_ADV_CAPITAL:
        # The spec validates this too, but a caller can hand in a cost model
        # directly and bypass the spec entirely — and this is the one place
        # every route converges on. Left through, an ADV charge against a
        # one-currency-unit book rounds to zero and reads as proof the
        # strategy has no capacity limit.
        raise ValueError(
            "This cost model prices impact from traded volume, which needs a "
            f"real fund size: initial_capital is {spec.initial_capital:g}. "
            "Set it to the capital being deployed."
        )
    adv_notional = None
    if adv_lookback > 0 and volumes is not None and prices is not None:
        adv_notional = trailing_dollar_volume(
            prices, volumes, adv_lookback, spec.costs.min_adv_observations
        ).reindex(index=index, columns=assets)
    # A model supplied directly owns its own share of volume; only fall back
    # to the spec's for the models this library builds from it.
    adv_share = float(
        getattr(model, "adv_share", spec.costs.impact_adv_share)
    )

    marks = rebalance_dates(index, spec.frequency)
    # A schedule date is always a decision: the walk-forward runner only emits
    # one when the optimizer has genuinely re-solved, so ignoring it would
    # silently discard the re-solve. A date that is not a bar is executed on
    # the next one rather than waiting for the next calendar mark.
    schedule_marks, schedule_notes = _place_schedule_on_bars(schedule.index, index)
    decisions = pd.DatetimeIndex(sorted(set(marks).union(schedule_marks)))
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
    missing_held: dict[str, int] = {}
    missing_periods = 0
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
        missing = np.isnan(period)
        if missing.any():
            # A missing return is a flat period, not a poison pill: an asset
            # with no print — not yet listed, delisted, a holiday elsewhere —
            # contributes nothing, and the rest of the book is unaffected.
            # Left as NaN, ``0 * NaN`` would turn the whole NAV path NaN from
            # here on, even for a name the book never held. A missing return
            # on a name that *is* held is recorded on the run so it can be
            # read as the data gap it is.
            missing_periods += 1
            for asset_position in np.flatnonzero(missing & (np.abs(held) > _TRADE_EPS)):
                name = assets[asset_position]
                missing_held[name] = missing_held.get(name, 0) + 1
            period = np.where(missing, 0.0, period)
        gross = float(held @ period)
        gross_returns.append(gross)
        net = gross - period_cost
        net_returns.append(net)
        nav = nav * (1.0 + net)
        nav_path.append(nav)

        # Drift: each position grows with its own return while the book as a
        # whole grows with the portfolio return, so the new weight is the
        # position's share of the *grown* book. The denominator is the book's
        # growth, not the sum of the positions — those only coincide when the
        # book is fully invested. Dividing by the positions' sum would silently
        # convert any cash residual into positions after one period, and turn
        # a 60/−40 long-short book into one levered many times over.
        grown = held * (1.0 + period)
        growth = 1.0 + gross
        held = grown / growth if growth > _TRADE_EPS else grown

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
    run_notes = dict(notes or {})
    run_notes.update(schedule_notes)
    if missing_periods:
        run_notes["missing_returns"] = {
            "periods": int(missing_periods),
            "held_assets": {k: int(v) for k, v in sorted(missing_held.items())},
        }

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
            notes=run_notes,
        ),
    )


__all__ = ["run_backtest"]
