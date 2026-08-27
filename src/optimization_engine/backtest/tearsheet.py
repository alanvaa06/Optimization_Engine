"""One object that answers the questions a backtest is asked in a meeting.

The frames a run produces are the raw material; a tearsheet is the reading.
It puts performance, drawdowns, costs, position statistics, and — when the
run came out of a search — the selection-bias correction side by side, so the
Sharpe ratio never appears without the three things that qualify it: what it
cost to trade, how much of it survives out of sample, and how many
configurations were tried before this one.

Assembly is deliberately lazy about what it is given. A tearsheet built from
an in-sample run says so and skips the out-of-sample panel rather than
inventing one. A tearsheet with no sweep behind it reports no deflation. The
absence is stated, not filled in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from optimization_engine.backtest.positions import (
    PositionStats,
    compute_position_stats,
    episodes_frame,
    position_episodes,
)
from optimization_engine.backtest.results import RunResult
from optimization_engine.backtest.tca import TcaPanel, compute_tca, cost_by_asset


@dataclass
class Tearsheet:
    """The assembled reading of one run.

    Attributes:
        run: The run this describes.
        performance: Summary statistics of the net return stream.
        drawdowns: The worst drawdowns, with their dates and recovery.
        tca: The cost panel.
        costs_by_asset: Where the cost was concentrated.
        positions: Round-trip statistics over position episodes.
        episodes: The episodes themselves.
        deflated_sharpe: Selection-bias correction, when trials were counted.
        overfitting: CSCV report, when a grid was run.
        caveats: What a reader must know before quoting any number here.
    """

    run: RunResult
    performance: pd.DataFrame
    drawdowns: pd.DataFrame
    tca: TcaPanel
    costs_by_asset: pd.DataFrame
    positions: PositionStats
    episodes: pd.DataFrame
    deflated_sharpe: Any = None
    overfitting: Any = None
    caveats: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Every panel as a frame, ready for the Excel exporter."""
        frames = {
            "performance": self.performance,
            "drawdowns": self.drawdowns,
            "tca": self.tca.to_frame(),
            "costs_by_asset": self.costs_by_asset,
            "positions": self.positions.to_frame(),
            "episodes": self.episodes,
            "weights": self.run.weights,
            "nav": self.run.nav.to_frame("nav"),
            "trades": self.run.trades,
            "caveats": pd.DataFrame({"caveat": list(self.caveats)}),
        }
        return frames

    def describe(self) -> str:
        lines = [self.run.describe(), self.tca.describe()]
        if self.deflated_sharpe is not None:
            lines.append(self.deflated_sharpe.describe())
        if self.overfitting is not None:
            lines.append(self.overfitting.describe())
        lines.extend(f"Caveat: {caveat}" for caveat in self.caveats)
        return "\n".join(lines)


def _caveats(run: RunResult, deflated: Any, overfitting: Any) -> tuple[str, ...]:
    caveats: list[str] = []
    if not run.meta.is_out_of_sample:
        caveats.append(
            "These weights were chosen knowing these returns. The figures "
            "describe a fit, not a track record."
        )
    if run.meta.spec.get("costs", {}).get("commission_bps", 0.0) == 0.0 and run.meta.spec.get(
        "costs", {}
    ).get("slippage_bps", 0.0) == 0.0:
        caveats.append(
            "Trading was modelled as free. Every turnover figure below is a "
            "cost this backtest did not charge."
        )
    if int(run.meta.spec.get("execution_lag", 0)) == 0:
        caveats.append(
            "Orders fill on the close of the decision date. A real desk trades "
            "at least one period later, at a price it has not seen."
        )
    if run.meta.degradations:
        caveats.append(
            f"{len(run.meta.degradations)} cost component(s) could not be computed "
            "and were charged as zero; the reported cost is a lower bound."
        )
    if deflated is None:
        caveats.append(
            "No trial count was supplied, so the Sharpe ratio is undeflated. If "
            "this configuration was chosen from several, it is optimistic."
        )
    if overfitting is not None and getattr(overfitting, "pbo", 0.0) > 0.5:
        caveats.append(
            "The in-sample winner lands below the out-of-sample median more "
            "often than not: the selection is no better than picking at random."
        )
    return tuple(caveats)


def build_tearsheet(
    run: RunResult,
    returns: pd.DataFrame,
    *,
    riskfree_rate: float = 0.0,
    n_trials: int | None = None,
    trial_sharpes: pd.Series | None = None,
    overfitting: Any = None,
    top_drawdowns: int = 5,
) -> Tearsheet:
    """Assemble the full reading of a run.

    Args:
        run: The simulation to describe.
        returns: The asset returns it was replayed on, for position episodes.
        riskfree_rate: Annual rate for the Sharpe and Sortino ratios.
        n_trials: How many configurations were tried before this one. Supply
            it and the Sharpe gets deflated; leave it out and the tearsheet
            says the number is undeflated rather than pretending otherwise.
        trial_sharpes: The trials' annualized Sharpes, when available. Their
            dispersion is what the deflation actually uses.
        overfitting: A pre-computed CSCV report from a sweep.
        top_drawdowns: How many drawdown episodes to table.
    """
    from optimization_engine.analytics.risk import drawdown_table

    performance = run.summary(riskfree_rate=riskfree_rate)
    drawdowns = drawdown_table(run.returns, top=top_drawdowns)

    deflated = None
    if n_trials is not None and n_trials >= 1:
        from optimization_engine.analytics.selection import deflated_sharpe_ratio

        try:
            deflated = deflated_sharpe_ratio(
                run.returns,
                n_trials=int(n_trials),
                trial_sharpes=trial_sharpes,
                riskfree_rate=riskfree_rate,
                periods_per_year=run.periods_per_year,
            )
        except Exception:  # noqa: BLE001 — a missing correction is reported, not raised
            deflated = None

    episodes = position_episodes(run.weights, returns)
    return Tearsheet(
        run=run,
        performance=performance,
        drawdowns=drawdowns,
        tca=compute_tca(run),
        costs_by_asset=cost_by_asset(run),
        positions=compute_position_stats(run, returns),
        episodes=episodes_frame(episodes),
        deflated_sharpe=deflated,
        overfitting=overfitting,
        caveats=_caveats(run, deflated, overfitting),
        metadata={
            "spec_hash": run.meta.spec_hash,
            "result_hash": run.meta.result_hash,
            "n_trials": n_trials,
        },
    )


__all__ = ["Tearsheet", "build_tearsheet"]
