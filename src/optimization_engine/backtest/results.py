"""The canonical result bundle a simulation returns.

One run produces one :class:`RunResult`: a fixed set of frames — NAV, net and
gross returns, held weights, targets, trades, costs — plus a
:class:`RunMeta` recording what was asked for and what came back. Everything
downstream (transaction-cost analysis, position statistics, the tearsheet,
the sweep) reads this bundle and nothing else, which is what keeps those
layers from having to know how the loop works.

The result carries a ``result_hash`` alongside the spec's ``spec_hash``. The
pair answers two different questions: the spec hash says *what was asked*,
the result hash says *what came back*. Same spec and same data must give the
same result hash — if it does not, something in the pipeline is not
deterministic, and finding that out cheaply is worth the hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from optimization_engine.backtest.spec import BacktestSpec

#: Rounding applied before hashing. Float arithmetic is deterministic given a
#: fixed operation order, but BLAS threading and platform libm are not
#: guaranteed to agree in the last bits; twelve decimals is far tighter than
#: any decision anyone would take on these numbers, and far looser than the
#: noise those differences introduce.
_HASH_DECIMALS = 12

#: Columns of the trades frame, in canonical order.
TRADE_COLUMNS = (
    "date",
    "asset",
    "side",
    "traded_weight",
    "notional",
    "commission",
    "slippage",
    "cost",
)

#: Columns of the per-date costs frame, in canonical order.
COST_COLUMNS = ("date", "commission", "slippage", "total", "turnover")


@dataclass(frozen=True)
class RunMeta:
    """What the run was asked to do, and what it reported back.

    Attributes:
        spec: The spec, as a plain dict, so the meta serializes on its own.
        spec_hash: Provenance hash of the spec.
        result_hash: Provenance hash of the frames this run produced.
        n_periods: Evaluated periods.
        n_assets: Universe size.
        start: First evaluated date.
        end: Last evaluated date.
        is_out_of_sample: Copied off the spec so a bare result still carries
            the caveat.
        degradations: Reasons any cost component switched itself off. Empty
            is the normal case; non-empty means the reported cost is a lower
            bound on the modelled one.
        notes: Free-form annotations added by whatever produced the run —
            the walk-forward runner records its window geometry here.
    """

    spec: dict[str, Any]
    spec_hash: str
    result_hash: str
    n_periods: int
    n_assets: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    is_out_of_sample: bool
    degradations: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """The frames one simulation produced.

    Attributes:
        returns: Portfolio returns net of costs — the headline series.
        gross_returns: The same series before costs.
        nav: Net asset value, starting at ``spec.initial_capital``.
        weights: Weights actually held at the start of each period, after
            drift. Not the targets: the difference between the two is the
            drift the book was carrying.
        targets: Target weights on each date one became effective.
        trades: One row per asset per traded date, with the cost split.
        costs: Per-date cost totals and turnover.
        rebalance_dates: Dates on which the book was traded back to target.
        meta: See :class:`RunMeta`.
    """

    returns: pd.Series
    gross_returns: pd.Series
    nav: pd.Series
    weights: pd.DataFrame
    targets: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex
    meta: RunMeta

    # -- derived views ------------------------------------------------------

    @property
    def turnover(self) -> pd.Series:
        """One-way turnover on each traded date."""
        if self.costs.empty:
            return pd.Series(dtype=float, name="turnover")
        return pd.Series(
            self.costs["turnover"].values,
            index=pd.DatetimeIndex(self.costs["date"]),
            name="turnover",
        )

    @property
    def cost_series(self) -> pd.Series:
        """Total cost charged on each traded date, as a fraction of NAV."""
        if self.costs.empty:
            return pd.Series(dtype=float, name="cost")
        return pd.Series(
            self.costs["total"].values,
            index=pd.DatetimeIndex(self.costs["date"]),
            name="cost",
        )

    @property
    def total_turnover(self) -> float:
        """Sum of one-way turnover over the whole run, as a fraction of NAV."""
        return float(self.costs["turnover"].sum()) if not self.costs.empty else 0.0

    @property
    def total_cost(self) -> float:
        """Every cost charged over the whole run, as a fraction of NAV."""
        return float(self.costs["total"].sum()) if not self.costs.empty else 0.0

    @property
    def periods_per_year(self) -> int:
        """Annualization basis the run was simulated on, read back from the spec."""
        return int(self.meta.spec.get("periods_per_year", 252))

    @property
    def annualized_turnover(self) -> float:
        """Turnover per year — the number a trading desk actually budgets."""
        years = len(self.returns) / self.periods_per_year
        return float(self.total_turnover / years) if years > 0 else float("nan")

    def wealth(self, starting: float = 1.0) -> pd.Series:
        """The net-of-cost equity curve.

        Args:
            starting: Value at the first period. Defaults to ``1.0``, which makes
                the series read as a growth multiple.

        Returns:
            A series indexed like :attr:`returns`, compounding them from
            ``starting``.
        """
        return starting * (1 + self.returns).cumprod()

    def summary(
        self, periods_per_year: int | None = None, riskfree_rate: float = 0.0
    ) -> pd.DataFrame:
        """Standard performance summary of the net-of-cost return stream.

        Args:
            periods_per_year: Annualization basis. Defaults to the run's own.
            riskfree_rate: Per-period risk-free rate for the ratio metrics.

        Returns:
            A one-row-per-statistic summary frame, computed on the extended set.
        """
        from optimization_engine.analytics.performance import summary_stats

        return summary_stats(
            self.returns.to_frame("portfolio"),
            periods_per_year=periods_per_year or self.periods_per_year,
            riskfree_rate=riskfree_rate,
            extended=True,
        )

    def cost_drag(self, periods_per_year: int | None = None) -> float:
        """Annualized return given up to transaction costs.

        Args:
            periods_per_year: Annualization basis. Defaults to the run's own.

        Returns:
            The gross annualized return minus the net one, as a fraction. This is
            what trading cost, expressed the way a return is.
        """
        from optimization_engine.analytics.performance import annualize_returns

        ppy = periods_per_year or self.periods_per_year
        return float(
            annualize_returns(self.gross_returns, ppy)
            - annualize_returns(self.returns, ppy)
        )

    def describe(self) -> str:
        """One line: sample, length, annualized turnover, total cost, degradations.

        Returns:
            A summary that names any cost degradation rather than letting a
            cheap-looking backtest hide a cost model that switched itself off.
        """
        sample = "out-of-sample" if self.meta.is_out_of_sample else "in-sample"
        degraded = (
            "" if not self.meta.degradations
            else f"; {len(self.meta.degradations)} cost degradation(s)"
        )
        return (
            f"{self.meta.spec.get('name', 'backtest')} — {len(self.returns)} periods "
            f"({sample}), turnover {self.annualized_turnover:.2f}x/yr, "
            f"cost {self.total_cost * 100:.2f}% of NAV{degraded}"
        )


def empty_trades() -> pd.DataFrame:
    """A trades frame with the canonical columns and no rows."""
    return pd.DataFrame({name: pd.Series(dtype=_trade_dtype(name)) for name in TRADE_COLUMNS})


def empty_costs() -> pd.DataFrame:
    """A costs frame with the canonical columns and no rows."""
    frame = pd.DataFrame({name: pd.Series(dtype=float) for name in COST_COLUMNS})
    frame["date"] = pd.Series(dtype="datetime64[ns]")
    return frame[list(COST_COLUMNS)]


def _trade_dtype(column: str) -> str:
    if column == "date":
        return "datetime64[ns]"
    if column in {"asset", "side"}:
        return "object"
    return "float64"


def compute_result_hash(
    nav: pd.Series, trades: pd.DataFrame, costs: pd.DataFrame
) -> str:
    """A deterministic fingerprint of what the run produced.

    Hashes the NAV path and the trade and cost totals — the three things that
    would have to change for the run to mean something different. Values are
    rounded first (see ``_HASH_DECIMALS``) and the rows are consumed in their
    canonical order, so the hash is stable across platforms without being so
    loose that a real change slips past it.

    Args:
        nav: The NAV path.
        trades: The trade frame, in canonical column order.
        costs: The cost frame, in canonical column order.

    Returns:
        A hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    for date, value in nav.items():
        digest.update(
            f"{pd.Timestamp(date).isoformat()}|{round(float(value), _HASH_DECIMALS)}|".encode()
        )
    if not trades.empty:
        ordered = trades.sort_values(["date", "asset", "side"], kind="mergesort")
        for row in ordered.itertuples(index=False):
            digest.update(
                (
                    f"{pd.Timestamp(row.date).isoformat()}|{row.asset}|{row.side}|"
                    f"{round(float(row.traded_weight), _HASH_DECIMALS)}|"
                    f"{round(float(row.cost), _HASH_DECIMALS)}|"
                ).encode()
            )
    if not costs.empty:
        for row in costs.sort_values("date", kind="mergesort").itertuples(index=False):
            digest.update(
                (
                    f"{pd.Timestamp(row.date).isoformat()}|"
                    f"{round(float(row.total), _HASH_DECIMALS)}|"
                    f"{round(float(row.turnover), _HASH_DECIMALS)}|"
                ).encode()
            )
    return digest.hexdigest()


def build_meta(
    spec: BacktestSpec,
    *,
    nav: pd.Series,
    trades: pd.DataFrame,
    costs: pd.DataFrame,
    weights: pd.DataFrame,
    degradations: tuple[str, ...] = (),
    notes: dict[str, Any] | None = None,
) -> RunMeta:
    """Assemble the metadata block, hashing the frames as it goes.

    Args:
        spec: The spec the run was simulated under.
        nav: The NAV path.
        trades: The trade frame.
        costs: The cost frame.
        weights: The realized weight path.
        degradations: Cost-model degradation notes collected during the run.
        notes: Anything else worth recording alongside the result.

    Returns:
        A :class:`RunMeta` carrying the serialized spec, the result hash and
        the degradations — everything needed to tell two runs apart.
    """
    index = pd.DatetimeIndex(weights.index)
    return RunMeta(
        spec=spec.to_dict(),
        spec_hash=spec.spec_hash,
        result_hash=compute_result_hash(nav, trades, costs),
        n_periods=int(len(index)),
        n_assets=int(weights.shape[1]),
        start=index[0] if len(index) else None,
        end=index[-1] if len(index) else None,
        is_out_of_sample=bool(spec.is_out_of_sample),
        degradations=tuple(degradations),
        notes=dict(notes or {}),
    )


def sanitize_weights(weights: pd.Series, assets: list[str]) -> np.ndarray:
    """A weight vector aligned to ``assets``, with missing names at zero.

    Args:
        weights: The solved weights, possibly over a different universe.
        assets: The universe to align to, in order.

    Returns:
        A float array of ``len(assets)``. An asset the solve did not hold
        contributes zero rather than a NaN that would poison the NAV path.
    """
    return weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)


__all__ = [
    "COST_COLUMNS",
    "TRADE_COLUMNS",
    "RunMeta",
    "RunResult",
    "build_meta",
    "compute_result_hash",
    "empty_costs",
    "empty_trades",
    "sanitize_weights",
]
