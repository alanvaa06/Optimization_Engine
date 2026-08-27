"""Cost models: pure functions from a trade to what it cost.

The models here are deliberately small and stateless. Each takes one trade —
an asset, the fraction of the book being traded in it, and the market context
as of the decision date — and returns a commission and a slippage charge, both
as fractions of NAV. Nothing reads a file, nothing remembers a previous call,
and the same inputs always produce the same output, which is what lets the
runner above be deterministic and the sweep below be parallelizable.

Two families:

* **Linear.** Commission and half-spread, quoted in basis points of traded
  notional. Cost is proportional to what you trade. This is the model almost
  every published backtest uses, usually as a single number.
* **Square-root impact.** ``eta · sigma · sqrt(q / participation)``, the
  Almgren law expressed in weight space: ``q`` is the fraction of the book
  traded in the name, ``participation`` is the fraction that can be traded
  without moving the price, and ``sigma`` is the asset's per-period
  volatility. Cost per unit traded *grows* with size, so this is the only
  model in which capacity is visible at all: an allocation that is free at
  fund size 1× need not be at 10×.

When impact cannot be computed — too little history to estimate ``sigma`` —
the trade degrades to the linear charge and the reason is recorded on the run
log. It is never silently replaced by zero, and it never raises: a missing
estimate is a fact about the data, not a failure of the simulation.

References:
    Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005). "Direct Estimation
    of Equity Market Impact". *Risk* 18(7).

    Grinold, R. and Kahn, R. (2000). *Active Portfolio Management*, ch. 16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from optimization_engine.backtest.spec import CostSpec

_BPS = 10_000.0


@dataclass(frozen=True)
class MarketContext:
    """What a cost model may know about an asset on a decision date.

    Attributes:
        volatility: Trailing per-period volatility, or ``None`` when there is
            not enough history to estimate it.
    """

    volatility: float | None = None


@dataclass(frozen=True)
class CostQuote:
    """What one trade costs, split into its economically distinct pieces.

    Attributes:
        commission: Broker charge, as a fraction of NAV.
        slippage: Spread plus impact, as a fraction of NAV.
        degraded_reason: Set when a component could not be computed and was
            dropped. Surfaced on the run log so a cheap-looking backtest
            cannot hide a cost model that quietly switched itself off.
    """

    commission: float = 0.0
    slippage: float = 0.0
    degraded_reason: str | None = None

    @property
    def total(self) -> float:
        return float(self.commission) + float(self.slippage)


class CostModel(Protocol):
    """``charge(asset, traded_weight, context) -> CostQuote``, in NAV fractions."""

    def volatility_lookback(self) -> int:
        """Trailing periods of context the model needs, or ``0`` for none.

        The model owns its requirement so the runner does not have to know
        which models are context-hungry: it asks, and computes exactly the
        windows it is asked for.
        """
        ...

    def charge(
        self, *, asset: str, traded_weight: float, context: MarketContext
    ) -> CostQuote:
        """Cost of trading ``traded_weight`` of NAV in ``asset``."""
        ...


@dataclass(frozen=True)
class ZeroCost:
    """Free trading. Honest only as a deliberate upper bound on performance."""

    def volatility_lookback(self) -> int:
        return 0

    def charge(
        self, *, asset: str, traded_weight: float, context: MarketContext
    ) -> CostQuote:
        return CostQuote()


@dataclass(frozen=True)
class LinearCost:
    """Commission and spread in basis points of the traded notional."""

    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def volatility_lookback(self) -> int:
        return 0

    def charge(
        self, *, asset: str, traded_weight: float, context: MarketContext
    ) -> CostQuote:
        traded = abs(float(traded_weight))
        return CostQuote(
            commission=traded * self.commission_bps / _BPS,
            slippage=traded * self.slippage_bps / _BPS,
        )


@dataclass(frozen=True)
class SquareRootImpactCost:
    """Linear costs plus market impact that grows with the square root of size.

    Impact in basis points is ``eta · sigma · sqrt(q / participation) · 10⁴``
    and is charged on top of the linear slippage, so the same object covers
    both the small trades where the spread dominates and the large ones where
    it does not.
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    eta: float = 0.0
    participation: float = 0.05
    lookback: int = 63
    min_observations: int = 21

    def volatility_lookback(self) -> int:
        return int(self.lookback)

    def charge(
        self, *, asset: str, traded_weight: float, context: MarketContext
    ) -> CostQuote:
        traded = abs(float(traded_weight))
        commission = traded * self.commission_bps / _BPS
        slippage = traded * self.slippage_bps / _BPS
        if traded <= 0.0:
            return CostQuote(commission=commission, slippage=slippage)
        sigma = context.volatility
        if sigma is None or not math.isfinite(sigma) or self.participation <= 0.0:
            reason = (
                f"square-root impact degraded to zero for {asset}: "
                "insufficient history to estimate volatility"
                if sigma is None or not math.isfinite(sigma)
                else f"square-root impact degraded to zero for {asset}: "
                "non-positive participation"
            )
            return CostQuote(
                commission=commission, slippage=slippage, degraded_reason=reason
            )
        impact = self.eta * sigma * math.sqrt(traded / self.participation)
        return CostQuote(commission=commission, slippage=slippage + traded * impact)


def build_cost_model(costs: CostSpec) -> CostModel:
    """Resolve a :class:`CostSpec` to the cheapest model that can express it."""
    if costs.is_free:
        return ZeroCost()
    if not costs.has_impact:
        return LinearCost(
            commission_bps=float(costs.commission_bps),
            slippage_bps=float(costs.slippage_bps),
        )
    return SquareRootImpactCost(
        commission_bps=float(costs.commission_bps),
        slippage_bps=float(costs.slippage_bps),
        eta=float(costs.impact_coefficient),
        participation=float(costs.impact_participation),
        lookback=int(costs.impact_volatility_lookback),
        min_observations=int(costs.min_impact_observations),
    )


def trailing_volatilities(
    returns: pd.DataFrame, lookback: int, min_observations: int
) -> pd.DataFrame:
    """Per-asset trailing volatility as of each date, using only the past.

    The window ending at ``t`` excludes ``t`` itself: a trade decided on
    ``t`` cannot be priced off that day's own return. Cells with fewer than
    ``min_observations`` are ``NaN``, which the models read as "unknown" and
    degrade on rather than guess.
    """
    if lookback <= 0:
        return pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    shifted = returns.shift(1)
    vol = shifted.rolling(window=lookback, min_periods=int(min_observations)).std(ddof=1)
    return vol.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "CostModel",
    "CostQuote",
    "LinearCost",
    "MarketContext",
    "SquareRootImpactCost",
    "ZeroCost",
    "build_cost_model",
    "trailing_volatilities",
]
