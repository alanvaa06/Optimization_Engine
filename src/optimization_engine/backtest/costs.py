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

Participation — the denominator of that square root — can come from either of
two places, and the choice is the whole reason volume is optional here.

* **Fixed.** A single number: "I can trade 5% of the book in one name in one
  period without moving it." It needs no volume data at all, which is what
  makes an index backtest possible: an index has no volume, has never had
  volume, and never will. The number is an assumption, and treating it as one
  is honest.
* **From ADV.** Trailing dollar volume times the share of it you are willing
  to be, divided by NAV. This is the real thing, and it is the only setting
  under which capacity is genuinely measured rather than assumed — the same
  strategy's costs now rise as the fund grows and as a name's turnover dries
  up. It needs a volume panel, which only some providers publish.

Choosing ADV without a volume panel does not fail. Each affected trade falls
back to the fixed rate and records why, so the result is still produced and
the run log says exactly which assumption it rests on.

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
        participation: The fraction of the *current* book that the asset's
            trailing traded volume supports in one period, or ``None`` when
            no volume panel was supplied. It is a fraction of NAV rather than
            a fixed quantity because that is how capacity actually behaves:
            the same dollar depth is a large trade for a big fund and a
            rounding error for a small one.
    """

    volatility: float | None = None
    participation: float | None = None


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
        """Commission plus slippage, as a fraction of NAV."""
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

    def participation_lookback(self) -> int:
        """Trailing periods of traded volume the model needs, or ``0``.

        Zero — the default for every model that does not price from ADV — is
        what lets the runner skip the volume machinery entirely, and what lets
        a universe with no volume run unchanged.
        """
        ...

    def charge(
        self, *, asset: str, traded_weight: float, context: MarketContext
    ) -> CostQuote:
        """Cost of trading ``traded_weight`` of NAV in ``asset``.

        Args:
            asset: The asset being traded.
            traded_weight: Change in its weight, as a fraction of NAV. Signed;
                implementations charge on the magnitude.
            context: Trailing volatility and participation for this asset on this
                decision date. Either may be ``None``.

        Returns:
            A :class:`CostQuote`, in fractions of NAV, with ``degraded_reason``
            set whenever a component could not be computed.
        """
        ...


@dataclass(frozen=True)
class ZeroCost:
    """Free trading. Honest only as a deliberate upper bound on performance."""

    def volatility_lookback(self) -> int:
        """Always ``0``: free trading needs no context."""
        return 0

    def participation_lookback(self) -> int:
        """Always ``0``: free trading needs no volume panel."""
        return 0

    def charge(
        self, *, asset: str, traded_weight: float, context: MarketContext
    ) -> CostQuote:
        """An empty quote. Nothing is ever charged.

        Args:
            asset: Ignored.
            traded_weight: Ignored.
            context: Ignored.

        Returns:
            A :class:`CostQuote` of zero commission and zero slippage.
        """
        return CostQuote()


@dataclass(frozen=True)
class LinearCost:
    """Commission and spread in basis points of the traded notional."""

    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def volatility_lookback(self) -> int:
        """Always ``0``: a linear charge needs no volatility estimate."""
        return 0

    def participation_lookback(self) -> int:
        """Always ``0``: a linear charge does not price from volume."""
        return 0

    def charge(
        self, *, asset: str, traded_weight: float, context: MarketContext
    ) -> CostQuote:
        """Commission and spread, both linear in the traded notional.

        Args:
            asset: Ignored. Rates here are universe-wide.
            traded_weight: Change in the asset's weight, as a fraction of NAV.
                Signed; only its magnitude is charged.
            context: Ignored.

        Returns:
            A :class:`CostQuote` whose commission is
            ``|traded_weight| · commission_bps / 10⁴`` and whose slippage is the
            same expression in ``slippage_bps`` — both fractions of NAV.
        """
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
    #: ``"fixed"`` uses :attr:`participation` for every trade; ``"adv"`` reads
    #: the participation the trailing volume panel supports and falls back to
    #: :attr:`participation` when there is none.
    participation_source: str = "fixed"
    adv_lookback: int = 21

    def volatility_lookback(self) -> int:
        """Trailing periods of returns needed to estimate ``sigma``.

        Returns:
            :attr:`lookback`, in periods.
        """
        return int(self.lookback)

    def participation_lookback(self) -> int:
        """Trailing periods of traded volume needed, or ``0``.

        Returns:
            :attr:`adv_lookback` when participation is read from volume, and ``0``
            under the fixed-rate source — which is what lets the runner skip the
            volume machinery for a universe that has none.
        """
        return int(self.adv_lookback) if self.participation_source == "adv" else 0

    def charge(
        self, *, asset: str, traded_weight: float, context: MarketContext
    ) -> CostQuote:
        """Linear costs plus square-root impact, in fractions of NAV.

        Impact is ``eta · sigma · sqrt(traded / participation)`` applied to the
        traded notional, so it grows with the square root of participation:
        doubling a trade costs more than twice as much.

        Degradation is reported, never silent. A missing volatility estimate
        drops impact to zero and says so; an ADV-priced model with no volume for
        this asset falls back to the fixed rate and says that too. Either way the
        reason travels on the quote and into the run's degradation notes.

        Args:
            asset: Name, used only to make the degradation message specific.
            traded_weight: Change in the asset's weight, as a fraction of NAV.
                Signed; only its magnitude is charged.
            context: Trailing volatility and participation for this asset on this
                decision date. Either may be ``None``.

        Returns:
            A :class:`CostQuote` with impact folded into ``slippage``, and
            ``degraded_reason`` set whenever a component was dropped or fell back.
        """
        traded = abs(float(traded_weight))
        commission = traded * self.commission_bps / _BPS
        slippage = traded * self.slippage_bps / _BPS
        if traded <= 0.0:
            return CostQuote(commission=commission, slippage=slippage)

        sigma = context.volatility
        if sigma is None or not math.isfinite(sigma):
            return CostQuote(
                commission=commission,
                slippage=slippage,
                degraded_reason=(
                    f"square-root impact degraded to zero for {asset}: "
                    "insufficient history to estimate volatility"
                ),
            )

        participation, reason = self._participation_for(asset, context)
        if participation is None:
            return CostQuote(
                commission=commission,
                slippage=slippage,
                degraded_reason=reason,
            )

        impact = self.eta * sigma * math.sqrt(traded / participation)
        return CostQuote(
            commission=commission,
            slippage=slippage + traded * impact,
            degraded_reason=reason,
        )

    def _participation_for(
        self, asset: str, context: MarketContext
    ) -> tuple[float | None, str | None]:
        """Resolve the participation rate to price this trade against.

        Returns a ``(participation, reason)`` pair. A ``reason`` alongside a
        usable participation means the model fell back — the trade is still
        priced, but the run log records that it rests on the fixed assumption
        rather than on observed volume.
        """
        if self.participation_source != "adv":
            if self.participation <= 0.0:
                return None, (
                    f"square-root impact degraded to zero for {asset}: "
                    "non-positive participation"
                )
            return float(self.participation), None

        observed = context.participation
        if observed is not None and math.isfinite(observed) and observed > 0.0:
            return float(observed), None

        if self.participation > 0.0:
            return float(self.participation), (
                f"ADV-based impact fell back to the fixed participation rate "
                f"for {asset}: no traded volume available"
            )
        return None, (
            f"square-root impact degraded to zero for {asset}: "
            "no traded volume and no positive fallback participation"
        )


def build_cost_model(costs: CostSpec) -> CostModel:
    """Resolve a :class:`CostSpec` to the cheapest model that can express it.

    Args:
        costs: The spec to realize.

    Returns:
        :class:`ZeroCost` for a free spec, :class:`LinearCost` when there is no
        impact term, and :class:`SquareRootImpactCost` otherwise. Picking the
        simplest is not only faster — the cheaper models declare a zero
        lookback, which is what lets the runner skip the trailing-window
        machinery entirely.
    """
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
        participation_source=str(costs.impact_participation_source),
        adv_lookback=int(costs.impact_adv_lookback),
    )


def trailing_volatilities(
    returns: pd.DataFrame, lookback: int, min_observations: int
) -> pd.DataFrame:
    """Per-asset trailing volatility as of each date, using only the past.

    The window ending at ``t`` excludes ``t`` itself: a trade decided on
    ``t`` cannot be priced off that day's own return.

    Args:
        returns: Periodic returns, one column per asset.
        lookback: Window length in periods.
        min_observations: Below this many observations the cell is left NaN.

    Returns:
        A frame shaped like ``returns``, holding per-period volatility. Cells
        with too little history are ``NaN``, which the cost models read as
        "unknown" and degrade on rather than guess.
    """
    if lookback <= 0:
        return pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    shifted = returns.shift(1)
    vol = shifted.rolling(window=lookback, min_periods=int(min_observations)).std(ddof=1)
    return vol.replace([np.inf, -np.inf], np.nan)


def trailing_dollar_volume(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    lookback: int,
    min_observations: int,
) -> pd.DataFrame:
    """Per-asset trailing average traded notional, using only the past.

    ``price × volume`` rather than volume alone, because share counts are not
    comparable across a $4 stock and a $400 one — the quantity that bounds a
    trade is money, not shares.

    Like :func:`trailing_volatilities`, the window ending at ``t`` excludes
    ``t``: a trade decided on ``t`` cannot be sized off that day's own
    turnover. Assets with no volume at all come back as all-NaN columns, which
    the cost model reads as "no ADV here" and prices from its fixed rate
    instead.

    Args:
        prices: Close prices, one column per asset.
        volumes: Traded volume on the same index and columns. Columns absent
            here are treated as having no volume.
        lookback: Trailing periods to average over.
        min_observations: Below this many observations in the window the
            average is not used.

    Returns:
        A frame on ``prices``' index and columns, in the currency the prices
        are quoted in.
    """
    if lookback <= 0:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    aligned = volumes.reindex(index=prices.index, columns=prices.columns)
    notional = prices.astype(float) * aligned.astype(float)
    notional = notional.replace([np.inf, -np.inf], np.nan)
    trailing = (
        notional.shift(1)
        .rolling(window=int(lookback), min_periods=int(min_observations))
        .mean()
    )
    # A zero average is "cannot trade", which is not something a volume panel
    # can actually assert — it means the field was padded. Treat it as absent.
    return trailing.where(trailing > 0.0)


__all__ = [
    "CostModel",
    "CostQuote",
    "LinearCost",
    "MarketContext",
    "SquareRootImpactCost",
    "ZeroCost",
    "build_cost_model",
    "trailing_dollar_volume",
    "trailing_volatilities",
]
