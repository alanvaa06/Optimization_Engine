"""The declarative description of a backtest, and its provenance hash.

Everything a run needs to be reproduced lives in one frozen object. That is
the point of separating the spec from the runner: a :class:`BacktestSpec` is
data, so it can be serialized, diffed, put in a grid, and hashed. Two runs
that carry the same ``spec_hash`` were asked the same question; two that do
not were not, however similar their charts look.

The spec is validated on construction rather than at use. A rebalance
cadence the runner does not know, a negative cost, an execution lag longer
than the sample — those are configuration errors, and the cheapest place to
find them is before any simulation has run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

RebalanceFrequency = Literal[
    "none", "daily", "weekly", "monthly", "quarterly", "annual"
]

#: Pandas offset aliases for each supported rebalance cadence. ``none`` and
#: ``daily`` are handled without resampling: one trade ever, or every period.
FREQUENCY_ALIASES: dict[str, str | None] = {
    "none": None,
    "daily": None,
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


class SpecValidationError(ValueError):
    """A spec that cannot describe a runnable backtest."""


@dataclass(frozen=True)
class CostSpec:
    """What trading costs, in three separable pieces.

    Costs are quoted as fractions of the traded notional and charged against
    NAV, so every figure here is comparable across books of any size.

    The split matters because the pieces behave differently. Commission is
    linear in what you trade and is the same whether you trade a lot or a
    little. Slippage — the half-spread you cross — is also linear, but it is
    a market property rather than a broker one. Impact is neither: it grows
    with the *square root* of participation, which is why doubling the size
    of a trade costs more than twice as much, and why an allocation that
    looks costless at one fund size stops being so at ten times that.

    Attributes:
        commission_bps: One-way broker commission, in basis points of the
            traded notional.
        slippage_bps: One-way spread cost, in basis points of the traded
            notional.
        impact_coefficient: The ``eta`` of the square-root impact law
            ``eta · sigma · sqrt(q / participation)``, where ``q`` is the
            fraction of the book traded in the name and ``sigma`` is that
            asset's per-period volatility. Zero disables impact entirely.
        impact_participation: The fraction of the book that can be traded in
            one name, in one period, without impact — the weight-space
            analogue of average daily volume. Must be positive when
            ``impact_coefficient`` is.
        impact_volatility_lookback: Trailing periods used to estimate the
            per-asset ``sigma`` that scales impact.
        min_impact_observations: Below this many trailing returns the
            volatility estimate is not trustworthy; impact degrades to zero
            for that trade and the run log says so, rather than silently
            inventing a number.
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_coefficient: float = 0.0
    impact_participation: float = 0.05
    impact_volatility_lookback: int = 63
    min_impact_observations: int = 21

    def __post_init__(self) -> None:
        # Normalize on construction so that ``CostSpec(commission_bps=10)`` and
        # ``CostSpec(commission_bps=10.0)`` are the same spec, and therefore
        # carry the same hash. An int that survives into the canonical JSON
        # would make two identical runs look like different ones.
        for name in ("commission_bps", "slippage_bps", "impact_coefficient",
                     "impact_participation"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("impact_volatility_lookback", "min_impact_observations"):
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in ("commission_bps", "slippage_bps", "impact_coefficient"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise SpecValidationError(f"{name} must be non-negative; got {value}.")
        if self.impact_coefficient > 0.0 and self.impact_participation <= 0.0:
            raise SpecValidationError(
                "impact_participation must be positive when impact is enabled; "
                f"got {self.impact_participation}."
            )
        if self.impact_volatility_lookback < 2:
            raise SpecValidationError(
                "impact_volatility_lookback must be at least 2 periods; "
                f"got {self.impact_volatility_lookback}."
            )
        if self.min_impact_observations < 2:
            raise SpecValidationError(
                "min_impact_observations must be at least 2; "
                f"got {self.min_impact_observations}."
            )

    @property
    def total_linear_bps(self) -> float:
        """Commission plus slippage: the part of the cost that is size-blind."""
        return float(self.commission_bps) + float(self.slippage_bps)

    @property
    def has_impact(self) -> bool:
        return float(self.impact_coefficient) > 0.0

    @property
    def is_free(self) -> bool:
        return self.total_linear_bps == 0.0 and not self.has_impact

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CostSpec:
        data = dict(data or {})
        return cls(
            commission_bps=float(data.get("commission_bps", 0.0)),
            slippage_bps=float(data.get("slippage_bps", 0.0)),
            impact_coefficient=float(data.get("impact_coefficient", 0.0)),
            impact_participation=float(data.get("impact_participation", 0.05)),
            impact_volatility_lookback=int(data.get("impact_volatility_lookback", 63)),
            min_impact_observations=int(data.get("min_impact_observations", 21)),
        )

    @classmethod
    def from_bps(cls, transaction_cost_bps: float) -> CostSpec:
        """The one-number cost model, kept for callers that only have one number."""
        return cls(commission_bps=float(transaction_cost_bps))


@dataclass(frozen=True)
class BacktestSpec:
    """A complete, reproducible description of one simulation.

    Attributes:
        frequency: How often the book is traded back to its target weights.
        costs: The cost model. See :class:`CostSpec`.
        execution_lag: Periods between the date a target becomes effective
            and the date it is traded. Zero means the book is rebalanced on
            the close of the decision date itself — free, instantaneous
            execution, which no desk gets. One period is the honest default
            for a daily panel where the decision is taken after the close.
        periods_per_year: Observations per year, for annualizing.
        initial_capital: Starting NAV. Cosmetic for returns; it makes the
            trade and cost frames read in currency rather than in fractions.
        is_out_of_sample: Whether the weights were chosen without seeing the
            returns they are replayed on. The single most important caveat
            attached to any backtest number, so it travels with the spec and
            ends up stamped on the result.
        name: A label carried into the result and the audit log.
    """

    frequency: RebalanceFrequency = "monthly"
    costs: CostSpec = field(default_factory=CostSpec)
    execution_lag: int = 0
    periods_per_year: int = 252
    initial_capital: float = 1.0
    is_out_of_sample: bool = False
    name: str = "backtest"

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_lag", int(self.execution_lag))
        object.__setattr__(self, "periods_per_year", int(self.periods_per_year))
        object.__setattr__(self, "initial_capital", float(self.initial_capital))
        if self.frequency not in FREQUENCY_ALIASES:
            raise SpecValidationError(
                f"Unknown rebalance frequency {self.frequency!r}. "
                f"Available: {sorted(FREQUENCY_ALIASES)}"
            )
        if self.execution_lag < 0:
            raise SpecValidationError(
                f"execution_lag cannot be negative; got {self.execution_lag}. "
                "A negative lag would trade on a decision not yet taken."
            )
        if self.periods_per_year < 1:
            raise SpecValidationError(
                f"periods_per_year must be at least 1; got {self.periods_per_year}."
            )
        if self.initial_capital <= 0.0:
            raise SpecValidationError(
                f"initial_capital must be positive; got {self.initial_capital}."
            )
        if not isinstance(self.costs, CostSpec):
            object.__setattr__(self, "costs", CostSpec.from_dict(dict(self.costs)))

    def with_(self, **changes: Any) -> BacktestSpec:
        """A copy with fields replaced — the way a sweep builds its cells."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency": self.frequency,
            "costs": self.costs.to_dict(),
            "execution_lag": int(self.execution_lag),
            "periods_per_year": int(self.periods_per_year),
            "initial_capital": float(self.initial_capital),
            "is_out_of_sample": bool(self.is_out_of_sample),
            "name": str(self.name),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> BacktestSpec:
        data = dict(data or {})
        return cls(
            frequency=str(data.get("frequency", "monthly")),  # type: ignore[arg-type]
            costs=CostSpec.from_dict(data.get("costs")),
            execution_lag=int(data.get("execution_lag", 0)),
            periods_per_year=int(data.get("periods_per_year", 252)),
            initial_capital=float(data.get("initial_capital", 1.0)),
            is_out_of_sample=bool(data.get("is_out_of_sample", False)),
            name=str(data.get("name", "backtest")),
        )

    def canonical_json(self) -> str:
        """Key-sorted JSON — the thing that actually gets hashed."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def spec_hash(self) -> str:
        """SHA-256 of the canonical form, excluding the cosmetic ``name``.

        Two specs share a hash exactly when they would produce the same
        simulation from the same data. The label is deliberately outside the
        hash: renaming a run does not make it a different run.
        """
        payload = dict(self.to_dict())
        payload.pop("name", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def describe(self) -> str:
        cost = (
            "costless"
            if self.costs.is_free
            else f"{self.costs.total_linear_bps:.1f} bps linear"
            + (f" + sqrt impact (eta={self.costs.impact_coefficient:g})" if self.costs.has_impact else "")
        )
        lag = "same-period fills" if self.execution_lag == 0 else f"{self.execution_lag}-period execution lag"
        sample = "out-of-sample" if self.is_out_of_sample else "in-sample"
        return f"{self.name}: {self.frequency} rebalance, {cost}, {lag} ({sample})"


__all__ = [
    "FREQUENCY_ALIASES",
    "REBALANCE_DESCRIPTIONS",
    "BacktestSpec",
    "CostSpec",
    "RebalanceFrequency",
    "SpecValidationError",
]
