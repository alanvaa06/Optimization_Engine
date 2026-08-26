"""Per-method input/support metadata for the optimizer registry.

Single source of truth shared by the engine (validation) and the Streamlit
UI (which fields to enable). Adding a new optimizer means: register it in
``factory._REGISTRY`` AND add a ``MethodRequirements`` entry here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ExtraKind = Literal["per_asset", "scalar", "choice", "view_table", "market_caps"]
BoundsMode = Literal["hard", "soft_iterated", "constrained"]


@dataclass(frozen=True)
class ExtraInput:
    key: str
    label: str
    kind: ExtraKind
    required: bool
    help: str
    default: Any | None = None
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MethodRequirements:
    """What one optimizer needs, supports, and assumes.

    Beyond the machine-readable capability flags, each entry carries the
    prose an analyst needs to choose between methods: a one-line summary,
    the situation it suits, and the assumptions it is buying into. The UI
    renders these next to the method picker so the choice is informed rather
    than alphabetical.
    """

    name: str
    requires_mu: bool
    requires_cov: bool
    requires_returns: bool
    supports_target_return: bool
    supports_target_volatility: bool
    supports_risk_aversion: bool
    supports_risk_free_rate: bool
    supports_group_bounds: bool
    bounds_mode: BoundsMode
    supports_frontier: bool
    extras: tuple[ExtraInput, ...] = field(default_factory=tuple)
    supports_turnover: bool = False
    label: str = ""
    summary: str = ""
    when_to_use: str = ""
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    risk_measure: str = "volatility"

    @property
    def display_name(self) -> str:
        return self.label or self.name.replace("_", " ").title()

    @property
    def bounds_note(self) -> str:
        """Plain-English description of how faithfully bounds are honoured."""
        return {
            "hard": "Weight bounds and group budgets are enforced inside the solve.",
            "constrained": (
                "Bounds are enforced inside the solve; tiny numerical drift is "
                "clipped afterwards."
            ),
            "soft_iterated": (
                "This method allocates first and applies bounds by projection "
                "afterwards, so a binding bound moves the result away from the "
                "method's own answer."
            ),
        }[self.bounds_mode]


_RISK_BUDGET = ExtraInput(
    key="risk_budget", label="Risk budget",
    kind="per_asset", required=False,
    help="Each asset's target share of total variance (sums to 1).",
)
_BL_VIEWS = ExtraInput(
    key="bl_views", label="Black-Litterman views",
    kind="view_table", required=False,
    help="Asset → annualized expected return.",
)
_BL_VIEW_CONFIDENCES = ExtraInput(
    key="bl_view_confidences", label="View confidences (Ω diagonal)",
    kind="view_table", required=False,
    help="Variance of each view's error term. Defaults to tau · σ_i².",
)
_BL_TAU = ExtraInput(
    key="bl_tau", label="Tau (prior uncertainty scale)",
    kind="scalar", required=False, default=0.05,
    help="Scales the prior covariance in the BL posterior.",
)
_BL_MARKET_CAPS = ExtraInput(
    key="bl_market_caps", label="Market caps / weights",
    kind="market_caps", required=False,
    help="Equilibrium market portfolio. Empty → equal weights.",
)
_CVAR_ALPHA = ExtraInput(
    key="cvar_alpha", label="CVaR tail probability α",
    kind="scalar", required=False, default=0.05,
    help="0.05 ⇒ 95% CVaR.",
)
_HRP_LINKAGE = ExtraInput(
    key="hrp_linkage", label="HRP linkage method",
    kind="choice", required=False, default="single",
    choices=("single", "average", "complete", "ward"),
    help="Hierarchical clustering linkage rule.",
)


REQUIREMENTS: dict[str, MethodRequirements] = {
    "mean_variance": MethodRequirements(
        name="mean_variance",
        label="Mean-variance (Markowitz)",
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, supports_turnover=True, extras=(),
        summary=(
            "Trade expected return against variance, subject to every "
            "constraint you set."
        ),
        when_to_use=(
            "You have expected returns you are willing to defend, and you want "
            "an explicit return or volatility target."
        ),
        assumptions=(
            "Investors care only about the mean and variance of returns.",
            "The expected-return vector is accurate — results are famously "
            "sensitive to it, so small estimation errors produce very "
            "different portfolios.",
            "The covariance matrix is stable over the holding period.",
        ),
    ),
    "min_variance": MethodRequirements(
        name="min_variance",
        label="Global minimum variance",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, supports_turnover=True, extras=(),
        summary="The lowest-variance portfolio the constraints allow.",
        when_to_use=(
            "You do not trust any expected-return estimate. This is the one "
            "mean-variance portfolio that does not need one."
        ),
        assumptions=(
            "Only risk matters; expected returns are ignored entirely.",
            "The covariance matrix is well estimated — with T/N below ~10 even "
            "this portfolio is unstable.",
        ),
    ),
    "max_sharpe": MethodRequirements(
        name="max_sharpe",
        label="Maximum Sharpe (tangency)",
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, supports_turnover=False, extras=(),
        summary="The portfolio with the highest excess return per unit of risk.",
        when_to_use=(
            "You want the tangency portfolio, and you will size the risk "
            "separately by blending it with cash."
        ),
        assumptions=(
            "Expected returns are accurate — the tangency portfolio is the "
            "most estimation-sensitive point on the whole frontier.",
            "You can borrow and lend at the risk-free rate.",
            "At least one asset earns more than the risk-free rate.",
            "A turnover budget cannot be imposed on this solve.",
        ),
    ),
    "risk_parity": MethodRequirements(
        name="risk_parity",
        label="Risk parity / risk budgeting",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="constrained",
        supports_frontier=False, supports_turnover=False,
        extras=(_RISK_BUDGET,),
        summary=(
            "Size positions so each asset contributes its target share of "
            "total risk."
        ),
        when_to_use=(
            "You want risk, not capital, spread evenly — the standard answer "
            "when a 60/40 book's equity sleeve dominates its risk."
        ),
        assumptions=(
            "Expected returns are ignored; only the covariance matters.",
            "Equalizing risk contributions is a good proxy for a good "
            "portfolio — true when Sharpe ratios are similar across assets.",
            "Weights must be long-only for risk contributions to be well "
            "defined.",
        ),
    ),
    "hrp": MethodRequirements(
        name="hrp",
        label="Hierarchical Risk Parity",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, supports_turnover=False,
        extras=(_HRP_LINKAGE,),
        summary=(
            "Cluster assets by correlation, then split risk down the tree — "
            "no matrix inversion anywhere."
        ),
        when_to_use=(
            "Many assets relative to your history, or a covariance matrix that "
            "is close to singular. This is the robust choice when T/N is small."
        ),
        assumptions=(
            "The correlation hierarchy it finds is economically meaningful — "
            "check the reported clusters.",
            "Group budgets cannot be enforced; the method has its own "
            "hierarchy and will disagree with a hand-specified one.",
            "Weight bounds are applied by projection after allocation.",
        ),
    ),
    "black_litterman": MethodRequirements(
        name="black_litterman",
        label="Black-Litterman",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, supports_turnover=True,
        extras=(_BL_VIEWS, _BL_VIEW_CONFIDENCES, _BL_TAU, _BL_MARKET_CAPS),
        summary=(
            "Start from the returns the market already implies, then tilt only "
            "where you hold a view."
        ),
        when_to_use=(
            "You have a few specific opinions and want the rest of the "
            "portfolio to stay at equilibrium instead of reacting to noisy "
            "historical means."
        ),
        assumptions=(
            "The market-cap portfolio you supply is in equilibrium.",
            "Views are expressed with an honest confidence — an over-confident "
            "view will dominate the posterior.",
            "The risk-aversion coefficient δ sets the level of the prior; "
            "calibrate it to an observed market Sharpe rather than guessing.",
        ),
    ),
    "cvar": MethodRequirements(
        name="cvar",
        label="Mean-CVaR (expected shortfall)",
        requires_mu=False, requires_cov=False, requires_returns=True,
        supports_target_return=True, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, supports_turnover=True,
        extras=(_CVAR_ALPHA,), risk_measure="CVaR",
        summary=(
            "Minimize the average loss in the worst α of scenarios, straight "
            "from the return history."
        ),
        when_to_use=(
            "Returns are visibly skewed or fat-tailed and variance is the "
            "wrong risk measure — credit, options overlays, EM debt."
        ),
        assumptions=(
            "The historical scenarios represent the tail you actually face.",
            "Enough observations fall in the tail to estimate it: at α = 5% "
            "you are averaging roughly T/20 scenarios.",
            "Scenarios are equally likely and drawn from one regime.",
        ),
    ),
    "max_diversification": MethodRequirements(
        name="max_diversification",
        label="Maximum diversification",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, supports_turnover=False, extras=(),
        summary=(
            "Maximize weighted-average asset volatility divided by portfolio "
            "volatility."
        ),
        when_to_use=(
            "You want correlation benefit to be the objective rather than a "
            "by-product, without taking a view on returns."
        ),
        assumptions=(
            "Expected returns are proportional to volatility — the condition "
            "under which max-diversification is also mean-variance optimal.",
            "The correlation structure is stable; the objective is driven "
            "entirely by it.",
            "A turnover budget cannot be imposed on this solve.",
        ),
    ),
    "equal_weight": MethodRequirements(
        name="equal_weight",
        label="Equal weight (1/N)",
        requires_mu=False, requires_cov=False, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, supports_turnover=False, extras=(),
        summary="Allocate 1/N to every asset.",
        when_to_use=(
            "As the benchmark every other method has to beat. It estimates "
            "nothing, so it has no estimation error to be wrong about."
        ),
        assumptions=(
            "The universe is deliberately constructed — 1/N inherits whatever "
            "concentration is in the asset list itself.",
        ),
    ),
    "inverse_vol": MethodRequirements(
        name="inverse_vol",
        label="Inverse volatility",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, supports_turnover=False, extras=(),
        summary="Weight each asset by the inverse of its own volatility.",
        when_to_use=(
            "A cheap risk-parity approximation when you do not trust the "
            "off-diagonal terms of the covariance matrix."
        ),
        assumptions=(
            "Correlations are ignored entirely — this equals true risk parity "
            "only when every pair is equally correlated.",
        ),
    ),
}


def requirements_for(name: str) -> MethodRequirements:
    """Return the :class:`MethodRequirements` for an optimizer name.

    Raises ``KeyError`` with the list of known names when ``name`` is unknown.
    """
    try:
        return REQUIREMENTS[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown optimizer '{name}'. Available: {sorted(REQUIREMENTS.keys())}"
        ) from e
