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
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, extras=(),
    ),
    "min_variance": MethodRequirements(
        name="min_variance",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, extras=(),
    ),
    "max_sharpe": MethodRequirements(
        name="max_sharpe",
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, extras=(),
    ),
    "risk_parity": MethodRequirements(
        name="risk_parity",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        # NB: flips to True + bounds_mode="constrained" once Task 7 lands a
        # CVXPY reformulation that honours per-asset and group bounds natively.
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, extras=(_RISK_BUDGET,),
    ),
    "hrp": MethodRequirements(
        name="hrp",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, extras=(_HRP_LINKAGE,),
    ),
    "black_litterman": MethodRequirements(
        name="black_litterman",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True,
        extras=(_BL_VIEWS, _BL_VIEW_CONFIDENCES, _BL_TAU, _BL_MARKET_CAPS),
    ),
    "cvar": MethodRequirements(
        name="cvar",
        requires_mu=False, requires_cov=False, requires_returns=True,
        supports_target_return=True, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, extras=(_CVAR_ALPHA,),
    ),
    "max_diversification": MethodRequirements(
        name="max_diversification",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, extras=(),
    ),
    "equal_weight": MethodRequirements(
        name="equal_weight",
        requires_mu=False, requires_cov=False, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, extras=(),
    ),
    "inverse_vol": MethodRequirements(
        name="inverse_vol",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, extras=(),
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
