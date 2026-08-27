"""Factory: build the right optimizer from a config object."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.optimizers import ConfigurationError
from optimization_engine.optimizers.base import BaseOptimizer, PortfolioConstraints
from optimization_engine.optimizers.benchmark_relative import (
    ActiveMeanVarianceOptimizer,
)
from optimization_engine.optimizers.black_litterman import BlackLittermanOptimizer
from optimization_engine.optimizers.cdar import CDaROptimizer
from optimization_engine.optimizers.cvar import CVaROptimizer
from optimization_engine.optimizers.herc import HERCOptimizer
from optimization_engine.optimizers.hrp import HRPOptimizer
from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer
from optimization_engine.optimizers.mean_variance import (
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
)
from optimization_engine.optimizers.naive import (
    EqualWeightOptimizer,
    InverseVolatilityOptimizer,
)
from optimization_engine.optimizers.nco import NCOOptimizer
from optimization_engine.optimizers.requirements import requirements_for
from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

_LOG = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseOptimizer]] = {
    "mean_variance": MeanVarianceOptimizer,
    "active_mean_variance": ActiveMeanVarianceOptimizer,
    "min_variance": MinVarianceOptimizer,
    "max_sharpe": MaxSharpeOptimizer,
    "risk_parity": RiskParityOptimizer,
    "hrp": HRPOptimizer,
    "herc": HERCOptimizer,
    "nco": NCOOptimizer,
    "black_litterman": BlackLittermanOptimizer,
    "cvar": CVaROptimizer,
    "cdar": CDaROptimizer,
    "max_diversification": MaxDiversificationOptimizer,
    "equal_weight": EqualWeightOptimizer,
    "inverse_vol": InverseVolatilityOptimizer,
}


def available_optimizers() -> list[str]:
    return sorted(_REGISTRY.keys())


def constraints_from_config(
    config: EngineConfig, assets: list[str] | None = None
) -> PortfolioConstraints:
    """Translate an :class:`EngineConfig` into the optimizer's constraint object.

    Every constraint the config can express is carried through here —
    including the exposure and turnover settings, which the engine documented
    but previously dropped on the floor between config and solver.

    Args:
        config: The configuration to translate.
        assets: The universe the solve will run over. Needed to expand a
            benchmark that is defined by rule rather than by vector (1/N is
            a different portfolio over a different universe); defaults to the
            assets the config itself names.
    """
    bounds = {k: tuple(v) for k, v in config.bounds.items()}
    group_bounds = {k: tuple(v) for k, v in config.group_bounds.items()}
    return PortfolioConstraints(
        benchmark_weights=config.benchmark_weight_map(assets),
        max_tracking_error=config.max_tracking_error,
        max_active_share=config.max_active_share,
        bounds=bounds,
        groups=dict(config.groups),
        group_bounds=group_bounds,
        fully_invested=config.fully_invested,
        long_only=config.long_only,
        leverage=config.leverage,
        target_return=config.optimizer.target_return,
        target_volatility=config.optimizer.target_volatility,
        previous_weights=(
            dict(config.previous_weights) if config.previous_weights else None
        ),
        turnover_limit=config.turnover_limit,
    )


#: Kept for callers that imported the private name.
_constraints_from_config = constraints_from_config


def _validate(
    spec: OptimizerSpec,
    expected_returns,
    cov_matrix,
    returns,
    constraints_turnover: bool = False,
    constraints: PortfolioConstraints | None = None,
) -> None:
    req = requirements_for(spec.name)
    if constraints is not None:
        validate_benchmark_constraints(spec, constraints)
    if req.requires_mu and (expected_returns is None or len(expected_returns) == 0):
        raise ConfigurationError(
            f"Optimizer '{spec.name}' requires expected_returns; got empty."
        )
    if req.requires_cov and cov_matrix is None:
        raise ConfigurationError(
            f"Optimizer '{spec.name}' requires a covariance matrix; got None."
        )
    if req.requires_returns and returns is None:
        raise ConfigurationError(
            f"Optimizer '{spec.name}' requires a returns DataFrame; got None."
        )
    if not req.supports_target_return and spec.target_return is not None:
        _LOG.warning(
            "Optimizer '%s' does not support target_return; ignoring value %s.",
            spec.name, spec.target_return,
        )
    if not req.supports_target_volatility and spec.target_volatility is not None:
        _LOG.warning(
            "Optimizer '%s' does not support target_volatility; ignoring value %s.",
            spec.name, spec.target_volatility,
        )
    if not req.supports_turnover and constraints_turnover:
        _LOG.warning(
            "Optimizer '%s' cannot enforce a turnover budget; the limit will be "
            "reported but not imposed. Use mean_variance or cvar to bind it.",
            spec.name,
        )


def validate_benchmark_constraints(
    spec: OptimizerSpec, constraints: PortfolioConstraints
) -> None:
    """Check the benchmark-relative settings against what the method can do.

    Called before the feasibility analysis rather than only at solve time: a
    budget with no benchmark to measure it against is a configuration error,
    and the LP that would otherwise hit it first reports it as a solver
    problem.

    Raises:
        ConfigurationError: When the method needs a benchmark and has none, or
            when a limit is set with no benchmark weights to measure against.
    """
    req = requirements_for(spec.name)
    has_limits = (
        constraints.max_tracking_error is not None
        or constraints.max_active_share is not None
    )
    if req.requires_benchmark and not constraints.benchmark_weights:
        raise ConfigurationError(
            f"Optimizer '{spec.name}' optimizes against a benchmark, but none "
            "was set. Choose a benchmark defined by weights, or use "
            "'mean_variance' for the absolute problem."
        )
    if has_limits and not constraints.benchmark_weights:
        raise ConfigurationError(
            "A tracking-error or active-share budget was set without a "
            "benchmark. Both are measured against benchmark *positions*, so "
            "name a benchmark defined by weights — 1/N, a single asset, or a "
            "custom vector. An external index has no holdings in the "
            "investable universe and cannot bound active risk."
        )
    if has_limits and not req.supports_benchmark_limits:
        _LOG.warning(
            "Optimizer '%s' cannot impose a tracking-error or active-share "
            "budget inside its solve; the limit will be reported in the "
            "compliance panel but not enforced. Use mean_variance, "
            "min_variance or active_mean_variance to bind it.",
            spec.name,
        )


def _decode_views(raw):
    """Accept both the flat ``{asset: return}`` form and serialized ``View``s.

    Configs round-trip through YAML, so a relative view arrives as a plain
    mapping with a ``weights`` key rather than as a :class:`View` instance.
    """
    if not raw or isinstance(raw, dict):
        return raw
    from optimization_engine.optimizers.black_litterman import View

    out = []
    for entry in raw:
        if isinstance(entry, View):
            out.append(entry)
            continue
        if not isinstance(entry, dict) or "weights" not in entry:
            raise ConfigurationError(
                "Each Black-Litterman view must be a mapping with a 'weights' "
                f"key; got {entry!r}."
            )
        out.append(
            View(
                weights={str(k): float(v) for k, v in entry["weights"].items()},
                expected_return=float(entry.get("expected_return", 0.0)),
                confidence=(
                    float(entry["confidence"])
                    if entry.get("confidence") is not None
                    else None
                ),
                label=str(entry.get("label", "")),
            )
        )
    return out


def effective_expected_returns(
    config: EngineConfig,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None = None,
) -> pd.Series | None:
    """The expected-return vector the optimizer will *actually* use.

    For most methods this is just the configured vector. Black-Litterman is
    the exception: it discards the supplied returns and optimizes against an
    equilibrium posterior, which sits at a different level entirely. Checking
    a return target against the configured vector therefore passes for
    targets Black-Litterman cannot reach — the feasibility report says
    "feasible" and the solve then comes back infeasible.

    Returns ``None`` when the method needs no expected returns at all.
    """
    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns)
    if config.optimizer.name != "black_litterman":
        return expected_returns

    from optimization_engine.optimizers.black_litterman import (
        black_litterman_posterior,
    )

    assets = list(cov_matrix.columns)
    caps = config.optimizer.bl_market_caps
    if caps:
        market = pd.Series(caps).reindex(assets).fillna(0.0)
        total = float(market.sum())
        market = market / total if total > 0 else pd.Series(1.0 / len(assets), index=assets)
    else:
        market = pd.Series(1.0 / len(assets), index=assets)
    try:
        posterior, _ = black_litterman_posterior(
            cov_matrix,
            market,
            _decode_views(config.optimizer.bl_views),
            config.optimizer.bl_view_confidences,
            tau=config.optimizer.bl_tau,
            risk_aversion=config.optimizer.risk_aversion,
            risk_free_rate=config.optimizer.risk_free_rate,
        )
        posterior.name = "black_litterman_posterior"
        return posterior
    except Exception:
        return expected_returns


def optimizer_factory(
    config: EngineConfig,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    returns: pd.DataFrame | None = None,
    **overrides: Any,
) -> BaseOptimizer:
    """Build an optimizer instance from an :class:`EngineConfig`."""
    spec: OptimizerSpec = config.optimizer
    name = spec.name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {available_optimizers()}"
        )
    cls = _REGISTRY[name]

    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns, name="expected_return")

    constraints = constraints_from_config(config, list(cov_matrix.columns))
    _validate(
        spec,
        expected_returns,
        cov_matrix,
        returns,
        constraints_turnover=constraints.turnover_limit is not None,
        constraints=constraints,
    )

    common = dict(
        cov_matrix=cov_matrix,
        constraints=constraints,
        risk_free_rate=spec.risk_free_rate,
    )
    if cls not in (CVaROptimizer, CDaROptimizer):
        common["expected_returns"] = expected_returns

    if cls in (MeanVarianceOptimizer, ActiveMeanVarianceOptimizer):
        return cls(risk_aversion=spec.risk_aversion, **common, **overrides)
    if cls is RiskParityOptimizer:
        return cls(risk_budget=spec.risk_budget, **common, **overrides)
    if cls is HRPOptimizer:
        return cls(linkage_method=spec.hrp_linkage, **common, **overrides)
    if cls is HERCOptimizer:
        return cls(
            linkage_method=spec.cluster_linkage,
            n_clusters=spec.n_clusters,
            max_clusters=spec.max_clusters,
            risk_measure=spec.herc_risk_measure,
            alpha=spec.cvar_alpha,
            returns=returns,
            **common,
            **overrides,
        )
    if cls is NCOOptimizer:
        return cls(
            objective=spec.nco_objective,
            linkage_method=spec.cluster_linkage,
            n_clusters=spec.n_clusters,
            max_clusters=spec.max_clusters,
            detone_for_clustering=spec.nco_detone_for_clustering,
            **common,
            **overrides,
        )
    if cls is BlackLittermanOptimizer:
        return cls(
            market_weights=spec.bl_market_caps,
            views=_decode_views(spec.bl_views),
            view_confidences=spec.bl_view_confidences,
            tau=spec.bl_tau,
            risk_aversion=spec.risk_aversion,
            market_return=spec.bl_market_return,
            calibrate_risk_aversion=spec.bl_calibrate_risk_aversion,
            **common,
            **overrides,
        )
    if cls is CVaROptimizer:
        return cls(
            returns=returns,
            alpha=spec.cvar_alpha,
            target_return=spec.target_return,
            periods_per_year=config.periods_per_year,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            constraints=constraints,
            risk_free_rate=spec.risk_free_rate,
            **overrides,
        )
    if cls is CDaROptimizer:
        return cls(
            returns=returns,
            alpha=spec.cdar_alpha,
            target_return=spec.target_return,
            periods_per_year=config.periods_per_year,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            constraints=constraints,
            risk_free_rate=spec.risk_free_rate,
            **overrides,
        )
    return cls(**common, **overrides)
