"""Factory: build the right optimizer from a config object."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.optimizers import ConfigurationError
from optimization_engine.optimizers.base import BaseOptimizer, PortfolioConstraints
from optimization_engine.optimizers.black_litterman import BlackLittermanOptimizer
from optimization_engine.optimizers.cvar import CVaROptimizer
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
from optimization_engine.optimizers.requirements import requirements_for
from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

_LOG = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseOptimizer]] = {
    "mean_variance": MeanVarianceOptimizer,
    "min_variance": MinVarianceOptimizer,
    "max_sharpe": MaxSharpeOptimizer,
    "risk_parity": RiskParityOptimizer,
    "hrp": HRPOptimizer,
    "black_litterman": BlackLittermanOptimizer,
    "cvar": CVaROptimizer,
    "max_diversification": MaxDiversificationOptimizer,
    "equal_weight": EqualWeightOptimizer,
    "inverse_vol": InverseVolatilityOptimizer,
}


def available_optimizers() -> list[str]:
    return sorted(_REGISTRY.keys())


def _constraints_from_config(config: EngineConfig) -> PortfolioConstraints:
    bounds = {k: tuple(v) for k, v in config.bounds.items()}
    group_bounds = {k: tuple(v) for k, v in config.group_bounds.items()}
    return PortfolioConstraints(
        bounds=bounds,
        groups=dict(config.groups),
        group_bounds=group_bounds,
        target_return=config.optimizer.target_return,
        target_volatility=config.optimizer.target_volatility,
    )


def _validate(spec: OptimizerSpec, expected_returns, cov_matrix, returns) -> None:
    req = requirements_for(spec.name)
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

    _validate(spec, expected_returns, cov_matrix, returns)

    constraints = _constraints_from_config(config)

    common = dict(
        cov_matrix=cov_matrix,
        constraints=constraints,
        risk_free_rate=spec.risk_free_rate,
    )
    if cls is not CVaROptimizer:
        common["expected_returns"] = expected_returns

    if cls is MeanVarianceOptimizer:
        return cls(risk_aversion=spec.risk_aversion, **common, **overrides)
    if cls is RiskParityOptimizer:
        return cls(risk_budget=spec.risk_budget, **common, **overrides)
    if cls is HRPOptimizer:
        return cls(linkage_method=spec.hrp_linkage, **common, **overrides)
    if cls is BlackLittermanOptimizer:
        return cls(
            market_weights=spec.bl_market_caps,
            views=spec.bl_views,
            view_confidences=spec.bl_view_confidences,
            tau=spec.bl_tau,
            risk_aversion=spec.risk_aversion,
            **common,
            **overrides,
        )
    if cls is CVaROptimizer:
        return cls(
            returns=returns,
            alpha=spec.cvar_alpha,
            target_return=spec.target_return,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            constraints=constraints,
            risk_free_rate=spec.risk_free_rate,
            **overrides,
        )
    return cls(**common, **overrides)
