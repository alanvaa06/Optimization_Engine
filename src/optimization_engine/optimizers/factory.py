"""Factory: build the right optimizer from a config object."""

from __future__ import annotations

from typing import Any

import pandas as pd

from optimization_engine.config import EngineConfig, OptimizerSpec
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
from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

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


def optimizer_factory(
    config: EngineConfig,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    returns: pd.DataFrame | None = None,
    **overrides: Any,
) -> BaseOptimizer:
    """Build an optimizer instance from an `EngineConfig`.

    `expected_returns` defaults to the values in ``config.expected_returns``.
    `returns` is required for the CVaR optimizer.
    """
    spec: OptimizerSpec = config.optimizer
    name = spec.name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {available_optimizers()}"
        )
    cls = _REGISTRY[name]

    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns, name="expected_return")

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
        if returns is None:
            raise ValueError("CVaR optimizer requires a returns DataFrame")
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
