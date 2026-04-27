"""Base classes shared by all optimizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PortfolioConstraints:
    """Bounds, group constraints, and an optional turnover budget.

    The constraints are applied uniformly by all CVXPY-based optimizers
    via the helper `apply_constraints`.
    """

    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    group_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    fully_invested: bool = True
    long_only: bool = True
    leverage: float | None = None
    target_return: float | None = None
    target_volatility: float | None = None
    previous_weights: dict[str, float] | None = None
    turnover_limit: float | None = None

    def get_bounds(self, asset: str, default: tuple[float, float] | None = None) -> tuple[float, float]:
        if asset in self.bounds:
            lo, hi = self.bounds[asset]
            return float(lo), float(hi)
        if default is not None:
            return default
        return (0.0, 1.0) if self.long_only else (-1.0, 1.0)


@dataclass
class OptimizationResult:
    """Output of an optimizer, with helpers for analytics."""

    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.to_dict(),
            "expected_return": float(self.expected_return),
            "expected_volatility": float(self.expected_volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
            **self.extras,
        }


class BaseOptimizer(ABC):
    """Abstract base for all optimizers in the engine.

    Subclasses implement ``_solve`` to return a 1-D weight vector. The base
    class handles input shaping, constraint plumbing and computing summary
    statistics for the resulting allocation.
    """

    name: str = "base"

    def __init__(
        self,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        constraints: PortfolioConstraints | None = None,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.constraints = constraints or PortfolioConstraints()
        self.risk_free_rate = float(risk_free_rate)

    @property
    def assets(self) -> list[str]:
        if self.cov_matrix is not None:
            return list(self.cov_matrix.columns)
        if self.expected_returns is not None:
            return list(self.expected_returns.index)
        raise ValueError("Optimizer needs either cov_matrix or expected_returns")

    @abstractmethod
    def _solve(self) -> np.ndarray: ...

    def optimize(self) -> OptimizationResult:
        weights = self._solve()
        weights = np.asarray(weights, dtype=float).flatten()
        weights = self._clean_weights(weights)
        w = pd.Series(weights, index=self.assets, name="weight")

        mu = self._mu_vector()
        sigma = self._sigma_matrix()
        port_return = float(w.values @ mu) if mu is not None else float("nan")
        port_var = float(w.values @ sigma @ w.values) if sigma is not None else float("nan")
        port_vol = float(np.sqrt(max(port_var, 0.0))) if not np.isnan(port_var) else float("nan")
        if not np.isnan(port_vol) and port_vol > 0 and not np.isnan(port_return):
            sharpe = (port_return - self.risk_free_rate) / port_vol
        else:
            sharpe = float("nan")
        return OptimizationResult(
            weights=w,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
        )

    def _mu_vector(self) -> np.ndarray | None:
        if self.expected_returns is None:
            return None
        return self.expected_returns.reindex(self.assets).fillna(0.0).values

    def _sigma_matrix(self) -> np.ndarray | None:
        if self.cov_matrix is None:
            return None
        return self.cov_matrix.reindex(self.assets, axis=0).reindex(self.assets, axis=1).values

    def _clean_weights(self, w: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        w = np.where(np.abs(w) < tol, 0.0, w)
        s = w.sum()
        if s and self.constraints.fully_invested:
            w = w / s
        return w
