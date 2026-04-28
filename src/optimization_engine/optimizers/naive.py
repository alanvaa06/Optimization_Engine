"""Naive baselines: equal weight (1/N) and inverse volatility."""

from __future__ import annotations

import numpy as np

from optimization_engine.optimizers._bounds import project_to_bounds_iterated
from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays
from optimization_engine.optimizers.base import BaseOptimizer


class EqualWeightOptimizer(BaseOptimizer):
    """Allocate 1/N to each asset, then project into bounds."""

    name = "equal_weight"

    def _solve(self) -> np.ndarray:
        n = len(self.assets)
        w = np.ones(n) / n
        lb, ub = bounds_arrays(self.assets, self.constraints)
        return project_to_bounds_iterated(w, lb, ub)


class InverseVolatilityOptimizer(BaseOptimizer):
    """Weights inversely proportional to per-asset volatility (no correlations)."""

    name = "inverse_vol"

    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        std = np.sqrt(np.diag(sigma))
        std = np.where(std <= 0, np.nan, std)
        inv = 1.0 / std
        inv = np.nan_to_num(inv, nan=0.0)
        if inv.sum() <= 0:
            raise RuntimeError("All variances are zero or NaN")
        w = inv / inv.sum()
        lb, ub = bounds_arrays(self.assets, self.constraints)
        return project_to_bounds_iterated(w, lb, ub)
