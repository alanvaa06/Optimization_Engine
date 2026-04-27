"""Maximum Diversification optimizer (Choueifaty & Coignard, 2008).

Maximizes the diversification ratio:

    DR(w) = (Σ_i w_i · σ_i) / √(w' Σ w)

A higher DR means weighted-average asset volatility is being meaningfully
reduced by correlation. The classic transformation reduces this to:

    minimize   y' Σ y      subject to σ' y = 1, y ≥ 0
    weights    w = y / Σ y
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays, project_to_bounds
from optimization_engine.optimizers.base import BaseOptimizer
from optimization_engine.optimizers.mean_variance import _solve_problem


class MaxDiversificationOptimizer(BaseOptimizer):
    name = "max_diversification"

    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        std = np.sqrt(np.diag(sigma))
        n = len(self.assets)
        y = cp.Variable(n, nonneg=True)
        sigma_psd = cp.psd_wrap(sigma)
        objective = cp.Minimize(cp.quad_form(y, sigma_psd))
        cons = [std @ y == 1]

        # Forward-scale bounds with the unknown sum(y) -- best-effort:
        # we apply post-hoc projection to bounds after normalizing.
        problem = cp.Problem(objective, cons)
        _solve_problem(problem)
        if y.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        w = np.array(y.value)
        s = w.sum()
        if s <= 0:
            raise RuntimeError("Degenerate Max-Diversification solution")
        w = w / s

        lb, ub = bounds_arrays(self.assets, self.constraints)
        return project_to_bounds(w, lb, ub)
