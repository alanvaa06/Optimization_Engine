"""Risk parity / risk-budgeting optimizer.

Implements Equal Risk Contribution (ERC) and arbitrary risk-budgeting via
the convex log-barrier formulation of Spinu (2013) / Maillard et al. (2010):

    minimize   ½ w' Σ w − Σ b_i log(w_i)

The solution is rescaled to satisfy the budget constraint and bounds. This
formulation produces the same allocation as the standard ERC fixed-point
iteration but is convex and trivially solvable with CVXPY.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays, project_to_bounds
from optimization_engine.optimizers.base import BaseOptimizer


class RiskParityOptimizer(BaseOptimizer):
    """Equal Risk Contribution / Risk Budgeting optimizer.

    Pass a ``risk_budget`` mapping ``asset -> target share of variance``
    (sums to 1) to deviate from equal-risk. If absent, defaults to ERC.
    """

    name = "risk_parity"

    def __init__(self, *args, risk_budget: dict[str, float] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.risk_budget = risk_budget

    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        n = len(self.assets)

        if self.risk_budget:
            b = np.array(
                [float(self.risk_budget.get(a, 1.0 / n)) for a in self.assets]
            )
        else:
            b = np.ones(n) / n
        if b.sum() <= 0:
            raise ValueError("Risk budget must sum to a positive number")
        b = b / b.sum()

        y = cp.Variable(n, pos=True)
        sigma_psd = cp.psd_wrap(sigma)
        objective = cp.Minimize(0.5 * cp.quad_form(y, sigma_psd) - b @ cp.log(y))
        problem = cp.Problem(objective)
        try:
            problem.solve()
        except cp.SolverError:
            problem.solve(solver=cp.SCS)
        if y.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")

        w = np.array(y.value) / float(np.sum(y.value))
        lb, ub = bounds_arrays(self.assets, self.constraints)
        return project_to_bounds(w, lb, ub)

    def risk_contributions(self, weights: np.ndarray | pd.Series) -> pd.Series:
        sigma = self._sigma_matrix()
        w = np.asarray(weights).flatten()
        total = float(w @ sigma @ w)
        marginal = sigma @ w
        rc = w * marginal / total if total > 0 else np.zeros_like(w)
        return pd.Series(rc, index=self.assets)
