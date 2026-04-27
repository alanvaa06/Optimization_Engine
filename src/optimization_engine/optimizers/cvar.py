"""Mean-CVaR optimizer (Rockafellar & Uryasev, 2000).

Minimizes Conditional Value-at-Risk (Expected Shortfall) directly from a
historical scenario set. Linear in scenarios — this is convex and scales
well to thousands of joint return paths.

CVaR formulation (β = confidence level, e.g. 0.95):

    minimize    α + (1 / (1 − β)·T) · Σ_t z_t
    subject to  z_t ≥ −r_t' w − α
                z_t ≥ 0
                μ' w ≥ R_target  (optional)
                Σ w = 1, w ∈ bounds
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from optimization_engine.optimizers._cvxpy_helpers import build_constraints
from optimization_engine.optimizers.base import BaseOptimizer
from optimization_engine.optimizers.mean_variance import _solve_problem


class CVaROptimizer(BaseOptimizer):
    """Mean-CVaR optimizer.

    Pass historical (or simulated) ``returns`` — the engine will minimize
    the empirical CVaR at the given confidence level. ``alpha`` here is
    the tail probability (e.g. ``0.05`` for 95% CVaR).
    """

    name = "cvar"

    def __init__(
        self,
        returns: pd.DataFrame,
        *args,
        alpha: float = 0.05,
        target_return: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if returns is None or returns.empty:
            raise ValueError("CVaR optimizer requires a returns DataFrame")
        self.returns = returns
        self.alpha = float(alpha)
        self.target_return = target_return

    @property
    def assets(self) -> list[str]:  # type: ignore[override]
        return list(self.returns.columns)

    def _solve(self) -> np.ndarray:
        T, n = self.returns.shape
        R = self.returns.values
        w = cp.Variable(n)
        alpha_var = cp.Variable()
        z = cp.Variable(T, nonneg=True)

        # Loss is the negative of return.
        losses = -(R @ w)
        portfolio_cvar = alpha_var + cp.sum(z) / (self.alpha * T)

        extras = [z >= losses - alpha_var]
        if self.target_return is not None and self.expected_returns is not None:
            mu = self._mu_vector()
            extras.append(mu @ w >= float(self.target_return))
        elif self.target_return is not None:
            mean_r = self.returns.mean().values * 252  # annualize ~ default
            extras.append(mean_r @ w >= float(self.target_return))

        cons = build_constraints(w, self.assets, self.constraints, extras)
        problem = cp.Problem(cp.Minimize(portfolio_cvar), cons)
        _solve_problem(problem)
        if w.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        return np.array(w.value)
