"""Maximum Diversification optimizer (Choueifaty & Coignard, 2008).

Maximizes the diversification ratio:

    DR(w) = (Σ_i w_i · σ_i) / √(w' Σ w)

A higher DR means weighted-average asset volatility is being meaningfully
reduced by correlation. The classic transformation reduces this to:

    minimize   y' Σ y      subject to σ' y = 1, y ≥ 0
    weights    w = y / Σ y

Because the objective and the normalization are homogeneous, per-asset and
group bounds carry over exactly when scaled by ``κ = Σy`` — so this solves
the *constrained* max-DR problem rather than solving it unconstrained and
projecting afterwards, which can cost several points of diversification
ratio whenever the caps actually bind.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from optimization_engine.optimizers._bounds import project_to_constraints
from optimization_engine.optimizers._cvxpy_helpers import (
    build_scaled_constraints,
    solve_problem,
)
from optimization_engine.optimizers.base import BaseOptimizer


class MaxDiversificationOptimizer(BaseOptimizer):
    """Maximum diversification-ratio portfolio, with hard bounds."""

    name = "max_diversification"
    bounds_mode = "hard"

    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        std = np.sqrt(np.diag(sigma))
        if not (std > 0).all():
            zero = [a for a, s in zip(self.assets, std) if s <= 0]
            raise ValueError(
                f"Zero-variance asset(s) {zero}: the diversification ratio is "
                "undefined when an asset has no volatility. Drop them from the "
                "universe."
            )

        if self.constraints.turnover_limit is not None:
            warnings.warn(
                "Max-diversification ignores the turnover budget: the solve "
                "works on a scaled ray where a turnover constraint is not well "
                "defined.",
                stacklevel=3,
            )
            self._diagnostics["ignored_constraints"] = ["turnover_limit"]

        n = len(self.assets)
        y = cp.Variable(n)
        kappa = cp.Variable(nonneg=True)
        sigma_psd = cp.psd_wrap(sigma)
        objective = cp.Minimize(cp.quad_form(y, sigma_psd))
        cons = [std @ y == 1]
        cons += build_scaled_constraints(y, kappa, self.assets, self.constraints)

        problem = cp.Problem(objective, cons)
        try:
            info = solve_problem(problem)
        except Exception as exc:
            return self._fallback_projection(sigma, std, exc)

        if y.value is None or kappa.value is None or float(kappa.value) <= 1e-10:
            return self._fallback_projection(
                sigma, std, RuntimeError("degenerate scaled solution")
            )
        self._diagnostics.update(info.as_dict())
        return np.array(y.value) / float(kappa.value)

    def _fallback_projection(
        self, sigma: np.ndarray, std: np.ndarray, cause: Exception
    ) -> np.ndarray:
        """Unconstrained solve + projection, used only if the exact solve fails.

        This is strictly worse than the constrained solve — it is a last
        resort, and it says so in the diagnostics so the result is never
        mistaken for the true constrained optimum.
        """
        n = len(self.assets)
        y = cp.Variable(n, nonneg=True)
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma))), [std @ y == 1]
        )
        info = solve_problem(problem)
        if y.value is None:
            raise RuntimeError(
                f"Max-diversification failed: {cause}"
            ) from cause
        w = np.array(y.value)
        total = w.sum()
        if total <= 0:
            raise RuntimeError("Degenerate Max-Diversification solution")
        w, distance = project_to_constraints(w / total, self.assets, self.constraints)
        self._diagnostics.update(info.as_dict())
        self._diagnostics["projection_distance"] = distance
        self._diagnostics["fallback_reason"] = (
            f"Exact bounded solve failed ({cause}); bounds were applied by "
            "projection, so the diversification ratio is below the true "
            "constrained optimum."
        )
        self.bounds_mode = "soft_iterated"
        return w
