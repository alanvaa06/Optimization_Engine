"""Mean-variance family: minimum variance, target-return MV, max Sharpe."""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from optimization_engine.optimizers._cvxpy_helpers import (
    build_constraints,
    build_scaled_constraints,
    solve_problem,
)
from optimization_engine.optimizers.base import BaseOptimizer

# Backwards-compatible alias: several modules imported ``_solve_problem``
# from here before solver dispatch moved into ``_cvxpy_helpers``.
_solve_problem = solve_problem
_SOLVER_FALLBACK = ["CLARABEL", "ECOS", "SCS", "OSQP"]


class MinVarianceOptimizer(BaseOptimizer):
    """Global Minimum-Variance portfolio (no return target).

    The one mean-variance portfolio that needs no expected-return vector,
    and therefore the one that is immune to the estimation error that
    dominates ``μ``. That robustness is why it is the natural anchor for
    the efficient frontier.
    """

    name = "min_variance"
    bounds_mode = "hard"

    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        n = len(self.assets)
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
        constraints = build_constraints(w, self.assets, self.constraints)
        problem = cp.Problem(objective, constraints)
        info = solve_problem(problem)
        if w.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        self._diagnostics.update(info.as_dict())
        return w.value


class MeanVarianceOptimizer(BaseOptimizer):
    """Markowitz mean-variance optimizer.

    Three modes determined by ``constraints``:

    * ``target_return`` set        → minimize variance s.t. ``μ'w = R*``
    * ``target_volatility`` set    → maximize ``μ'w`` s.t. ``√(w'Σw) ≤ σ*``
    * neither set                  → maximize ``μ'w − λ·w'Σw`` (utility)

    The volatility target is imposed as ``w'Σw ≤ σ*²`` — a convex quadratic
    constraint, so the solve stays a QP rather than becoming a second-order
    cone problem.
    """

    name = "mean_variance"
    bounds_mode = "hard"

    def __init__(self, *args, risk_aversion: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.risk_aversion = float(risk_aversion)
        if self.risk_aversion < 0:
            raise ValueError(
                f"risk_aversion must be non-negative; got {risk_aversion}. "
                "A negative coefficient rewards risk and makes the problem "
                "unbounded."
            )

    def _solve(self) -> np.ndarray:
        mu = self._mu_vector()
        sigma = self._sigma_matrix()
        if mu is None or sigma is None:
            raise ValueError("Mean-variance needs both expected_returns and cov_matrix")
        n = len(self.assets)
        w = cp.Variable(n)
        sigma_psd = cp.psd_wrap(sigma)

        if self.constraints.target_return is not None:
            mode = "target_return"
            objective = cp.Minimize(cp.quad_form(w, sigma_psd))
            extra = [mu @ w == float(self.constraints.target_return)]
        elif self.constraints.target_volatility is not None:
            mode = "target_volatility"
            target_vol = float(self.constraints.target_volatility)
            if target_vol <= 0:
                raise ValueError(
                    f"target_volatility must be positive; got {target_vol}."
                )
            objective = cp.Maximize(mu @ w)
            extra = [cp.quad_form(w, sigma_psd) <= target_vol**2]
        else:
            mode = "utility"
            if self.risk_aversion == 0:
                warnings.warn(
                    "risk_aversion=0 makes this a pure return-maximization: the "
                    "result will sit at the corner of the constraint set and "
                    "ignore risk entirely.",
                    stacklevel=3,
                )
            objective = cp.Maximize(mu @ w - self.risk_aversion * cp.quad_form(w, sigma_psd))
            extra = []

        cons = build_constraints(w, self.assets, self.constraints, extra)
        problem = cp.Problem(objective, cons)
        info = solve_problem(problem)
        if w.value is None:
            raise RuntimeError(
                f"Solver failed for {self.name}: status={problem.status}. "
                "Constraints may be infeasible (check bounds and target)."
            )
        self._diagnostics.update(info.as_dict())
        self._diagnostics["mode"] = mode
        return w.value


class MaxSharpeOptimizer(BaseOptimizer):
    """Maximum Sharpe ratio (tangency) portfolio.

    Solved as the standard homogeneous reformulation: minimize ``y'Σy``
    subject to ``(μ − rf)·y = 1``, then renormalize ``w = y / Σy``. Bounds
    and group budgets are scaled by ``κ = Σy`` so they stay exact.

    Two things do not survive the transform, and both are reported rather
    than silently dropped: a turnover budget (affine, not homogeneous) and
    the case where the tangency ray points the wrong way because no asset
    earns more than the risk-free rate.
    """

    name = "max_sharpe"
    bounds_mode = "hard"

    def _solve(self) -> np.ndarray:
        mu = self._mu_vector()
        sigma = self._sigma_matrix()
        if mu is None or sigma is None:
            raise ValueError("Max-Sharpe needs both expected_returns and cov_matrix")
        rf = self.risk_free_rate
        excess = mu - rf
        if np.all(excess <= 0):
            raise ValueError(
                f"Every expected return is at or below the risk-free rate "
                f"({rf:.2%}), so no portfolio has a positive Sharpe ratio and "
                "the tangency portfolio is undefined. Lower the risk-free rate "
                "or revisit the expected returns."
            )

        if self.constraints.turnover_limit is not None:
            warnings.warn(
                "Max-Sharpe ignores the turnover budget: the tangency solve "
                "works on a scaled ray where a turnover constraint is not "
                "well defined. Use mean_variance with a return target to "
                "respect turnover.",
                stacklevel=3,
            )
            self._diagnostics["ignored_constraints"] = ["turnover_limit"]

        n = len(self.assets)
        y = cp.Variable(n)
        kappa = cp.Variable(nonneg=True)

        sigma_psd = cp.psd_wrap(sigma)
        objective = cp.Minimize(cp.quad_form(y, sigma_psd))
        cons = [excess @ y == 1]
        cons += build_scaled_constraints(y, kappa, self.assets, self.constraints)

        problem = cp.Problem(objective, cons)
        info = solve_problem(problem)
        if y.value is None or kappa.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        scale = float(kappa.value)
        if scale <= 1e-10:
            raise RuntimeError(
                "Degenerate tangency solution (Σy ≈ 0). The constraints likely "
                "forbid any portfolio with positive excess return."
            )
        self._diagnostics.update(info.as_dict())
        return np.array(y.value) / scale
