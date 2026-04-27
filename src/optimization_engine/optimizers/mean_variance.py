"""Mean-variance family: minimum variance, target-return MV, max Sharpe."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from optimization_engine.optimizers._cvxpy_helpers import build_constraints
from optimization_engine.optimizers.base import BaseOptimizer

_SOLVER_FALLBACK = ["CLARABEL", "ECOS", "SCS", "OSQP"]


def _solve_problem(problem: cp.Problem) -> None:
    last_err: Exception | None = None
    for solver in _SOLVER_FALLBACK:
        try:
            if solver in cp.installed_solvers():
                problem.solve(solver=solver)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    return
        except Exception as e:  # solver missing or numerical
            last_err = e
            continue
    try:
        problem.solve()
    except Exception as e:
        if last_err is not None:
            raise RuntimeError(
                f"All solvers failed. Last error: {last_err}"
            ) from e
        raise


class MinVarianceOptimizer(BaseOptimizer):
    """Global Minimum-Variance portfolio (no return target)."""

    name = "min_variance"

    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        n = len(self.assets)
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
        constraints = build_constraints(w, self.assets, self.constraints)
        problem = cp.Problem(objective, constraints)
        _solve_problem(problem)
        if w.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        return w.value


class MeanVarianceOptimizer(BaseOptimizer):
    """Markowitz mean-variance optimizer.

    Three modes determined by ``constraints``:

    * ``target_return`` set        → minimize variance s.t. ``μ'w = R*``
    * ``target_volatility`` set    → maximize ``μ'w`` s.t. ``√(w'Σw) ≤ σ*``
    * neither set                  → maximize ``μ'w − λ·w'Σw`` (utility)
    """

    name = "mean_variance"

    def __init__(self, *args, risk_aversion: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.risk_aversion = float(risk_aversion)

    def _solve(self) -> np.ndarray:
        mu = self._mu_vector()
        sigma = self._sigma_matrix()
        if mu is None or sigma is None:
            raise ValueError("Mean-variance needs both expected_returns and cov_matrix")
        n = len(self.assets)
        w = cp.Variable(n)
        sigma_psd = cp.psd_wrap(sigma)

        if self.constraints.target_return is not None:
            objective = cp.Minimize(cp.quad_form(w, sigma_psd))
            extra = [mu @ w == float(self.constraints.target_return)]
        elif self.constraints.target_volatility is not None:
            objective = cp.Maximize(mu @ w)
            extra = [
                cp.quad_form(w, sigma_psd)
                <= float(self.constraints.target_volatility) ** 2
            ]
        else:
            objective = cp.Maximize(mu @ w - self.risk_aversion * cp.quad_form(w, sigma_psd))
            extra = []

        cons = build_constraints(w, self.assets, self.constraints, extra)
        problem = cp.Problem(objective, cons)
        _solve_problem(problem)
        if w.value is None:
            raise RuntimeError(
                f"Solver failed for {self.name}: status={problem.status}. "
                "Constraints may be infeasible (check bounds and target)."
            )
        return w.value


class MaxSharpeOptimizer(BaseOptimizer):
    """Maximum Sharpe ratio portfolio.

    Solved as the well-known second-cone reformulation: minimize ``y'Σy``
    subject to ``(μ − rf)·y = 1`` and ``y ≥ 0`` (within scaled bounds),
    then renormalize ``w = y / sum(y)``.
    """

    name = "max_sharpe"

    def _solve(self) -> np.ndarray:
        mu = self._mu_vector()
        sigma = self._sigma_matrix()
        if mu is None or sigma is None:
            raise ValueError("Max-Sharpe needs both expected_returns and cov_matrix")
        rf = self.risk_free_rate
        excess = mu - rf
        if np.all(excess <= 0):
            raise ValueError(
                "All expected returns are below the risk-free rate; "
                "Max-Sharpe is undefined."
            )

        n = len(self.assets)
        y = cp.Variable(n)
        kappa = cp.Variable(nonneg=True)

        sigma_psd = cp.psd_wrap(sigma)
        objective = cp.Minimize(cp.quad_form(y, sigma_psd))
        cons = [excess @ y == 1, cp.sum(y) == kappa, kappa >= 1e-6]

        for i, asset in enumerate(self.assets):
            lo, hi = self.constraints.get_bounds(asset)
            cons.append(y[i] >= lo * kappa)
            cons.append(y[i] <= hi * kappa)

        if self.constraints.groups and self.constraints.group_bounds:
            grouped: dict[str, list[int]] = {}
            for i, asset in enumerate(self.assets):
                g = self.constraints.groups.get(asset)
                if g is not None:
                    grouped.setdefault(g, []).append(i)
            for group, idx in grouped.items():
                if group in self.constraints.group_bounds:
                    lo, hi = self.constraints.group_bounds[group]
                    cons.append(cp.sum(y[idx]) >= float(lo) * kappa)
                    cons.append(cp.sum(y[idx]) <= float(hi) * kappa)

        problem = cp.Problem(objective, cons)
        _solve_problem(problem)
        if y.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        weights = np.array(y.value) / float(kappa.value)
        return weights
