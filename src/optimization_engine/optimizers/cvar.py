"""Mean-CVaR optimizer (Rockafellar & Uryasev, 2000).

Minimizes Conditional Value-at-Risk (Expected Shortfall) directly from a
historical scenario set. Linear in scenarios — this is convex and scales
well to thousands of joint return paths.

CVaR formulation (α = tail probability, e.g. 0.05 for the 95% level):

    minimize    ζ + (1 / (α·T)) · Σ_t z_t
    subject to  z_t ≥ −r_t' w − ζ
                z_t ≥ 0
                μ' w ≥ R_target  (optional)
                Σ w = 1, w ∈ bounds

At the optimum ``ζ`` is the Value-at-Risk and the objective is the CVaR,
both expressed in the units of one period of ``returns``.

Unlike variance, CVaR is estimated from the tail alone: with α = 0.05 and
250 observations only ~12 scenarios drive the answer. The optimizer reports
that tail count so an under-powered estimate is visible rather than implied.
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from optimization_engine.optimizers._cvxpy_helpers import build_constraints, solve_problem
from optimization_engine.optimizers.base import BaseOptimizer

#: Fewer scenarios than this in the tail and the CVaR estimate is anecdote.
MIN_TAIL_SCENARIOS = 10


class CVaROptimizer(BaseOptimizer):
    """Mean-CVaR optimizer.

    Pass historical (or simulated) ``returns`` — the engine minimizes the
    empirical CVaR at the given tail probability. ``alpha`` is the tail
    probability (``0.05`` ⇒ 95% CVaR), not the confidence level.

    Args:
        returns: Scenario matrix of *periodic* returns, one column per asset.
        alpha: Tail probability in ``(0, 0.5)``.
        target_return: Optional minimum *annualized* expected return.
        periods_per_year: Used to annualize the reported CVaR/VaR and to
            convert historical means when no ``expected_returns`` is supplied.
    """

    name = "cvar"
    bounds_mode = "hard"

    def __init__(
        self,
        returns: pd.DataFrame,
        *args,
        alpha: float = 0.05,
        target_return: float | None = None,
        periods_per_year: int = 252,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if returns is None or returns.empty:
            raise ValueError("CVaR optimizer requires a returns DataFrame")
        if not 0 < alpha < 0.5:
            raise ValueError(
                f"alpha is the tail probability and must be in (0, 0.5); got "
                f"{alpha}. Pass 0.05 for the 95% CVaR, not 0.95."
            )
        self.returns = returns
        self.alpha = float(alpha)
        self.target_return = target_return
        self.periods_per_year = int(periods_per_year)

    @property
    def assets(self) -> list[str]:  # type: ignore[override]
        return list(self.returns.columns)

    def _solve(self) -> np.ndarray:
        T, n = self.returns.shape
        expected_tail = self.alpha * T
        if expected_tail < MIN_TAIL_SCENARIOS:
            warnings.warn(
                f"Only ~{expected_tail:.0f} of {T} scenarios fall in the {self.alpha:.0%} "
                "tail, so the CVaR estimate rests on a handful of observations. "
                "Use a longer history, a larger alpha, or simulated scenarios.",
                stacklevel=3,
            )
        self._diagnostics["tail_scenarios"] = float(expected_tail)
        self._diagnostics["n_scenarios"] = int(T)

        R = self.returns.values
        w = cp.Variable(n)
        zeta = cp.Variable()
        z = cp.Variable(T, nonneg=True)

        # Loss is the negative of return.
        losses = -(R @ w)
        portfolio_cvar = zeta + cp.sum(z) / (self.alpha * T)

        extras = [z >= losses - zeta]
        if self.target_return is not None:
            mu = self._target_mu_vector()
            extras.append(mu @ w >= float(self.target_return))

        cons = build_constraints(w, self.assets, self.constraints, extras)
        problem = cp.Problem(cp.Minimize(portfolio_cvar), cons)
        info = solve_problem(problem)
        if w.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")
        self._diagnostics.update(info.as_dict())

        weights = np.array(w.value)
        self._record_tail_metrics(weights, float(zeta.value))
        return weights

    def _target_mu_vector(self) -> np.ndarray:
        """Annualized expected returns used for the optional return floor."""
        if self.expected_returns is not None:
            return self._mu_vector()
        periodic = self.returns.mean().values
        warnings.warn(
            "No expected returns supplied; the CVaR return target is being "
            "compared against annualized historical means from the same "
            "scenario set.",
            stacklevel=4,
        )
        return (1 + periodic) ** self.periods_per_year - 1

    def _record_tail_metrics(self, weights: np.ndarray, zeta: float) -> None:
        """Report the realized tail statistics of the chosen allocation.

        The optimizer's own objective is the number an analyst wants to see,
        and it is not recoverable from the mean/variance summary the base
        class produces.
        """
        port = self.returns.values @ weights
        var_hist = float(-np.quantile(port, self.alpha))
        tail = port[port <= -var_hist]
        cvar_hist = float(-tail.mean()) if len(tail) else float("nan")
        scale = float(np.sqrt(self.periods_per_year))
        self._diagnostics.update(
            {
                "cvar_alpha": self.alpha,
                "cvar_period": cvar_hist,
                "var_period": var_hist,
                "cvar_solver_zeta": zeta,
                "cvar_annualized": cvar_hist * scale,
                "var_annualized": var_hist * scale,
                "tail_observations": int(len(tail)),
                "cvar_note": (
                    f"{1 - self.alpha:.0%} CVaR of {cvar_hist:.2%} per period "
                    f"({cvar_hist * scale:.2%} scaled by √{self.periods_per_year}), "
                    f"averaged over {len(tail)} tail scenario(s)."
                ),
            }
        )
