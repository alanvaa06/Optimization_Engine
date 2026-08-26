"""Base classes shared by all optimizers."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


@dataclass
class PortfolioConstraints:
    """Bounds, group constraints, exposure limits and a turnover budget.

    The constraints are applied uniformly by all CVXPY-based optimizers
    via the helper ``build_constraints``.

    Attributes:
        bounds: ``asset -> (min, max)`` weight.
        groups: ``asset -> group`` label (e.g. asset class).
        group_bounds: ``group -> (min, max)`` aggregate weight.
        fully_invested: Force ``sum(w) == 1``.
        long_only: Disallow negative weights. Tightens any bound whose
            minimum is negative up to ``0``.
        leverage: Cap on gross exposure ``Σ|w_i|``. ``None`` means uncapped
            (and, under ``long_only`` with a unit budget, gross is 1 anyway).
        target_return: Hard expected-return target ``μ'w == R*``.
        target_volatility: Hard volatility cap ``√(w'Σw) ≤ σ*``.
        previous_weights: The book being traded *from*. Required for a
            turnover budget and for reporting realized turnover.
        turnover_limit: Cap on ``Σ|w_i − w_prev,i|``. A one-way turnover of
            0.20 means at most 20% of the portfolio changes hands.
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

    def get_bounds(
        self, asset: str, default: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """Resolve the effective ``(min, max)`` weight for one asset.

        ``long_only`` is applied here rather than as a separate constraint so
        that every consumer — the CVXPY builder, the projection helpers and
        the post-solve checker — sees exactly the same box.
        """
        if asset in self.bounds:
            lo, hi = self.bounds[asset]
            lo, hi = float(lo), float(hi)
        elif default is not None:
            lo, hi = default
        else:
            lo, hi = (0.0, 1.0) if self.long_only else (-1.0, 1.0)
        if self.long_only:
            lo = max(lo, 0.0)
            hi = max(hi, 0.0)
        return lo, hi


@dataclass
class OptimizationResult:
    """Output of an optimizer, with helpers for analytics.

    ``extras`` carries everything that is not the allocation itself: solver
    status, concentration diagnostics, constraint violations, and any
    method-specific output (the Black-Litterman posterior, the realized CVaR,
    the risk-contribution error of an ERC solve).
    """

    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def solver_status(self) -> str:
        return str(self.extras.get("solver_status", "unknown"))

    @property
    def violations(self) -> list[str]:
        """Human-readable constraint breaches, empty when fully compliant."""
        return list(self.extras.get("violations", []))

    @property
    def is_compliant(self) -> bool:
        return not self.violations

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
    class handles input shaping, constraint plumbing, post-solve validation
    and summary statistics for the resulting allocation.
    """

    name: str = "base"

    #: How faithfully this optimizer honours per-asset and group bounds.
    #: ``"hard"``          — enforced inside the convex program.
    #: ``"soft_iterated"`` — solved unconstrained, then projected into the box.
    #: ``"constrained"``   — enforced in the program, up to solver tolerance.
    bounds_mode: str = "hard"

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
        #: Populated by subclasses; surfaced through ``result.extras``.
        self._diagnostics: dict[str, Any] = {}

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
        """Solve, validate, and package the allocation with its diagnostics."""
        weights = self._solve()
        weights = np.asarray(weights, dtype=float).flatten()
        if not np.isfinite(weights).all():
            raise RuntimeError(
                f"{self.name} produced non-finite weights — the problem is "
                "likely unbounded or numerically degenerate."
            )
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

        extras: dict[str, Any] = {
            "optimizer": self.name,
            "bounds_mode": self.bounds_mode,
            **self._diagnostics,
        }
        extras.update(self._post_solve_diagnostics(w))

        return OptimizationResult(
            weights=w,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            extras=extras,
        )

    def _post_solve_diagnostics(self, weights: pd.Series) -> dict[str, Any]:
        """Concentration, diversification and constraint-compliance summary."""
        from optimization_engine.optimizers.diagnostics import portfolio_diagnostics

        diag = portfolio_diagnostics(
            weights, cov_matrix=self.cov_matrix, constraints=self.constraints
        )
        if diag.violations:
            _LOG.warning(
                "%s produced weights violating %d constraint(s): %s",
                self.name,
                len(diag.violations),
                "; ".join(v.describe() for v in diag.violations),
            )
        return {"diagnostics": diag, "violations": [v.describe() for v in diag.violations]}

    def _mu_vector(self) -> np.ndarray | None:
        """Expected returns aligned to ``self.assets``.

        Missing entries are treated as zero, which is a strong and usually
        wrong assumption — so it is warned about rather than done silently.
        """
        if self.expected_returns is None:
            return None
        aligned = self.expected_returns.reindex(self.assets)
        missing = [a for a, v in aligned.items() if pd.isna(v)]
        if missing:
            warnings.warn(
                f"{self.name}: no expected return for {len(missing)} asset(s) "
                f"({', '.join(map(str, missing[:5]))}"
                f"{' …' if len(missing) > 5 else ''}); assuming 0.0. "
                "A zero expected return is an active view, not a neutral one.",
                stacklevel=3,
            )
            self._diagnostics["missing_expected_returns"] = [str(m) for m in missing]
        return aligned.fillna(0.0).values

    def _sigma_matrix(self) -> np.ndarray | None:
        if self.cov_matrix is None:
            return None
        return self.cov_matrix.reindex(self.assets, axis=0).reindex(self.assets, axis=1).values

    def _clean_weights(self, w: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Zero out dust, restore the budget, and keep the result inside the box.

        Renormalizing after truncation can push a weight through its bound,
        so the result is re-projected onto ``{lb ≤ w ≤ ub, Σw = 1}`` whenever
        the naive rescale would breach it. Portfolios that are not fully
        invested (or that net to ~0, where rescaling is meaningless) are
        returned as-is apart from the dust removal.
        """
        from optimization_engine.optimizers._bounds import (
            InfeasibleBoundsError,
            project_to_bounds_iterated,
        )

        w = np.where(np.abs(w) < tol, 0.0, w)
        if not self.constraints.fully_invested:
            return w

        s = float(w.sum())
        if abs(s) < 1e-9:
            return w
        rescaled = w / s

        lb = np.array([self.constraints.get_bounds(a)[0] for a in self.assets])
        ub = np.array([self.constraints.get_bounds(a)[1] for a in self.assets])
        if (rescaled >= lb - tol).all() and (rescaled <= ub + tol).all():
            return rescaled
        try:
            return project_to_bounds_iterated(rescaled, lb, ub)
        except (InfeasibleBoundsError, RuntimeError):
            # Bounds and budget are mutually impossible; the feasibility
            # report is the right place to explain that, so keep the solver's
            # answer rather than silently mangling it.
            return rescaled
