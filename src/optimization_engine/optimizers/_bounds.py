"""Projection of a weight vector onto the feasible set.

Methods that allocate first and constrain afterwards — the naive baselines,
HRP, and max-diversification's fallback path — need a way to move an
unconstrained answer into the mandate. Two routines do that here:

* :func:`project_to_constraints` finds the *closest* feasible vector in the
  Euclidean sense, honouring per-asset bounds, the budget and group budgets
  together. Closest matters: it keeps as much of the method's own answer as
  the mandate allows.
* :func:`project_to_bounds_iterated` handles per-asset bounds and the budget
  only, by clipping and redistributing. It is the cheaper path taken when no
  group budgets are set, and it cannot see them at all.
"""

from __future__ import annotations

import numpy as np


class InfeasibleBoundsError(ValueError):
    """The bounds and the unit-budget are mutually infeasible."""


def project_to_bounds_iterated(
    w: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_iter: int = 50,
    atol: float = 1e-8,
) -> np.ndarray:
    """Project ``w`` onto ``{w : lb <= w <= ub, sum(w) = 1}``.

    Clips into the box, then spreads the leftover budget across whatever
    slack remains, repeating until both invariants hold.

    Raises:
        InfeasibleBoundsError: When ``sum(lb) > 1`` or ``sum(ub) < 1``.
    """
    w = np.asarray(w, dtype=float).copy()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if not (lb <= ub).all():
        raise ValueError("lb must be <= ub element-wise")
    if lb.sum() > 1.0 + atol or ub.sum() < 1.0 - atol:
        raise InfeasibleBoundsError(
            f"sum(lb)={lb.sum():.6f}, sum(ub)={ub.sum():.6f} cannot meet sum=1"
        )
    for _ in range(max_iter):
        w = np.clip(w, lb, ub)
        s = float(w.sum())
        if abs(s - 1.0) < atol:
            return w
        residual = 1.0 - s
        if residual > 0:
            slack = ub - w
            total = float(slack.sum())
            if total <= atol:
                raise InfeasibleBoundsError(
                    "no upper-side slack to absorb residual"
                )
            w = w + slack * (residual / total)
        else:
            slack = w - lb
            total = float(slack.sum())
            if total <= atol:
                raise InfeasibleBoundsError(
                    "no lower-side slack to absorb deficit"
                )
            w = w - slack * (-residual / total)
    raise RuntimeError("project_to_bounds_iterated did not converge")


def _without_turnover(constraints):
    """Copy of ``constraints`` with the turnover budget dropped."""
    from dataclasses import replace

    return replace(constraints, turnover_limit=None, previous_weights=None)


def project_to_constraints(
    w: np.ndarray,
    assets: list[str],
    constraints,
) -> tuple[np.ndarray, float]:
    """Closest feasible weight vector to ``w``, group budgets included.

    Solves ``min ‖x − w‖²`` subject to the full constraint set. Where
    :func:`project_to_bounds_iterated` can only clip and redistribute — and is
    blind to group budgets, so a 1/N book quietly ends up over its
    alternatives cap — this respects every linear constraint at once while
    staying as close as possible to the method's own allocation.

    Args:
        w: The unconstrained allocation.
        assets: Universe, in the order ``w`` is indexed.
        constraints: A :class:`PortfolioConstraints`.

    Returns:
        ``(projected_weights, distance)`` where ``distance`` is
        ``½‖x − w‖₁`` — the one-way fraction of the book the mandate moved,
        on the same scale as turnover and active share. A large distance means
        the constraints, not the method, produced the answer.

    Raises:
        InfeasibleBoundsError: When no feasible vector exists.
    """
    from optimization_engine.optimizers._cvxpy_helpers import (
        SolverFailure,
        bounds_arrays,
        build_constraints,
        solve_problem,
    )

    w = np.asarray(w, dtype=float).flatten()
    lb, ub = bounds_arrays(assets, constraints)

    if not (constraints.groups and constraints.group_bounds):
        projected = project_to_bounds_iterated(w, lb, ub)
        return projected, float(np.abs(projected - w).sum()) / 2.0

    try:
        import cvxpy as cp

        x = cp.Variable(len(assets))
        # Turnover is meaningless for a projection: this is not a trade.
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(x - w)),
            build_constraints(x, assets, _without_turnover(constraints)),
        )
        solve_problem(problem)
        if x.value is None:
            raise SolverFailure("unknown", ())
        projected = np.asarray(x.value).flatten()
    except SolverFailure as exc:
        raise InfeasibleBoundsError(
            "No allocation satisfies the weight bounds, the budget and the "
            f"group budgets at once. ({exc})"
        ) from exc

    return projected, float(np.abs(projected - w).sum()) / 2.0
