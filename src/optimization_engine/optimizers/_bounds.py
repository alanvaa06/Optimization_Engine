"""Iterated bounds projection.

Replaces the one-shot ``project_to_bounds`` for optimizers that solve
unconstrained then need their weights forced into per-asset bounds while
still summing to one. Iterates clip + redistribute-residual until both
invariants are satisfied or until ``max_iter`` is reached.
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

    Raises :class:`InfeasibleBoundsError` when ``sum(lb) > 1`` or
    ``sum(ub) < 1``.
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
