"""Shared CVXPY constraint plumbing and weight-projection utilities."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from optimization_engine.optimizers.base import PortfolioConstraints


def project_to_bounds(
    w: np.ndarray, lb: np.ndarray, ub: np.ndarray, max_iter: int = 100
) -> np.ndarray:
    """Project ``w`` into ``[lb, ub]`` while preserving ``Σ w = 1``.

    Iteratively clips violators and redistributes residual mass to assets
    that still have headroom. Falls back to a single clipped renorm when
    the bounds are themselves infeasible.
    """
    w = np.asarray(w, dtype=float).copy()
    if lb.sum() > 1 + 1e-9 or ub.sum() < 1 - 1e-9:
        clipped = np.clip(w, lb, ub)
        s = clipped.sum()
        return clipped / s if s > 0 else clipped

    w = np.clip(w, lb, ub)
    for _ in range(max_iter):
        s = w.sum()
        diff = 1.0 - s
        if abs(diff) < 1e-10:
            return w
        if diff > 0:
            slack = ub - w
            mask = slack > 1e-12
        else:
            slack = w - lb
            mask = slack > 1e-12
        if not mask.any():
            return w
        share = slack * mask
        share = share / share.sum()
        w = w + diff * share
        w = np.clip(w, lb, ub)
    return w


def bounds_arrays(
    assets: list[str], constraints: PortfolioConstraints
) -> tuple[np.ndarray, np.ndarray]:
    lb = np.array([constraints.get_bounds(a)[0] for a in assets])
    ub = np.array([constraints.get_bounds(a)[1] for a in assets])
    return lb, ub


def build_constraints(
    weights: cp.Variable,
    assets: list[str],
    constraints: PortfolioConstraints,
    extra_constraints: list[cp.Constraint] | None = None,
) -> list[cp.Constraint]:
    """Translate a `PortfolioConstraints` object into a CVXPY constraint list."""
    cons: list[cp.Constraint] = []

    if constraints.fully_invested:
        if constraints.leverage is not None:
            cons.append(cp.norm(weights, 1) <= float(constraints.leverage))
            cons.append(cp.sum(weights) == 1)
        else:
            cons.append(cp.sum(weights) == 1)

    lb = np.zeros(len(assets))
    ub = np.ones(len(assets))
    for i, asset in enumerate(assets):
        lo, hi = constraints.get_bounds(asset)
        lb[i] = lo
        ub[i] = hi
    cons.append(weights >= lb)
    cons.append(weights <= ub)

    if constraints.groups and constraints.group_bounds:
        grouped: dict[str, list[int]] = {}
        for i, asset in enumerate(assets):
            g = constraints.groups.get(asset)
            if g is not None:
                grouped.setdefault(g, []).append(i)
        for group, idx in grouped.items():
            if group in constraints.group_bounds:
                lo, hi = constraints.group_bounds[group]
                cons.append(cp.sum(weights[idx]) >= float(lo))
                cons.append(cp.sum(weights[idx]) <= float(hi))

    if constraints.previous_weights and constraints.turnover_limit is not None:
        prev = np.array(
            [float(constraints.previous_weights.get(a, 0.0)) for a in assets]
        )
        cons.append(cp.norm(weights - prev, 1) <= float(constraints.turnover_limit))

    if extra_constraints:
        cons.extend(extra_constraints)
    return cons
