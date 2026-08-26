"""Shared CVXPY constraint plumbing and solver dispatch."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from optimization_engine.optimizers.base import PortfolioConstraints

_LOG = logging.getLogger(__name__)

#: Tried in order. CLARABEL and ECOS are interior-point conic solvers that
#: handle these problems accurately; SCS and OSQP are first-order fallbacks
#: that converge on harder instances at looser tolerance.
SOLVER_FALLBACK = ("CLARABEL", "ECOS", "SCS", "OSQP")

_OK_STATUSES = ("optimal", "optimal_inaccurate")

# CVXPY warns on stderr whenever a solve returns ``optimal_inaccurate``. That
# is not actionable for a caller of this module: the fallback chain below
# treats an inaccurate answer as a *reason to try the next solver*, and when
# it does end up settling for one it says so through ``SolveInfo.status``, a
# logged warning, and the UI's compliance banner. Suppressing the duplicate
# has to happen with a process-wide filter rather than a context manager,
# because frontier sweeps and walk-forward runs solve on a thread pool and
# ``warnings.catch_warnings`` is not thread-safe -- scoping it per solve would
# leak across threads and could swallow unrelated warnings. The filter is
# matched on the exact message so nothing else is silenced; ``module`` is not
# usable here because CVXPY raises with a stacklevel that attributes the
# warning to this file rather than to its own.
warnings.filterwarnings(
    "ignore", message="Solution may be inaccurate", category=UserWarning
)


@dataclass(frozen=True)
class SolveInfo:
    """Which solver answered, how, and how long it took."""

    solver: str
    status: str
    solve_seconds: float
    objective_value: float | None
    attempts: tuple[str, ...] = ()

    @property
    def is_exact(self) -> bool:
        return self.status == "optimal"

    def as_dict(self) -> dict[str, object]:
        return {
            "solver": self.solver,
            "solver_status": self.status,
            "solve_seconds": self.solve_seconds,
            "objective_value": self.objective_value,
            "solvers_attempted": list(self.attempts),
        }


class SolverFailure(RuntimeError):
    """No solver could return a usable solution for the problem as posed."""

    def __init__(self, status: str, attempts: tuple[str, ...], detail: str = "") -> None:
        self.status = status
        self.attempts = attempts
        message = f"No solver produced a solution (last status: {status!r})."
        if status == "infeasible":
            message = (
                "The problem is infeasible: no allocation satisfies every "
                "constraint at once."
            )
        elif status == "unbounded":
            message = (
                "The problem is unbounded: the objective improves without "
                "limit. Add weight bounds or a budget constraint."
            )
        if attempts:
            message += f" Solvers tried: {', '.join(attempts)}."
        if detail:
            message += f" {detail}"
        super().__init__(message)


def _snapshot(problem: cp.Problem) -> dict[int, object]:
    """Capture the current value of every variable in ``problem``."""
    return {id(v): (v, None if v.value is None else np.array(v.value)) for v in problem.variables()}


def _restore(snapshot: dict[int, object]) -> None:
    """Put a snapshotted solution back onto the problem's variables.

    Needed because CVXPY overwrites ``variable.value`` on every solve: once a
    later solver in the fallback chain has run, the earlier (usable) answer is
    gone unless it was saved.
    """
    for variable, value in snapshot.values():
        variable.value = value


def solve_problem(
    problem: cp.Problem,
    solvers: tuple[str, ...] = SOLVER_FALLBACK,
    accept_inaccurate: bool = True,
) -> SolveInfo:
    """Solve ``problem``, walking a fallback chain, and report what happened.

    An ``optimal_inaccurate`` answer is *not* accepted straight away: the rest
    of the chain is tried first in case another solver converges properly, and
    the loose answer is only returned if nothing better turns up. Settling for
    the first solver's inaccurate result is how a portfolio ends up a few
    basis points outside its own constraints for no reason.

    Args:
        problem: The CVXPY problem to solve.
        solvers: Fallback chain, tried in order. Missing solvers are skipped.
        accept_inaccurate: Return a loose solution when no solver converges
            exactly. Set False to raise instead.

    Raises:
        SolverFailure: When every solver fails or reports a non-optimal
            status. The exception distinguishes *infeasible* (the constraints
            are impossible) from *unbounded* and from numerical failure,
            because the analyst's next action differs in each case.
    """
    attempts: list[str] = []
    last_status = "unknown"
    last_error: Exception | None = None
    inaccurate: tuple[SolveInfo, dict[int, object]] | None = None
    installed = set(cp.installed_solvers())

    candidates = [s for s in solvers if s in installed] or [None]
    for solver in candidates:
        attempts.append(solver or "default")
        start = time.perf_counter()
        try:
            if solver is None:
                problem.solve()
            else:
                problem.solve(solver=solver)
        except Exception as exc:  # numerical failure inside the solver
            last_error = exc
            _LOG.debug("Solver %s raised: %s", solver, exc)
            continue
        elapsed = time.perf_counter() - start
        last_status = str(problem.status)

        if last_status == "optimal":
            return SolveInfo(
                solver=solver or "default",
                status=last_status,
                solve_seconds=elapsed,
                objective_value=(
                    float(problem.value) if problem.value is not None else None
                ),
                attempts=tuple(attempts),
            )
        if last_status == "optimal_inaccurate" and inaccurate is None:
            inaccurate = (
                SolveInfo(
                    solver=solver or "default",
                    status=last_status,
                    solve_seconds=elapsed,
                    objective_value=(
                        float(problem.value) if problem.value is not None else None
                    ),
                    attempts=tuple(attempts),
                ),
                _snapshot(problem),
            )
            continue
        # infeasible / unbounded are properties of the problem, not the
        # solver -- no point trying the rest of the chain.
        if last_status in ("infeasible", "unbounded"):
            break

    if inaccurate is not None and accept_inaccurate:
        info, snapshot = inaccurate
        _restore(snapshot)
        _LOG.warning(
            "No solver converged exactly; falling back to %s's approximate "
            "solution. Treat the weights as indicative and check the "
            "constraint-compliance report.",
            info.solver,
        )
        return SolveInfo(
            solver=info.solver,
            status=info.status,
            solve_seconds=info.solve_seconds,
            objective_value=info.objective_value,
            attempts=tuple(attempts),
        )

    raise SolverFailure(
        last_status,
        tuple(attempts),
        detail=(f"Last error: {last_error}" if last_error is not None else ""),
    )


def bounds_arrays(
    assets: list[str], constraints: PortfolioConstraints
) -> tuple[np.ndarray, np.ndarray]:
    """Per-asset lower/upper weight bounds as aligned arrays."""
    lb = np.array([constraints.get_bounds(a)[0] for a in assets])
    ub = np.array([constraints.get_bounds(a)[1] for a in assets])
    return lb, ub


def group_index_map(
    assets: list[str], constraints: PortfolioConstraints
) -> dict[str, list[int]]:
    """Map each constrained group to the positions of its member assets."""
    grouped: dict[str, list[int]] = {}
    if not (constraints.groups and constraints.group_bounds):
        return grouped
    for i, asset in enumerate(assets):
        g = constraints.groups.get(asset)
        if g is not None and g in constraints.group_bounds:
            grouped.setdefault(g, []).append(i)
    return grouped


def build_constraints(
    weights: cp.Variable,
    assets: list[str],
    constraints: PortfolioConstraints,
    extra_constraints: list[cp.Constraint] | None = None,
) -> list[cp.Constraint]:
    """Translate a :class:`PortfolioConstraints` object into CVXPY constraints."""
    cons: list[cp.Constraint] = []

    if constraints.fully_invested:
        cons.append(cp.sum(weights) == 1)
    if constraints.leverage is not None:
        cons.append(cp.norm(weights, 1) <= float(constraints.leverage))

    lb, ub = bounds_arrays(assets, constraints)
    cons.append(weights >= lb)
    cons.append(weights <= ub)

    for group, idx in group_index_map(assets, constraints).items():
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


def build_scaled_constraints(
    y: cp.Variable,
    kappa: cp.Variable,
    assets: list[str],
    constraints: PortfolioConstraints,
) -> list[cp.Constraint]:
    """Constraints for the homogeneous ``w = y / κ`` reformulations.

    Max-Sharpe and max-diversification are both solved by minimizing a
    quadratic over a ray and normalizing afterwards. Every constraint that is
    *linear and homogeneous of degree one* in ``w`` carries over exactly once
    it is scaled by ``κ = Σy`` — so per-asset bounds and group budgets stay
    hard constraints instead of being applied by post-hoc projection.

    Turnover budgets do **not** carry over: ``‖w − w_prev‖₁ ≤ τ`` is affine,
    not homogeneous, so it is left to the caller to reject or warn.
    """
    cons: list[cp.Constraint] = [cp.sum(y) == kappa, kappa >= 1e-8]

    lb, ub = bounds_arrays(assets, constraints)
    cons.append(y >= cp.multiply(lb, kappa))
    cons.append(y <= cp.multiply(ub, kappa))

    for group, idx in group_index_map(assets, constraints).items():
        lo, hi = constraints.group_bounds[group]
        cons.append(cp.sum(y[idx]) >= float(lo) * kappa)
        cons.append(cp.sum(y[idx]) <= float(hi) * kappa)

    return cons
