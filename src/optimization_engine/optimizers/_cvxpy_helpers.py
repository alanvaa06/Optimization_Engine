"""Shared CVXPY constraint plumbing and solver dispatch."""

from __future__ import annotations

import logging
import threading
import time
import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from optimization_engine.constraints import layer_cvxpy_constraints
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
#
# That argument is unchanged, and so is the filter: it is still installed on
# the interpreter's global filter list and still outlives the solve that put
# it there. What moved is *when* it goes on. Installing it at import time made
# ``import optimization_engine`` -- a statement that solves nothing -- mutate a
# process-global belonging to the host application, which is not a library's
# to change until the library is actually asked to do something. So the install
# happens on the first ``solve_problem`` call instead, once, behind
# ``_FILTER_LOCK`` because that first call may well be N pool threads at once.
#
# One residual behaviour difference, deliberate: a consumer who calls
# ``warnings.resetwarnings()`` after solving wipes the filter, and under the
# lazy scheme the next solve reinstalls it, where the eager scheme would have
# left it off for the rest of the process. Reinstalling is the better of the
# two -- the filter exists to keep a solve quiet, and every solve should get
# the same treatment -- but it does mean ``resetwarnings()`` no longer holds
# against this module.
_FILTER_LOCK = threading.Lock()
_filter_installed = False


def _install_inaccurate_warning_filter() -> None:
    """Silence CVXPY's ``optimal_inaccurate`` warning, once per process.

    Idempotent and thread-safe: concurrent first solves add exactly one entry
    to ``warnings.filters`` between them. See the comment above for why the
    filter is process-wide rather than scoped to the solve.
    """
    global _filter_installed
    if _filter_installed:
        return
    with _FILTER_LOCK:
        if _filter_installed:
            return
        warnings.filterwarnings(
            "ignore", message="Solution may be inaccurate", category=UserWarning
        )
        _filter_installed = True


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
        """Whether the solver converged properly rather than approximately.

        Returns:
            ``True`` only for ``"optimal"``. An ``"optimal_inaccurate"`` answer is
            usable and is accepted, but it is not this.
        """
        return self.status == "optimal"

    def as_dict(self) -> dict[str, object]:
        """This record as a flat, JSON-serializable mapping.

        Returns:
            The solver that answered, its status, the wall time in seconds, the
            objective value, and every solver attempted along the way. Merged into
            a result's ``extras``, so it reaches the ``--json`` payload.
        """
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
        """Build the failure, interpreting the status where it can.

        Args:
            status: The last solver status. ``"infeasible"`` and ``"unbounded"``
                get a message explaining what to change; anything else is
                reported as-is.
            attempts: Every solver tried, in order. Appended to the message and
                kept on the exception as ``exc.attempts``.
            detail: Extra context appended verbatim.
        """
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


VariableSnapshot = dict[int, "tuple[cp.Variable, np.ndarray | None]"]


def _snapshot(problem: cp.Problem) -> VariableSnapshot:
    """Capture the current value of every variable in ``problem``."""
    return {
        id(v): (v, None if v.value is None else np.array(v.value))
        for v in problem.variables()
    }


def _restore(snapshot: VariableSnapshot) -> None:
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

    The first call in a process also installs the process-wide warnings filter
    described at the top of this module. Importing the package does not; only
    asking it to solve something does.

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
    _install_inaccurate_warning_filter()
    attempts: list[str] = []
    last_status = "unknown"
    last_error: Exception | None = None
    inaccurate: tuple[SolveInfo, VariableSnapshot] | None = None
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
    """Per-asset lower/upper weight bounds as aligned arrays.

    Args:
        assets: The universe, in the order the solve indexes it.
        constraints: The mandate whose bounds to read.

    Returns:
        A ``(lower, upper)`` pair of arrays, both aligned to ``assets``, in
        weight units.
    """
    lb = np.array([constraints.get_bounds(a)[0] for a in assets])
    ub = np.array([constraints.get_bounds(a)[1] for a in assets])
    return lb, ub


def group_index_map(
    assets: list[str], constraints: PortfolioConstraints
) -> dict[str, list[int]]:
    """Map each constrained group to the positions of its member assets.

    Covers the flat ``groups`` mapping only. Kept because callers outside this
    package use it; every optimizer in the engine now goes through
    :func:`layer_constraints`, which sees the whole layered policy.

    Args:
        assets: The universe, in the order the solve indexes it.
        constraints: The mandate whose ``groups`` to read.

    Returns:
        ``group -> positions into assets``, covering only the groups that
        carry a bound.
    """
    grouped: dict[str, list[int]] = {}
    if not (constraints.groups and constraints.group_bounds):
        return grouped
    for i, asset in enumerate(assets):
        g = constraints.groups.get(asset)
        if g is not None and g in constraints.group_bounds:
            grouped.setdefault(g, []).append(i)
    return grouped


def layer_constraints(
    weights: cp.Variable,
    assets: list[str],
    constraints: PortfolioConstraints,
    scale: cp.Variable | None = None,
) -> list[cp.Constraint]:
    """Every layered bucket budget, in weight space or in ray space.

    ``scale`` is ``None`` for the direct formulations and the ``κ`` variable
    for the homogeneous ``w = y/κ`` ones. See
    :func:`optimization_engine.constraints.layer_cvxpy_constraints` for why a
    percent-of-parent limit needs no rescaling in either.

    Args:
        weights: The CVXPY weight variable, or the ray variable ``y``.
        assets: The universe, in the order the solve indexes it.
        constraints: The mandate whose layers to translate.
        scale: The homogenizing variable ``κ``, or ``None`` in weight space.

    Returns:
        CVXPY constraints, one pair per bounded bucket. Empty when nothing is
        constrained above the per-asset level.
    """
    return layer_cvxpy_constraints(weights, assets, constraints.layers, scale=scale)


def build_constraints(
    weights: cp.Variable,
    assets: list[str],
    constraints: PortfolioConstraints,
    extra_constraints: list[cp.Constraint] | None = None,
    cov_matrix: np.ndarray | None = None,
) -> list[cp.Constraint]:
    """Translate a :class:`PortfolioConstraints` object into CVXPY constraints.

    Args:
        weights: The decision variable.
        assets: Column order the vectors are aligned to.
        constraints: The constraint set to translate.
        extra_constraints: Method-specific constraints to append.
        cov_matrix: Annualized covariance, aligned to ``assets``. Required
            only for a tracking-error budget, which is the one constraint
            that cannot be written from weights alone. Passing ``None`` while
            a budget is set raises rather than dropping it: a portfolio
            returned without the active-risk limit its mandate specifies is
            worse than an error.
    """
    cons: list[cp.Constraint] = []

    if constraints.fully_invested:
        cons.append(cp.sum(weights) == 1)
    if constraints.leverage is not None:
        cons.append(cp.norm(weights, 1) <= float(constraints.leverage))

    lb, ub = bounds_arrays(assets, constraints)
    cons.append(weights >= lb)
    cons.append(weights <= ub)

    cons.extend(layer_constraints(weights, assets, constraints))

    if constraints.previous_weights and constraints.turnover_limit is not None:
        prev = np.array(
            [float(constraints.previous_weights.get(a, 0.0)) for a in assets]
        )
        cons.append(cp.norm(weights - prev, 1) <= float(constraints.turnover_limit))

    cons.extend(benchmark_constraints(weights, assets, constraints, cov_matrix))

    if extra_constraints:
        cons.extend(extra_constraints)
    return cons


def benchmark_constraints(
    weights: cp.Variable,
    assets: list[str],
    constraints: PortfolioConstraints,
    cov_matrix: np.ndarray | None = None,
) -> list[cp.Constraint]:
    """Active-risk and active-share limits, relative to the benchmark.

    Both are convex in ``w``: tracking error is a quadratic form in the active
    weights and active share is an L1 norm of them, so neither costs the solve
    its convexity.

    Args:
        weights: The CVXPY weight variable.
        assets: The universe, in the order the solve indexes it.
        constraints: The mandate carrying the benchmark and its budgets.
        cov_matrix: Asset covariance, needed to measure tracking error. Its
            periodicity sets the units of ``max_tracking_error``.

    Returns:
        CVXPY constraints, or an empty list when no benchmark is set.

    Raises:
        ValueError: When a tracking-error budget is set without a covariance
            matrix to measure it with, or when either limit is negative.
    """
    benchmark = constraints.benchmark_vector(assets)
    if benchmark is None:
        if (
            constraints.max_tracking_error is not None
            or constraints.max_active_share is not None
        ):
            raise ValueError(
                "A tracking-error or active-share budget was set without "
                "benchmark_weights. Name the benchmark the limit is relative to."
            )
        return []

    cons: list[cp.Constraint] = []
    active = weights - benchmark

    if constraints.max_active_share is not None:
        limit = float(constraints.max_active_share)
        if limit < 0:
            raise ValueError(f"max_active_share must be non-negative; got {limit}.")
        # Active share is half the L1 distance, hence the factor of two.
        cons.append(cp.norm(active, 1) <= 2.0 * limit)

    if constraints.max_tracking_error is not None:
        te = float(constraints.max_tracking_error)
        if te < 0:
            raise ValueError(f"max_tracking_error must be non-negative; got {te}.")
        if cov_matrix is None:
            raise ValueError(
                "A tracking-error budget needs a covariance matrix. This "
                "optimizer does not have one, so the limit could not be "
                "imposed — use mean_variance, min_variance or "
                "active_mean_variance to bind it."
            )
        cons.append(cp.quad_form(active, cp.psd_wrap(cov_matrix)) <= te**2)

    return cons


def psd_sqrt(sigma: np.ndarray) -> np.ndarray:
    """Symmetric square root of a PSD matrix, tolerant of tiny negative eigenvalues.

    Cholesky would be faster but requires strict positive definiteness, and
    the engine's covariance estimates are routinely on the boundary — a
    detoned or shrunk matrix can carry eigenvalues at ``-1e-18``. Clipping
    them to zero and rebuilding from the eigendecomposition gives the nearest
    PSD square root instead of an exception.

    Args:
        sigma: A symmetric, positive semi-definite matrix.

    Returns:
        A symmetric matrix ``A`` with ``A @ A == sigma`` up to the clipping.
    """
    values, vectors = np.linalg.eigh(np.asarray(sigma, dtype=float))
    return vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def build_scaled_constraints(
    y: cp.Variable,
    kappa: cp.Variable,
    assets: list[str],
    constraints: PortfolioConstraints,
    cov_matrix: np.ndarray | None = None,
) -> list[cp.Constraint]:
    """Constraints for the homogeneous ``w = y / κ`` reformulations.

    Max-Sharpe and max-diversification are both solved by minimizing a
    quadratic over a ray and normalizing afterwards. Every constraint that is
    *linear and homogeneous of degree one* in ``w`` carries over exactly once
    it is scaled by ``κ = Σy`` — so per-asset bounds and every layer's bucket
    budgets stay hard constraints instead of being applied by post-hoc
    projection. A percent-of-parent limit needs no rescaling at all: both
    sides are homogeneous, so ``Σ_child y ≤ hi·Σ_parent y`` says the same
    thing in ray space as in weight space.

    The benchmark-relative limits carry over too, though they are not linear.
    With ``w − b = (y − κb)/κ`` and ``κ > 0``, an active-share cap becomes
    ``‖y − κb‖₁ ≤ 2·AS·κ`` and a tracking-error budget becomes the
    second-order cone ``‖Σ^½(y − κb)‖₂ ≤ TE·κ``. Both stay convex, so the
    tangency and maximum-diversification portfolios honour an active-risk
    mandate rather than reporting a violation after the fact.

    A gross-exposure cap is homogeneous too — ``‖y‖₁ ≤ L·κ`` — and carries
    over as a hard constraint.

    Two things do **not** carry over, and the caller has to say so (see
    :func:`homogeneous_ignored_constraints`). A turnover budget
    ``‖w − w_prev‖₁ ≤ τ`` is affine in ``w`` with a constant term that has
    no ``κ`` to absorb it. And an *open* budget — ``fully_invested=False`` —
    cannot be expressed on a ray at all: the ray fixes only the direction,
    and ``w = y/κ`` always sums to one.

    Args:
        y: The unnormalized ray variable.
        kappa: The scaling variable, ``κ = Σy``.
        assets: Column order the vectors are aligned to.
        constraints: The constraint set to translate.
        cov_matrix: Annualized covariance aligned to ``assets``. Required
            only for a tracking-error budget.

    Raises:
        ValueError: When a benchmark-relative limit is set without the
            benchmark, or a tracking-error budget without a covariance matrix.
    """
    cons: list[cp.Constraint] = [cp.sum(y) == kappa, kappa >= 1e-8]

    lb, ub = bounds_arrays(assets, constraints)
    cons.append(y >= cp.multiply(lb, kappa))
    cons.append(y <= cp.multiply(ub, kappa))

    if constraints.leverage is not None:
        cons.append(cp.norm(y, 1) <= float(constraints.leverage) * kappa)

    cons.extend(layer_constraints(y, assets, constraints, scale=kappa))

    benchmark = constraints.benchmark_vector(assets)
    if benchmark is None:
        if (
            constraints.max_tracking_error is not None
            or constraints.max_active_share is not None
        ):
            raise ValueError(
                "A tracking-error or active-share budget was set without "
                "benchmark_weights. Name the benchmark the limit is relative to."
            )
        return cons

    active = y - kappa * benchmark
    if constraints.max_active_share is not None:
        limit = float(constraints.max_active_share)
        if limit < 0:
            raise ValueError(f"max_active_share must be non-negative; got {limit}.")
        cons.append(cp.norm(active, 1) <= 2.0 * limit * kappa)
    if constraints.max_tracking_error is not None:
        te = float(constraints.max_tracking_error)
        if te < 0:
            raise ValueError(f"max_tracking_error must be non-negative; got {te}.")
        if cov_matrix is None:
            raise ValueError(
                "A tracking-error budget needs a covariance matrix, and this "
                "optimizer was not given one."
            )
        cons.append(cp.norm(psd_sqrt(cov_matrix) @ active, 2) <= te * kappa)

    return cons


def homogeneous_ignored_constraints(
    constraints: PortfolioConstraints, method: str
) -> list[str]:
    """Name the constraints a ray-space solve cannot honour, and warn once each.

    The homogeneous reformulations behind max-Sharpe and max-diversification
    carry every constraint that scales with the book (see
    :func:`build_scaled_constraints`) and none that does not. This is the
    one place that decides which those are, so the two optimizers cannot
    disagree about it, and so a mandate that is only partly honoured says so
    in ``result.extras["ignored_constraints"]`` rather than in a violation
    the reader has to notice.

    Args:
        constraints: The mandate being translated.
        method: The optimizer's display name, for the warning text.

    Returns:
        The names of the ignored constraints, in a fixed order — empty when
        everything carried over. Each one also raises a ``UserWarning``.
    """
    ignored: list[str] = []
    if constraints.turnover_limit is not None:
        ignored.append("turnover_limit")
        warnings.warn(
            f"{method} ignores the turnover budget: the solve works on a "
            "scaled ray where a turnover constraint is not well defined. "
            "Use mean_variance with a return target to respect turnover.",
            stacklevel=4,
        )
    if not constraints.fully_invested:
        ignored.append("fully_invested")
        warnings.warn(
            f"{method} ignores fully_invested=False: the solve fixes only the "
            "direction of the book, and the weights it returns always sum to "
            "one. Size the result against cash separately.",
            stacklevel=4,
        )
    return ignored
