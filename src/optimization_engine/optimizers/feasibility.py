"""Pre-solve feasibility analysis.

``status=infeasible`` is the least useful thing an optimizer can tell an
analyst: it names no constraint, suggests no fix, and looks identical whether
the cause is a typo in one bound or a genuinely impossible mandate.

This module answers the question the solver won't: *which* constraint makes
the problem impossible, and what has to change. The cheap structural checks
(budget vs. bounds, every layer's bucket budgets vs. member bounds) run
without a solver;
the reachable-return range is computed with two small LPs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optimization_engine.constraints import (
    ConstraintLayer,
    parent_bucket_map,
    resolve_parent,
)
from optimization_engine.optimizers.base import PortfolioConstraints

_TOL = 1e-9


@dataclass(frozen=True)
class FeasibilityIssue:
    """One reason the problem cannot be solved as specified."""

    code: str
    message: str
    #: Concrete, imperative suggestion the analyst can act on.
    suggestion: str
    fatal: bool = True


@dataclass(frozen=True)
class FeasibilityReport:
    """Result of a pre-solve feasibility analysis.

    Attributes:
        issues: Problems found, fatal ones first.
        min_return: Lowest expected return reachable under the constraints.
        max_return: Highest expected return reachable under the constraints.
        min_variance_return: Expected return of the minimum-variance portfolio;
            the point below which the frontier stops being efficient.
    """

    issues: tuple[FeasibilityIssue, ...] = field(default_factory=tuple)
    min_return: float | None = None
    max_return: float | None = None
    min_variance_return: float | None = None

    @property
    def is_feasible(self) -> bool:
        return not any(i.fatal for i in self.issues)

    @property
    def fatal_issues(self) -> tuple[FeasibilityIssue, ...]:
        return tuple(i for i in self.issues if i.fatal)

    @property
    def warnings(self) -> tuple[FeasibilityIssue, ...]:
        return tuple(i for i in self.issues if not i.fatal)

    def describe(self) -> str:
        """One multi-line, human-readable explanation of everything found."""
        if not self.issues:
            return "Constraints are feasible."
        return "\n".join(
            f"• {i.message}\n  → {i.suggestion}" for i in self.issues
        )


def _structural_issues(
    assets: list[str], constraints: PortfolioConstraints
) -> list[FeasibilityIssue]:
    """Checks that need no solver: budget arithmetic on the bounds themselves."""
    issues: list[FeasibilityIssue] = []
    lb = np.array([constraints.get_bounds(a)[0] for a in assets], dtype=float)
    ub = np.array([constraints.get_bounds(a)[1] for a in assets], dtype=float)

    inverted = [a for a, lo, hi in zip(assets, lb, ub) if lo > hi + _TOL]
    if inverted:
        issues.append(
            FeasibilityIssue(
                code="inverted_bounds",
                message=(
                    f"{len(inverted)} asset(s) have a minimum weight above their "
                    f"maximum: {', '.join(inverted[:5])}"
                    + (" …" if len(inverted) > 5 else "")
                ),
                suggestion="Fix the Min/Max columns so Min ≤ Max for every asset.",
            )
        )

    if constraints.fully_invested:
        if lb.sum() > 1.0 + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="min_weights_exceed_budget",
                    message=(
                        f"Minimum weights sum to {lb.sum():.2%}, which is more "
                        "than the 100% budget."
                    ),
                    suggestion=(
                        f"Lower the minimums by at least {lb.sum() - 1.0:.2%} in "
                        "total, or allow leverage."
                    ),
                )
            )
        if ub.sum() < 1.0 - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="max_weights_below_budget",
                    message=(
                        f"Maximum weights sum to only {ub.sum():.2%}, so the "
                        "portfolio cannot reach 100% invested."
                    ),
                    suggestion=(
                        f"Raise the caps by at least {1.0 - ub.sum():.2%} in total, "
                        "or add more assets to the universe."
                    ),
                )
            )

    if constraints.long_only and (lb < -_TOL).any():
        shorts = [a for a, lo in zip(assets, lb) if lo < -_TOL]
        issues.append(
            FeasibilityIssue(
                code="negative_bound_long_only",
                message=(
                    "Long-only is on but these assets have a negative minimum "
                    f"weight: {', '.join(shorts[:5])}"
                ),
                suggestion="Set those minimums to 0, or turn long-only off.",
            )
        )

    for layer in constraints.layers:
        issues.extend(_layer_issues(layer, constraints, assets, lb, ub))

    if constraints.leverage is not None and constraints.leverage < 1.0 - _TOL:
        issues.append(
            FeasibilityIssue(
                code="leverage_below_budget",
                message=(
                    f"Gross exposure is capped at {constraints.leverage:.2f}× while "
                    "the portfolio must be 100% invested."
                ),
                suggestion="Set the gross-exposure cap to at least 1.0.",
            )
        )

    return issues


def _layer_issues(
    layer: ConstraintLayer,
    constraints: PortfolioConstraints,
    assets: list[str],
    lb: np.ndarray,
    ub: np.ndarray,
) -> list[FeasibilityIssue]:
    """Structural checks for one layer of the allocation policy.

    Each layer is a partition of (part of) the universe with a budget per
    bucket, so the same arithmetic applies whether the layer is asset class,
    sub-asset class or currency. The messages name the layer, because "Group
    'EM' is capped at 20%" is ambiguous once three layers can own a bucket
    called EM.
    """
    issues: list[FeasibilityIssue] = []
    members: dict[str, list[int]] = {}
    for i, asset in enumerate(assets):
        bucket = layer.assignments.get(str(asset))
        if bucket is not None:
            members.setdefault(bucket, []).append(i)

    where = f"{layer.name} · " if layer.name else ""
    parent_layer = None
    parent_map: dict[str, str] = {}
    if layer.is_relative:
        parent_layer = resolve_parent(layer, constraints.layers)
        if parent_layer is None:
            return [
                FeasibilityIssue(
                    code="missing_parent_layer",
                    message=(
                        f"Layer {layer.name!r} states its limits as a share of "
                        f"{layer.parent!r}, but there is no layer by that name."
                    ),
                    suggestion=(
                        "Point the layer at an existing one, or switch its "
                        "limits to percent-of-portfolio."
                    ),
                )
            ]
        parent_map, ambiguous = parent_bucket_map(layer, parent_layer, assets)
        for bucket, parents in ambiguous.items():
            issues.append(
                FeasibilityIssue(
                    code="ambiguous_parent_bucket",
                    message=(
                        f"{where}{bucket!r} is capped as a share of "
                        f"{parent_layer.name!r}, but its assets sit in more "
                        f"than one: {', '.join(parents)}."
                    ),
                    suggestion=(
                        f"Split {bucket!r} so all its assets share a parent, or "
                        "switch this layer to percent-of-portfolio."
                    ),
                )
            )

    total_min = 0.0
    total_max = 0.0
    covered: set[int] = set()
    for bucket, (blo, bhi) in layer.limits.items():
        blo, bhi = float(blo), float(bhi)
        idx = members.get(bucket, [])
        if not idx:
            issues.append(
                FeasibilityIssue(
                    code="empty_group",
                    message=(
                        f"{where}{bucket!r} has limits but no assets assigned "
                        "to it."
                    ),
                    suggestion=(
                        f"Assign assets to {bucket!r}, or remove its limits."
                    ),
                    fatal=blo > _TOL,
                )
            )
            continue
        covered.update(idx)
        if blo > bhi + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="inverted_group_bounds",
                    message=(
                        f"{where}{bucket!r} has a minimum ({blo:.2%}) above its "
                        f"maximum ({bhi:.2%})."
                    ),
                    suggestion="Swap or correct that bucket's Min/Max weights.",
                )
            )

        if layer.is_relative:
            if bhi > 1.0 + 1e-8:
                issues.append(
                    FeasibilityIssue(
                        code="relative_cap_above_parent",
                        message=(
                            f"{where}{bucket!r} is capped at {bhi:.0%} of its "
                            "parent, which is more than the whole parent."
                        ),
                        suggestion=(
                            "A share-of-parent cap above 100% never binds; "
                            "lower it or switch the layer to "
                            "percent-of-portfolio."
                        ),
                        fatal=False,
                    )
                )
            continue

        member_max = float(ub[idx].sum())
        member_min = float(lb[idx].sum())
        if member_max < blo - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="group_min_unreachable",
                    message=(
                        f"{where}{bucket!r} needs at least {blo:.2%} but its "
                        f"members' caps only add up to {member_max:.2%}."
                    ),
                    suggestion=(
                        f"Raise the per-asset maximums inside {bucket!r}, or "
                        f"lower its minimum below {member_max:.2%}."
                    ),
                )
            )
        if member_min > bhi + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="group_max_unreachable",
                    message=(
                        f"{where}{bucket!r} is capped at {bhi:.2%} but its "
                        f"members' minimums already force {member_min:.2%}."
                    ),
                    suggestion=(
                        f"Lower the per-asset minimums inside {bucket!r}, or "
                        f"raise its maximum above {member_min:.2%}."
                    ),
                )
            )
        total_min += blo
        total_max += bhi

    if layer.is_relative:
        issues.extend(_relative_layer_issues(layer, parent_map, constraints, assets))
        return issues

    if constraints.fully_invested and covered and len(covered) == len(assets):
        if total_min > 1.0 + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="group_mins_exceed_budget",
                    message=(
                        f"{where}bucket minimums sum to {total_min:.2%}, more "
                        "than the 100% budget."
                    ),
                    suggestion=(
                        "Lower the minimums on this layer so they sum to at "
                        "most 100%."
                    ),
                )
            )
        if total_max < 1.0 - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="group_maxes_below_budget",
                    message=(
                        f"{where}bucket maximums sum to only {total_max:.2%}, "
                        "so the portfolio cannot be fully invested."
                    ),
                    suggestion=(
                        "Raise the maximums on this layer so they sum to at "
                        "least 100%, or leave some assets out of it."
                    ),
                )
            )
    return issues


def _relative_layer_issues(
    layer: ConstraintLayer,
    parent_map: dict[str, str],
    constraints: PortfolioConstraints,
    assets: list[str],
) -> list[FeasibilityIssue]:
    """Whether a percent-of-parent layer can fill its parent at all.

    Sub-buckets that cap out below 100% of the parent while covering all of it
    do not make the problem infeasible — the parent simply cannot be filled —
    but that is almost always a typo (30/30 inside a sleeve the allocator
    means to fill), so it is reported as a warning with the arithmetic shown.
    """
    issues: list[FeasibilityIssue] = []
    if constraints.long_only is False:
        issues.append(
            FeasibilityIssue(
                code="relative_layer_shorts",
                message=(
                    f"Layer {layer.name!r} is a share of its parent while short "
                    "positions are allowed."
                ),
                suggestion=(
                    "A share-of-parent limit assumes the parent sleeve is "
                    "positive; with shorts it can invert. Use "
                    "percent-of-portfolio limits instead."
                ),
                fatal=False,
            )
        )
    by_parent: dict[str, list[str]] = {}
    for bucket, up in parent_map.items():
        if bucket in layer.limits:
            by_parent.setdefault(up, []).append(bucket)
    parent_layer = resolve_parent(layer, constraints.layers)
    for up, children in by_parent.items():
        covers_parent = parent_layer is not None and all(
            layer.assignments.get(str(a)) in layer.limits
            for a in assets
            if parent_layer.assignments.get(str(a)) == up
        )
        if not covers_parent:
            continue
        cap_total = sum(float(layer.limits[c][1]) for c in children)
        floor_total = sum(float(layer.limits[c][0]) for c in children)
        if cap_total < 1.0 - 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="relative_caps_below_parent",
                    message=(
                        f"Inside {up!r}, the {layer.name} caps sum to "
                        f"{cap_total:.0%} of the sleeve, so at most that much "
                        "of it can be filled."
                    ),
                    suggestion=(
                        f"Raise the {layer.name} caps inside {up!r} so they sum "
                        "to at least 100%, or lower the cap on "
                        f"{up!r} itself to match."
                    ),
                    fatal=False,
                )
            )
        if floor_total > 1.0 + 1e-8:
            issues.append(
                FeasibilityIssue(
                    code="relative_floors_exceed_parent",
                    message=(
                        f"Inside {up!r}, the {layer.name} minimums sum to "
                        f"{floor_total:.0%} of the sleeve — more than the "
                        "sleeve itself."
                    ),
                    suggestion=(
                        f"Lower the {layer.name} minimums inside {up!r} so they "
                        "sum to at most 100%."
                    ),
                )
            )
    return issues


def reachable_return_range(
    expected_returns: pd.Series,
    constraints: PortfolioConstraints,
    assets: list[str] | None = None,
    cov_matrix: pd.DataFrame | None = None,
) -> tuple[float, float] | None:
    """Solve two programs for the min and max expected return the constraints allow.

    This is the range a target return must fall inside. Returns ``None`` when
    the constraint set is infeasible or no solver is available.

    Args:
        expected_returns: The vector the target is expressed in.
        constraints: The constraint set about to be solved.
        assets: Universe order. Defaults to the returns' index.
        cov_matrix: Needed only when a tracking-error budget is set — that
            budget genuinely narrows the reachable range, and omitting it
            would report a range the solve cannot deliver.
    """
    import cvxpy as cp

    assets = assets or list(expected_returns.index)
    mu = expected_returns.reindex(assets).fillna(0.0).values.astype(float)
    n = len(assets)
    if n == 0:
        return None

    from optimization_engine.optimizers._cvxpy_helpers import build_constraints

    bounds: list[float] = []
    for sense in (cp.Minimize, cp.Maximize):
        w = cp.Variable(n)
        sigma = None
        if cov_matrix is not None:
            sigma = (
                cov_matrix.reindex(assets, axis=0).reindex(assets, axis=1).values
            )
        problem = cp.Problem(
            sense(mu @ w),
            build_constraints(w, assets, constraints, cov_matrix=sigma),
        )
        try:
            problem.solve()
        except Exception:
            return None
        if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            return None
        bounds.append(float(mu @ np.asarray(w.value).flatten()))
    lo, hi = bounds
    return (min(lo, hi), max(lo, hi))


def min_variance_return(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    constraints: PortfolioConstraints,
) -> float | None:
    """Expected return of the constrained global minimum-variance portfolio.

    Frontier targets below this point are *inefficient*: the same volatility
    buys a higher return above the GMV, so plotting them as "efficient" is
    misleading.
    """
    from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer

    try:
        result = MinVarianceOptimizer(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            constraints=constraints,
        ).optimize()
    except Exception:
        return None
    return float(result.expected_return)


def analyze_feasibility(
    assets: list[str],
    constraints: PortfolioConstraints,
    expected_returns: pd.Series | None = None,
    cov_matrix: pd.DataFrame | None = None,
    check_targets: bool = True,
) -> FeasibilityReport:
    """Full pre-solve feasibility analysis.

    Runs the structural checks first; only if those pass does it spend LP
    solves on the reachable-return range, since an infeasible constraint set
    would make the LPs meaningless anyway.

    Args:
        assets: The universe, in the order the optimizer will see it.
        constraints: The constraint object about to be solved.
        expected_returns: Needed to check a return target and report the
            reachable range.
        cov_matrix: Needed to locate the minimum-variance return.
        check_targets: Validate ``target_return`` / ``target_volatility``
            against what the constraints can actually reach.
    """
    issues = _structural_issues(assets, constraints)
    lo = hi = gmv = None

    if not any(i.fatal for i in issues) and expected_returns is not None:
        rng = reachable_return_range(
            expected_returns, constraints, assets, cov_matrix=cov_matrix
        )
        if rng is None:
            issues.append(
                FeasibilityIssue(
                    code="lp_infeasible",
                    message=(
                        "The constraint set has no solution — no weight vector "
                        "satisfies every bound and budget at once."
                    ),
                    suggestion=(
                        "Relax the tightest constraints: widen the per-asset "
                        "caps, or loosen the group budgets."
                    ),
                )
            )
        else:
            lo, hi = rng
            if cov_matrix is not None:
                gmv = min_variance_return(expected_returns, cov_matrix, constraints)

            target = constraints.target_return
            if check_targets and target is not None:
                if target > hi + 1e-8:
                    issues.append(
                        FeasibilityIssue(
                            code="target_return_too_high",
                            message=(
                                f"Target return of {target:.2%} is above the "
                                f"{hi:.2%} maximum these constraints can reach."
                            ),
                            suggestion=(
                                f"Lower the target to at most {hi:.2%}, or raise the "
                                "caps on the highest-return assets."
                                + (
                                    " Note that Black-Litterman optimizes "
                                    "against its equilibrium posterior, which "
                                    "usually sits well below historical means "
                                    "— so a target that suits mean-variance "
                                    "can be unreachable here."
                                    if expected_returns is not None
                                    and getattr(expected_returns, "name", "")
                                    == "black_litterman_posterior"
                                    else ""
                                )
                            ),
                        )
                    )
                elif target < lo - 1e-8:
                    issues.append(
                        FeasibilityIssue(
                            code="target_return_too_low",
                            message=(
                                f"Target return of {target:.2%} is below the "
                                f"{lo:.2%} minimum these constraints allow."
                            ),
                            suggestion=(
                                f"Raise the target to at least {lo:.2%}, or relax the "
                                "minimum weights on low-return assets."
                            ),
                        )
                    )
                elif gmv is not None and target < gmv - 1e-8:
                    issues.append(
                        FeasibilityIssue(
                            code="target_return_inefficient",
                            message=(
                                f"Target return of {target:.2%} sits below the "
                                f"minimum-variance portfolio's {gmv:.2%}."
                            ),
                            suggestion=(
                                f"The result will be solvable but dominated — a "
                                f"target of {gmv:.2%} gives more return at less "
                                "risk. Raise the target."
                            ),
                            fatal=False,
                        )
                    )

    if (
        check_targets
        and constraints.target_volatility is not None
        and cov_matrix is not None
        and not any(i.fatal for i in issues)
    ):
        from optimization_engine.optimizers.mean_variance import MinVarianceOptimizer

        try:
            gmv_result = MinVarianceOptimizer(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                constraints=constraints,
            ).optimize()
            floor_vol = float(gmv_result.expected_volatility)
            if constraints.target_volatility < floor_vol - 1e-8:
                issues.append(
                    FeasibilityIssue(
                        code="target_vol_below_gmv",
                        message=(
                            f"Target volatility of {constraints.target_volatility:.2%} "
                            f"is below the {floor_vol:.2%} floor set by the "
                            "minimum-variance portfolio."
                        ),
                        suggestion=(
                            f"Raise the target to at least {floor_vol:.2%}, or add a "
                            "lower-risk asset (e.g. cash) to the universe."
                        ),
                    )
                )
        except Exception:
            pass

    issues.sort(key=lambda i: not i.fatal)
    return FeasibilityReport(
        issues=tuple(issues),
        min_return=lo,
        max_return=hi,
        min_variance_return=gmv,
    )


class InfeasibleConstraintsError(ValueError):
    """Raised with a full :class:`FeasibilityReport` attached."""

    def __init__(self, report: FeasibilityReport) -> None:
        self.report = report
        super().__init__(
            "The constraints cannot be satisfied:\n" + report.describe()
        )
