"""Pre-solve feasibility analysis.

``status=infeasible`` is the least useful thing an optimizer can tell an
analyst: it names no constraint, suggests no fix, and looks identical whether
the cause is a typo in one bound or a genuinely impossible mandate.

This module answers the question the solver won't: *which* constraint makes
the problem impossible, and what has to change. The cheap structural checks
(budget vs. bounds, group budgets vs. member bounds) run without a solver;
the reachable-return range is computed with two small LPs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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

    if constraints.groups and constraints.group_bounds:
        members: dict[str, list[int]] = {}
        for i, a in enumerate(assets):
            g = constraints.groups.get(a)
            if g is not None:
                members.setdefault(g, []).append(i)

        total_group_min = 0.0
        total_group_max = 0.0
        covered = set()
        for group, (glo, ghi) in constraints.group_bounds.items():
            idx = members.get(group, [])
            if not idx:
                issues.append(
                    FeasibilityIssue(
                        code="empty_group",
                        message=(
                            f"Group {group!r} has bounds but no assets assigned "
                            "to it."
                        ),
                        suggestion=(
                            f"Assign assets to {group!r} in the Group column, or "
                            "remove its bounds."
                        ),
                        fatal=float(glo) > _TOL,
                    )
                )
                continue
            covered.update(idx)
            member_max = float(ub[idx].sum())
            member_min = float(lb[idx].sum())
            if member_max < float(glo) - 1e-8:
                issues.append(
                    FeasibilityIssue(
                        code="group_min_unreachable",
                        message=(
                            f"Group {group!r} needs at least {float(glo):.2%} but "
                            f"its members' caps only add up to {member_max:.2%}."
                        ),
                        suggestion=(
                            f"Raise the per-asset maximums inside {group!r}, or "
                            f"lower the group minimum below {member_max:.2%}."
                        ),
                    )
                )
            if member_min > float(ghi) + 1e-8:
                issues.append(
                    FeasibilityIssue(
                        code="group_max_unreachable",
                        message=(
                            f"Group {group!r} is capped at {float(ghi):.2%} but its "
                            f"members' minimums already force {member_min:.2%}."
                        ),
                        suggestion=(
                            f"Lower the per-asset minimums inside {group!r}, or "
                            f"raise the group maximum above {member_min:.2%}."
                        ),
                    )
                )
            if float(glo) > float(ghi) + 1e-8:
                issues.append(
                    FeasibilityIssue(
                        code="inverted_group_bounds",
                        message=(
                            f"Group {group!r} has a minimum ({float(glo):.2%}) above "
                            f"its maximum ({float(ghi):.2%})."
                        ),
                        suggestion="Swap or correct the group's Min/Max weights.",
                    )
                )
            total_group_min += float(glo)
            total_group_max += float(ghi)

        if constraints.fully_invested and covered and len(covered) == len(assets):
            if total_group_min > 1.0 + 1e-8:
                issues.append(
                    FeasibilityIssue(
                        code="group_mins_exceed_budget",
                        message=(
                            f"Group minimums sum to {total_group_min:.2%}, more than "
                            "the 100% budget."
                        ),
                        suggestion=(
                            "Lower the group minimums so they sum to at most 100%."
                        ),
                    )
                )
            if total_group_max < 1.0 - 1e-8:
                issues.append(
                    FeasibilityIssue(
                        code="group_maxes_below_budget",
                        message=(
                            f"Group maximums sum to only {total_group_max:.2%}, so "
                            "the portfolio cannot be fully invested."
                        ),
                        suggestion="Raise the group maximums so they sum to at least 100%.",
                    )
                )

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


def reachable_return_range(
    expected_returns: pd.Series,
    constraints: PortfolioConstraints,
    assets: list[str] | None = None,
) -> tuple[float, float] | None:
    """Solve two LPs for the min and max expected return the constraints allow.

    This is the range a target return must fall inside. Returns ``None`` when
    the constraint set is infeasible or no LP solver is available.
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
        problem = cp.Problem(sense(mu @ w), build_constraints(w, assets, constraints))
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
        rng = reachable_return_range(expected_returns, constraints, assets)
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
