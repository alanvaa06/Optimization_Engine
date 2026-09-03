"""The post-solve mandate audit: a public surface over the compliance check.

Every solve already runs :func:`~optimization_engine.optimizers.diagnostics.check_constraints`
on the way out — box bounds, the budget, the sign convention, gross exposure,
every layer of the policy, the turnover budget, tracking error and active
share. What it did not have was a *name*. The result carried a list of
sentences under ``extras["violations"]``, which is fine to read and useless to
act on: a caller wanting the size of the breach had to parse it back out of
prose, and a caller wanting the run to stop had no way to say so.

This module is that name. :func:`audit_weights` answers "does this book obey
this mandate?" for weights from anywhere — a solve, a backtest's schedule, a
spreadsheet someone emailed — and :class:`AuditReport` is the answer, with the
breaches as objects rather than strings. :class:`MandateViolationError` is the
gate: under ``strict_mandate`` a reported breach becomes a raised one.

The checking itself is not reimplemented here. There is exactly one compliance
check in the package and this delegates to it, because two implementations of
"is this book compliant" is precisely the defect the audit exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from optimization_engine.optimizers.base import PortfolioConstraints
from optimization_engine.optimizers.diagnostics import (
    DEFAULT_TOLERANCE,
    ConstraintViolation,
    check_constraints,
)

__all__ = [
    "DEFAULT_TOLERANCE",
    "AuditReport",
    "MandateViolationError",
    "audit_weights",
]


@dataclass(frozen=True)
class AuditReport:
    """What a mandate audit found, and the tolerance it found it at.

    Attributes:
        violations: Every breach beyond ``tolerance``, as
            :class:`~optimization_engine.optimizers.diagnostics.ConstraintViolation`
            objects — the same type the post-solve diagnostics already carry, so
            a caller reading one is reading the other.
        tolerance: The drift below which a difference was treated as
            floating-point noise rather than a breach. Recorded because a clean
            report means nothing without it: "compliant at 1e-6" and "compliant
            at 1e-2" are different claims.
    """

    violations: tuple[ConstraintViolation, ...] = field(default_factory=tuple)
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def is_clean(self) -> bool:
        """Whether the audited weights breach nothing beyond :attr:`tolerance`.

        Returns:
            ``True`` when :attr:`violations` is empty. Note what this does *not*
            say: an audit run without a covariance matrix cannot check a
            tracking-error budget, and comes back clean because it did not look.
            :func:`audit_weights` documents which limits need what.
        """
        return not self.violations

    @property
    def worst(self) -> ConstraintViolation | None:
        """The largest breach by weight, or ``None`` when there is none.

        Returns:
            The violation with the greatest ``magnitude``. Which constraint is
            worst is the first question asked of a failed audit, and ordering by
            the size of the breach answers it in the same units as the weights.
        """
        if not self.violations:
            return None
        return max(self.violations, key=lambda v: v.magnitude)

    def describe(self) -> str:
        """A human-readable verdict: one line per breach, or a clean bill.

        Returns:
            ``"No constraint is breached beyond a tolerance of 1e-06."`` when
            clean, otherwise a header naming the count followed by one indented
            line per violation, each giving the limit, the actual figure and the
            distance between them.
        """
        if not self.violations:
            return (
                f"No constraint is breached beyond a tolerance of {self.tolerance:g}."
            )
        head = (
            f"{len(self.violations)} constraint(s) breached beyond a tolerance "
            f"of {self.tolerance:g}:"
        )
        return "\n".join([head] + [f"  - {v.describe()}" for v in self.violations])


class MandateViolationError(ValueError):
    """The solved book breaches the mandate, and the caller asked to be told loudly.

    Raised only under ``strict_mandate``. Without it a breach is *reported* —
    on ``result.audit``, in ``result.extras["violations"]``, and in the run's
    warnings — because the methods that apply bounds by projection cannot
    always satisfy a mandate and refusing would make them unusable. Strict mode
    is for the caller who would rather have no book than a non-compliant one.

    It is recoverable in the sense that matters: the mandate is not
    unsatisfiable, this *method* could not satisfy it. Loosen the limit the
    report names, or pick a method whose ``bounds_mode`` is ``"hard"`` and have
    it enforced inside the convex program instead of projected onto afterwards.
    """

    def __init__(self, report: AuditReport) -> None:
        """Build the error and attach the audit that explains it.

        Args:
            report: The audit that found the breach. It stays reachable as
                ``exc.report``, and its :meth:`AuditReport.describe` output
                becomes the exception's message.
        """
        self.report = report
        super().__init__("The solved weights breach the mandate:\n" + report.describe())


def audit_weights(
    weights: pd.Series,
    assets: Sequence[str] | None = None,
    constraints: PortfolioConstraints | None = None,
    cov_matrix: pd.DataFrame | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> AuditReport:
    """Check an allocation against a mandate and report every breach.

    A thin, named wrapper over
    :func:`~optimization_engine.optimizers.diagnostics.check_constraints`, which
    already checks the box bounds, the budget, the sign convention, gross
    exposure, every layer of the allocation policy, the turnover budget, active
    share and tracking error. This adds two things that matter to a caller
    holding weights from outside a solve:

    * **The universe is the mandate's, not the vector's.** ``check_constraints``
      iterates the weights it is given, so an asset the solve dropped — or one a
      spreadsheet never listed — silently escapes its *lower* bound. Passing
      ``assets`` reindexes the book onto the declared universe at zero first, so
      a missing 5% floor is a breach rather than an absence.
    * **An object instead of a list.** The result is an :class:`AuditReport`
      that knows whether it is clean, which breach is worst, and how to describe
      itself.

    Args:
        weights: The allocation to check, indexed by asset.
        assets: The universe the mandate is written over. ``None`` audits
            exactly the assets ``weights`` carries.
        constraints: The mandate. ``None`` means an unconstrained long-only book
            summing to one — the same default a
            :class:`~optimization_engine.optimizers.base.PortfolioConstraints`
            constructed with no arguments expresses, and *not* the same thing as
            "do not check".
        cov_matrix: Needed for a tracking-error budget, which is a risk
            statement rather than a weights one. Without it that single limit
            goes unchecked and the report says nothing about it — which is why
            the methods that cannot impose it are the same ones that cannot
            verify it.
        tolerance: Drift below this is floating-point noise, not a breach. The
            default is
            :data:`~optimization_engine.optimizers.diagnostics.DEFAULT_TOLERANCE`
            — ``1e-6``, the same threshold ``_clean_weights`` calls dust, so the
            audit and the weight cleaning agree on what counts as zero. The
            design note that proposed this function suggested ``1e-5``; keeping
            the package's own constant is the point, since two tolerances would
            let a book be compliant by one measure and not the other. Override
            it here when a mandate is genuinely stated to a looser precision.

    Returns:
        An :class:`AuditReport`. Empty violations mean every limit that *could*
        be checked was met.
    """
    constraints = constraints if constraints is not None else PortfolioConstraints()
    book = weights.astype(float)
    if assets is not None:
        book = book.reindex([str(a) for a in assets]).fillna(0.0)
    violations = check_constraints(
        book, constraints, tolerance=tolerance, cov_matrix=cov_matrix
    )
    return AuditReport(violations=tuple(violations), tolerance=float(tolerance))
