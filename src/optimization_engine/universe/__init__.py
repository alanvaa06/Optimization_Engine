"""The point-in-time universe: who was investable, and when.

The rest of this library takes its universe from the columns of a returns
frame, which is the universe as it looks *today*. Every backtest run that way
is run on the survivors: names that were delisted, acquired or dropped from the
index never appear, and the ones that were added appear from the first day of
the sample rather than from the day they listed. That is survivorship bias and
look-ahead in the same panel, and no amount of care in the optimizer removes
it.

This package adds the missing axis. Three pieces, layered:

``signal``
    :class:`~optimization_engine.universe.signal.Signal` — a ``date × asset``
    frame with **three** states, because a rule with a warm-up does not know
    whether a name was eligible on day one. Kleene logic throughout.

``eligibility``
    :class:`~optimization_engine.universe.eligibility.Eligibility` — the
    membership rules (thresholds, ranks, prior-window rolling statistics,
    hysteresis, reconstitution calendars), plus
    :meth:`~optimization_engine.universe.eligibility.Eligibility.to_mask`, the
    single place the third state collapses, under a policy with no default.

``classification``
    :class:`~optimization_engine.universe.classification.Classification` —
    labels that know when they became true, so a sector cap in 2019 uses the
    2019 sector map.

Nothing here needs a solver, and nothing here reads a file. The integration
points are :func:`~optimization_engine.backtest.runner.run_backtest` and
:func:`~optimization_engine.backtest.walkforward.walk_forward_run`, which both
take a ``universe`` and the policy to read it under, and
:meth:`~optimization_engine.constraints.ConstraintLayer.from_classification`,
which snaps a layer to a date.
"""

from optimization_engine.universe.classification import (
    Classification,
    LabelRecord,
)
from optimization_engine.universe.eligibility import (
    COMPARISONS,
    MASK_POLICIES,
    ROLLING_AGGS,
    Eligibility,
    Rule,
    collapse,
    point_in_time_mask,
)
from optimization_engine.universe.signal import (
    BOOLEAN_DTYPE,
    Signal,
    UniverseError,
    to_boolean_frame,
)

__all__ = [
    "BOOLEAN_DTYPE",
    "COMPARISONS",
    "MASK_POLICIES",
    "ROLLING_AGGS",
    "Classification",
    "Eligibility",
    "LabelRecord",
    "Rule",
    "Signal",
    "UniverseError",
    "collapse",
    "point_in_time_mask",
    "to_boolean_frame",
]
