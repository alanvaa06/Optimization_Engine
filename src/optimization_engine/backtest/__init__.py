"""The backtesting stack: a stateless simulation core and the layers on it.

The boundary that organizes this package is the one between *simulating* and
*judging*. At the centre sits a pure function — returns and weights in, a
:class:`~optimization_engine.backtest.results.RunResult` out — that performs
no I/O, keeps no state between calls, and is deterministic given its inputs.
Everything else is arranged around it:

``spec``
    The declarative, validated description of a run, with a provenance hash.
    Data, not code, so it can be serialized, diffed, and put in a grid.

``costs`` / ``calendar``
    The pieces the core consults: what a trade costs, and when it happens.
    Both are pure and both are replaceable.

``runner`` / ``walkforward``
    The simulation itself, in-sample and out. The walk-forward runner only
    ever hands the optimizer returns from strictly before the decision date.

``results``
    The canonical bundle every downstream layer reads: NAV, returns, held
    weights, targets, trades, costs, and the metadata that hashes them.

``tca`` / ``positions``
    Analytics over one run: where the cost went, and how the individual
    positions actually did.

``sweep`` / ``holdout``
    Analytics over *many* runs — the part that measures the search rather
    than the strategy. The sweep evaluates whole grids and refuses to rank
    them; the holdout keeps a segment of history untouched and writes down
    every time it is looked at.

``tearsheet``
    The assembled reading, with the caveats attached to the numbers rather
    than to a footnote nobody reaches.
"""

from optimization_engine.backtest.calendar import execution_positions, rebalance_dates
from optimization_engine.backtest.costs import (
    ContextRequest,
    CostModel,
    CostQuote,
    LinearCost,
    MarketContext,
    SquareRootImpactCost,
    ZeroCost,
    build_cost_model,
    context_request,
    trailing_dollar_volume,
    trailing_volatilities,
)
from optimization_engine.backtest.holdout import (
    DEFAULT_AUDIT_PATH,
    REPEATED,
    SHIFTED_HOLDOUT,
    HoldoutOutcome,
    HoldoutViolationError,
    assert_within_holdout,
    final_holdout_run,
    gate_returns,
    holdout_segment,
    read_audit_log,
)
from optimization_engine.backtest.positions import (
    PositionEpisode,
    PositionStats,
    compute_position_stats,
    episodes_frame,
    position_episodes,
)
from optimization_engine.backtest.results import RunMeta, RunResult, compute_result_hash
from optimization_engine.backtest.runner import run_backtest
from optimization_engine.backtest.spec import (
    MIN_ADV_CAPITAL,
    PARTICIPATION_SOURCES,
    REBALANCE_DESCRIPTIONS,
    BacktestSpec,
    CostSpec,
    RebalanceFrequency,
    SpecValidationError,
)
from optimization_engine.backtest.sweep import (
    HARD_CELL_CAP,
    GridCell,
    SweepResults,
    SweepSpec,
    SweepValidationError,
    expand_grid,
    run_sweep,
    sweep_from_optimizers,
)
from optimization_engine.backtest.tca import TcaPanel, compute_tca, cost_by_asset
from optimization_engine.backtest.tearsheet import Tearsheet, build_tearsheet
from optimization_engine.backtest.walkforward import WalkForwardRun, walk_forward_run

__all__ = [
    "DEFAULT_AUDIT_PATH",
    "HARD_CELL_CAP",
    "REBALANCE_DESCRIPTIONS",
    "REPEATED",
    "SHIFTED_HOLDOUT",
    "BacktestSpec",
    "ContextRequest",
    "CostModel",
    "CostQuote",
    "MIN_ADV_CAPITAL",
    "PARTICIPATION_SOURCES",
    "CostSpec",
    "GridCell",
    "HoldoutOutcome",
    "HoldoutViolationError",
    "LinearCost",
    "MarketContext",
    "PositionEpisode",
    "PositionStats",
    "RebalanceFrequency",
    "RunMeta",
    "RunResult",
    "SpecValidationError",
    "SquareRootImpactCost",
    "SweepResults",
    "SweepSpec",
    "SweepValidationError",
    "TcaPanel",
    "Tearsheet",
    "WalkForwardRun",
    "ZeroCost",
    "assert_within_holdout",
    "build_cost_model",
    "context_request",
    "build_tearsheet",
    "compute_position_stats",
    "compute_result_hash",
    "compute_tca",
    "cost_by_asset",
    "episodes_frame",
    "execution_positions",
    "expand_grid",
    "final_holdout_run",
    "gate_returns",
    "holdout_segment",
    "position_episodes",
    "read_audit_log",
    "rebalance_dates",
    "run_backtest",
    "run_sweep",
    "sweep_from_optimizers",
    "trailing_dollar_volume",
    "trailing_volatilities",
    "walk_forward_run",
]
