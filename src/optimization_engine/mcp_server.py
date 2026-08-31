"""An MCP server exposing the engine as tools.

Run it with ``optengine-mcp``, or point an MCP client at
``python -m optimization_engine.mcp_server``. It speaks stdio, which is what
desktop clients launch.

Every tool returns the same payloads the CLI's ``--json`` emits, from
:mod:`optimization_engine.reporting.payloads`. That is the point of having
written the contract as a module: this server is a transport, not a second
serialisation layer that could drift from the first.

**What this server can reach.** It runs as a local process with the
permissions of whoever launched it, and two tools take a filesystem path —
``config_path`` for a mandate and ``prices_path`` for a price panel. They
read those files. Nothing here writes to disk, fetches over the network, or
reaches a data provider that needs a key; a caller that wants live data
should ingest it separately and hand over a file. Start from ``sample=True``,
which needs no paths at all.

**Solving blocks.** A large mean-variance solve is seconds of CPU, and these
tools are synchronous, so a client waiting on one waits for the whole thing.
That is the honest behaviour for an optimizer; it is not a hung server.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from optimization_engine._optional import require
from optimization_engine.reporting.payloads import (
    SCHEMA_VERSION,
    backtest_payload,
    check_payload,
    describe_payload,
    optimization_payload,
)

INSTRUCTIONS = """\
A multi-asset portfolio optimization engine.

Call `describe_optimizer` before building a config: a method that does not
support a constraint ignores it silently rather than rejecting it, so the
support flags are the only way to know a turnover budget or a benchmark
limit will actually bind.

Then `check_mandate` before `optimize`. It reports whether the constraint
set is satisfiable at all and what expected-return range is reachable,
which turns an infeasible solve from an error into a number you can act on.

Read the diagnostics, not just the weights. `effective_n` counts positions
by capital and `effective_n_risk` counts them by risk contribution; a book
that is wide by capital and narrow by risk looks diversified in a weights
table and is not.

Data: pass `sample=True` for a built-in synthetic panel, or `prices_path`
for a CSV, Excel or Parquet file of prices (not returns — the engine
differences them).
"""


def _build() -> tuple[Any, type[Exception]]:
    """Build the server, or explain what is missing.

    Returns the server and the SDK's ``ToolError``. That second value
    matters more than it looks: it is the only exception class whose
    message reaches the client. Anything else is wrapped in
    ``UnexpectedToolError`` and the caller sees "Error executing tool
    optimize" with the actual reason discarded — which for an agent means
    a failure it cannot act on or explain.

    The ``mcp`` SDK is an extra and needs Python 3.10+, one minor version
    above this package's own floor. Someone on 3.9 gets a resolver error
    from pip rather than anything from here, which is why the message names
    the version too.
    """
    mcpserver = require(
        "mcp.server.mcpserver",
        extra="mcp",
        purpose="running the MCP server (needs Python 3.10 or newer)",
    )
    exceptions = require(
        "mcp.server.mcpserver.exceptions",
        extra="mcp",
        purpose="running the MCP server",
    )
    from optimization_engine import __version__

    server = mcpserver.MCPServer(
        name="optimization-engine",
        title="Portfolio Optimization Engine",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    return server, exceptions.ToolError


mcp, ToolError = _build()


def _panel(sample: bool, prices_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve a price panel and its returns from exactly one source.

    Both are returned because the two are used for different things —
    data-quality analysis reads prices, the optimizers read returns — and
    reconstructing one from the other would introduce a rounding difference
    between what was checked and what was solved.

    Every entry point here takes *prices*, because that is what a data file
    holds and differencing them in one place keeps the two from being
    confused. Handing this a returns file silently builds a portfolio on
    second differences, so the parameter is named for what it wants.
    """
    from optimization_engine.data.loader import load_prices, prices_to_returns, sample_dataset

    if sample and prices_path:
        raise ToolError(
            "Pass either sample=True or prices_path, not both — otherwise it "
            "is ambiguous which panel the result describes."
        )
    if sample:
        prices = sample_dataset()
    elif prices_path:
        try:
            prices = load_prices(prices_path)
        except FileNotFoundError as exc:
            raise ToolError(f"No such price file: {prices_path}") from exc
        except Exception as exc:  # unreadable, wrong shape, unsupported suffix
            raise ToolError(
                f"Could not read {prices_path} as a price panel: {exc}"
            ) from exc
    else:
        raise ToolError(
            "No data given. Pass sample=True for the built-in panel, or "
            "prices_path pointing at a CSV, Excel or Parquet file of prices."
        )
    return prices, prices_to_returns(prices).dropna(how="any")


def _config(config_path: str | None, optimizer: str | None) -> Any:
    """Load a mandate, or build the smallest one that will solve."""
    from optimization_engine.config import EngineConfig, OptimizerSpec, load_config

    if config_path:
        try:
            config = load_config(config_path)
        except FileNotFoundError as exc:
            raise ToolError(f"No such config file: {config_path}") from exc
        except Exception as exc:
            raise ToolError(f"Could not read {config_path}: {exc}") from exc
        if optimizer:
            config.optimizer = OptimizerSpec(name=optimizer)
        return config
    return EngineConfig(optimizer=OptimizerSpec(name=optimizer or "risk_parity"))


@mcp.tool(
    title="List optimizers",
    description=(
        "Every optimization method this engine can run, with a one-line "
        "summary of each. Start here when you do not know which method a "
        "mandate needs."
    ),
)
def list_optimizers() -> dict[str, Any]:
    """Enumerate the available methods."""
    from optimization_engine.optimizers.factory import available_optimizers
    from optimization_engine.optimizers.requirements import requirements_for

    entries = []
    for name in available_optimizers():
        req = requirements_for(name)
        entries.append({"name": name, "label": req.label, "summary": req.summary})
    return {
        "schema_version": SCHEMA_VERSION,
        "optimizers": entries,
    }


@mcp.tool(
    title="Describe an optimizer",
    description=(
        "What one method needs as input and which constraints it will "
        "honour. Read this before building a config: a constraint a method "
        "does not support is ignored silently, not rejected."
    ),
)
def describe_optimizer(name: str) -> dict[str, Any]:
    """Report one optimizer's contract.

    Args:
        name: An optimizer name, e.g. ``"risk_parity"``. Use
            ``list_optimizers`` if you are unsure.
    """
    from optimization_engine.optimizers.factory import available_optimizers
    from optimization_engine.optimizers.requirements import requirements_for

    try:
        req = requirements_for(name)
    except KeyError as exc:
        # Naming the alternatives turns a dead end into the next call.
        raise ToolError(
            f"No optimizer named {name!r}. Available: "
            + ", ".join(available_optimizers())
        ) from exc
    return describe_payload(req)


@mcp.tool(
    title="Check a mandate before solving",
    description=(
        "Pre-flight the data and the constraints. Reports whether the "
        "constraint set can be satisfied at all, what expected-return range "
        "is reachable, and whether the covariance estimate is conditioned "
        "well enough to be worth optimizing on."
    ),
)
def check_mandate(
    config_path: str | None = None,
    sample: bool = False,
    prices_path: str | None = None,
    optimizer: str | None = None,
) -> dict[str, Any]:
    """Say whether this mandate is solvable, before spending a solve on it.

    Args:
        config_path: Path to a YAML mandate. Omit for a minimal default.
        sample: Use the built-in synthetic price panel.
        prices_path: A CSV, Excel or Parquet file of prices.
        optimizer: Override the config's method.

    Returns:
        A payload whose ``ready`` field is the single boolean to branch on.
    """
    from optimization_engine.data.covariance import (
        covariance_diagnostics,
        covariance_from_config,
    )
    from optimization_engine.data.quality import analyze_prices
    from optimization_engine.engine import resolve_expected_returns
    from optimization_engine.optimizers.factory import (
        constraints_from_config,
        effective_expected_returns,
    )
    from optimization_engine.optimizers.feasibility import analyze_feasibility

    config = _config(config_path, optimizer)
    prices, returns = _panel(sample, prices_path)

    quality = analyze_prices(prices, periods_per_year=config.periods_per_year)
    cov = covariance_from_config(returns, config)
    diagnostics = covariance_diagnostics(
        cov, len(returns), config.covariance_method, config.ewma_lambda
    )
    assets = list(returns.columns)
    # Must be the vector `optimize` will use, or this tool validates a
    # different mandate from the one that gets solved.
    mu = resolve_expected_returns(config, returns, cov)
    feasibility = analyze_feasibility(
        assets,
        constraints_from_config(config, assets),
        expected_returns=effective_expected_returns(config, cov, mu),
        cov_matrix=cov,
    )
    return check_payload(quality, feasibility, diagnostics)


@mcp.tool(
    title="Optimize a portfolio",
    description=(
        "Solve for weights and report what they rest on: which solver "
        "answered, whether the constraints held, how well-conditioned the "
        "covariance estimate was, and how concentrated the book is in risk "
        "rather than capital. Blocks while solving."
    ),
)
def optimize(
    config_path: str | None = None,
    sample: bool = False,
    prices_path: str | None = None,
    optimizer: str | None = None,
) -> dict[str, Any]:
    """Run one optimization end to end.

    Args:
        config_path: Path to a YAML mandate. Omit for a minimal default.
        sample: Use the built-in synthetic price panel.
        prices_path: A CSV, Excel or Parquet file of prices.
        optimizer: Override the config's method — see ``list_optimizers``.

    Returns:
        Weights under ``weights``, and the evidence under ``diagnostics``,
        ``covariance`` and ``feasibility``. Reporting the weights alone
        discards what distinguishes this engine.
    """
    from optimization_engine.engine import run_engine
    from optimization_engine.optimizers.feasibility import InfeasibleConstraintsError

    config = _config(config_path, optimizer)
    _, returns = _panel(sample, prices_path)
    try:
        run = run_engine(returns, config)
    except InfeasibleConstraintsError as exc:
        # Anticipated, and the most useful failure this tool has: the
        # message names which constraints cannot hold together. Call
        # `check_mandate` to get the same finding as data rather than text.
        raise ToolError(f"The mandate has no solution: {exc}") from exc
    return optimization_payload(run)


@mcp.tool(
    title="Backtest the process",
    description=(
        "Walk the allocation process forward over history, priced with "
        "commission, slippage and square-root market impact. Returns the "
        "spec and result hashes that identify the run, so a re-run can be "
        "told apart from a real change. Blocks while running."
    ),
)
def backtest(
    config_path: str | None = None,
    sample: bool = False,
    prices_path: str | None = None,
    optimizer: str | None = None,
    lookback: int = 504,
    rebalance_every: int = 63,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> dict[str, Any]:
    """Simulate the process, rather than replaying a fitted allocation.

    Args:
        config_path: Path to a YAML mandate. Omit for a minimal default.
        sample: Use the built-in synthetic price panel.
        prices_path: A CSV, Excel or Parquet file of prices.
        optimizer: Override the config's method.
        lookback: Estimation window, in periods.
        rebalance_every: Periods between rebalances.
        commission_bps: Broker commission, in basis points of traded value.
        slippage_bps: Slippage, in basis points of traded value.

    Returns:
        ``spec_hash`` and ``result_hash`` identify the run; ``degradations``
        names anywhere the simulation had to fall back, which is where its
        costs are optimistic.
    """
    from optimization_engine.backtest.spec import BacktestSpec, CostSpec
    from optimization_engine.backtest.tearsheet import build_tearsheet
    from optimization_engine.engine import run_engine

    config = _config(config_path, optimizer)
    _, returns = _panel(sample, prices_path)
    spec = BacktestSpec(
        costs=CostSpec(commission_bps=commission_bps, slippage_bps=slippage_bps),
    )
    # Feasibility is checked by `check_mandate`; re-running it here would
    # reject a mandate the walk-forward could still say something useful
    # about on the windows where it does solve.
    run = run_engine(returns, config, check_feasibility=False)
    walk = run.walk_forward_run(
        lookback=lookback,
        rebalance_every=rebalance_every,
        spec=spec,
    )
    return backtest_payload(
        walk.run, tearsheet=build_tearsheet(walk.run, returns)
    )


def main() -> None:
    """Console-script entry point: serve over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
