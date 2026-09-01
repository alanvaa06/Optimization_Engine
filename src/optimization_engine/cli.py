"""Command-line entrypoint: ``optengine``."""

from __future__ import annotations

import argparse
import contextlib
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from optimization_engine.benchmark import BenchmarkError, BenchmarkSpec
from optimization_engine.config import load_config
from optimization_engine.data.fred import FREDError, load_fred_series
from optimization_engine.data.fx import FXError
from optimization_engine.data.loader import load_prices, prices_to_returns, sample_dataset
from optimization_engine.data.yahoo import YahooFinanceError, load_prices_yahoo
from optimization_engine.engine import (
    apply_fx_conversion,
    resolve_expected_returns,
    run_engine,
)
from optimization_engine.ingest import (
    IngestError,
    IngestRequest,
    describe_providers,
    ingest,
    load_dotenv,
)
from optimization_engine.ingest import fields as ingest_fields
from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
from optimization_engine.optimizers.factory import available_optimizers
from optimization_engine.optimizers.requirements import requirements_for
from optimization_engine.reporting.exporters import run_sheets, write_excel_report
from optimization_engine.reporting.payloads import (
    SCHEMA_VERSION,
    backtest_payload,
    check_payload,
    describe_payload,
    optimization_payload,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optengine",
        description="Multi-asset portfolio optimization engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    optimize = sub.add_parser("optimize", help="Run an optimization end-to-end.")
    optimize.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the result as JSON on stdout instead of a formatted report. Human-readable narration moves to stderr, so stdout stays parseable."
        ),
    )
    optimize.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    optimize.add_argument("--prices", help="Excel/CSV/Parquet file of prices.")
    optimize.add_argument("--sheet", default="Precios", help="Excel sheet name.")
    optimize.add_argument("--sample", action="store_true", help="Use built-in sample data.")
    optimize.add_argument(
        "--yahoo",
        help="Download prices from Yahoo Finance. "
             "Pass tickers as a comma- or space-separated list.",
    )
    optimize.add_argument(
        "--yahoo-period", default="5y",
        help="Yahoo period when --yahoo is used (default: 5y).",
    )
    optimize.add_argument("--yahoo-start", help="Yahoo start date (YYYY-MM-DD).")
    optimize.add_argument("--yahoo-end", help="Yahoo end date (YYYY-MM-DD).")
    optimize.add_argument(
        "--base-currency",
        help="Override config.base_currency. Conversion uses FRED FX rates.",
    )
    _add_ingest_arguments(optimize)
    optimize.add_argument("--output", default="outputs.xlsx", help="Output Excel path.")
    optimize.add_argument("--frontier", action="store_true", help="Also compute the frontier.")
    optimize.add_argument("--frontier-points", type=int, default=25)
    optimize.add_argument(
        "--walk-forward", action="store_true",
        help="Also run an out-of-sample walk-forward evaluation of the config.",
    )
    optimize.add_argument(
        "--lookback", type=int,
        help="Walk-forward estimation window in periods (default: two years).",
    )
    optimize.add_argument(
        "--rebalance-every", type=int,
        help="Walk-forward periods between re-solves (default: one quarter).",
    )
    optimize.add_argument(
        "--cost-bps", type=float, default=0.0,
        help="One-way transaction cost in basis points, applied to the backtest.",
    )
    optimize.add_argument(
        "--resample", type=int, metavar="N",
        help="Estimate the frontier's confidence band from N resampled "
             "histories. Requires --frontier.",
    )
    optimize.add_argument(
        "--denoise", action="store_true",
        help="Filter the covariance's noise eigenvalues through the "
             "Marchenko-Pastur cutoff before optimizing.",
    )
    optimize.add_argument(
        "--detone", type=int, default=None, metavar="K",
        help="Remove the K leading eigenvectors (the market component) after "
             "denoising. Makes the covariance singular — use only with the "
             "clustering methods (hrp, herc, nco).",
    )
    optimize.add_argument(
        "--trials", type=int, default=1, metavar="N",
        help="How many configurations you tried before settling on this one. "
             "Used to deflate the walk-forward Sharpe for selection bias; "
             "leaving it at 1 claims you tried exactly one.",
    )
    optimize.add_argument(
        "--mcos", type=int, metavar="N",
        help="Run Monte Carlo Optimization Selection over N simulated "
             "histories, ranking the methods by how reliably each recovers "
             "the allocation the fitted distribution implies.",
    )
    optimize.add_argument(
        "--mcos-methods",
        default="mean_variance,min_variance,hrp,herc,nco",
        help="Comma-separated optimizer names for --mcos.",
    )
    optimize.add_argument(
        "--benchmark", metavar="SPEC",
        help="Benchmark to measure — and optionally optimize — against. Use "
             "'equal_weight' for 1/N, an asset name for a single-asset index, "
             "or 'none' to override the config. Omit to use the config's own "
             "benchmark block.",
    )
    optimize.add_argument(
        "--max-tracking-error", type=float, metavar="TE",
        help="Cap annualized tracking error against the benchmark "
             "(0.03 = 300bp). Imposed inside the solve by the mean-variance "
             "family, mean-CVaR/CDaR and active_mean_variance.",
    )
    optimize.add_argument(
        "--max-active-share", type=float, metavar="AS",
        help="Cap active share against the benchmark (0.4 = 40%%).",
    )
    optimize.add_argument(
        "--strict", action="store_true",
        help="Refuse to run when the constraints are infeasible, naming the "
             "constraint instead of letting the solver fail.",
    )

    backtest = sub.add_parser(
        "backtest",
        help="Walk-forward the configured process, price the trading, and "
             "report what the search cost. Optionally sweep a grid.",
    )
    backtest.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the result as JSON on stdout instead of a formatted report. Human-readable narration moves to stderr, so stdout stays parseable."
        ),
    )
    backtest.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    backtest.add_argument("--prices", help="Excel/CSV/Parquet file of prices.")
    backtest.add_argument("--sheet", default="Precios", help="Excel sheet name.")
    backtest.add_argument("--sample", action="store_true", help="Use built-in sample data.")
    backtest.add_argument(
        "--yahoo", help="Comma- or space-separated tickers to download from Yahoo."
    )
    backtest.add_argument(
        "--yahoo-period", default="10y", help="Yahoo lookback when no dates are given."
    )
    backtest.add_argument("--yahoo-start", help="Yahoo start date (YYYY-MM-DD).")
    backtest.add_argument("--yahoo-end", help="Yahoo end date (YYYY-MM-DD).")
    backtest.add_argument(
        "--lookback", type=int, metavar="N",
        help="Estimation window in periods. Defaults to two years.",
    )
    backtest.add_argument(
        "--rebalance-every", type=int, metavar="N",
        help="Periods between re-solves — the re-optimization cadence. "
             "Defaults to one quarter.",
    )
    backtest.add_argument(
        "--rebalance", default="none",
        choices=["none", "daily", "weekly", "monthly", "quarterly", "annual"],
        help="How often the book is traded back to the current target "
             "*between* re-solves — the rebalancing cadence, which is a "
             "separate decision from --rebalance-every. Defaults to 'none': "
             "hold each solution untouched until the next one and let the "
             "weights drift. A committee that re-solves quarterly but "
             "rebalances monthly wants --rebalance-every 63 --rebalance "
             "monthly on a daily panel.",
    )
    backtest.add_argument(
        "--expanding", action="store_true",
        help="Grow the estimation window from the start instead of rolling it.",
    )
    backtest.add_argument(
        "--commission-bps", type=float, default=0.0, metavar="BPS",
        help="One-way broker commission on traded notional.",
    )
    backtest.add_argument(
        "--slippage-bps", type=float, default=0.0, metavar="BPS",
        help="One-way spread cost on traded notional.",
    )
    backtest.add_argument(
        "--impact-eta", type=float, default=0.0, metavar="ETA",
        help="Square-root market-impact coefficient. Non-zero makes cost grow "
             "with the square root of trade size, which is the only way "
             "capacity shows up in a backtest.",
    )
    backtest.add_argument(
        "--impact-participation-source", default="fixed", choices=["fixed", "adv"],
        help=(
            "Where the impact model's participation rate comes from. 'fixed' "
            "(default) uses --impact-participation and needs no volume data "
            "at all, which is what lets an index universe be backtested. "
            "'adv' derives it from traded volume, falling back to the fixed "
            "rate — and saying so — for any asset that has none."
        ),
    )
    backtest.add_argument(
        "--impact-adv-share", type=float, default=0.10, metavar="S",
        help="Share of an asset's average daily traded notional this book is "
             "willing to be, under --impact-participation-source adv.",
    )
    backtest.add_argument(
        "--impact-adv-lookback", type=int, default=21, metavar="N",
        help="Trailing periods averaged when computing ADV.",
    )
    backtest.add_argument(
        "--initial-capital", type=float, default=1.0, metavar="NAV",
        help=(
            "Starting NAV, in the currency the prices are quoted in. Cosmetic "
            "for returns, but required by --impact-participation-source adv: "
            "capacity is a currency amount, so the fund's size is what decides "
            "whether a name's daily volume is deep or thin for this book."
        ),
    )
    backtest.add_argument(
        "--impact-participation", type=float, default=0.05, metavar="Q",
        help="Fraction of the book tradable in one name, in one period, "
             "without impact. Smaller means a thinner market.",
    )
    backtest.add_argument(
        "--execution-lag", type=int, default=1, metavar="N",
        help="Periods between a decision and its fill. Defaults to 1 — a desk "
             "does not trade on a close it has not seen. Pass 0 for the "
             "conventional (optimistic) same-period fill.",
    )
    backtest.add_argument(
        "--holdout", metavar="YYYY-MM-DD",
        help="Withhold everything after this date from the walk-forward, then "
             "evaluate on it once and append the visit to the audit log.",
    )
    backtest.add_argument(
        "--audit-log", default="runs/holdout_audit.jsonl",
        help="Where the holdout audit trail is appended.",
    )
    backtest.add_argument(
        "--sweep", metavar="PATH=V1,V2",
        action="append",
        help="Sweep a config path over values, e.g. "
             "'optimizer.name=min_variance,risk_parity'. Repeat for a grid. "
             "Every cell is walk-forwarded and the trial count is carried "
             "into the deflated Sharpe.",
    )
    backtest.add_argument(
        "--output", help="Optional Excel path for the tearsheet frames."
    )

    _add_ingest_arguments(backtest)

    check = sub.add_parser(
        "check",
        help="Validate data and constraints without solving. Reports data "
             "quality, covariance conditioning and constraint feasibility.",
    )
    check.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the result as JSON on stdout instead of a formatted report. Human-readable narration moves to stderr, so stdout stays parseable."
        ),
    )
    check.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    check.add_argument("--prices", help="Excel/CSV/Parquet file of prices.")
    check.add_argument("--sheet", default="Precios", help="Excel sheet name.")
    check.add_argument("--sample", action="store_true", help="Use built-in sample data.")
    check.add_argument(
        "--denoise", action="store_true",
        help="Report the conditioning of the denoised covariance instead.",
    )
    check.add_argument(
        "--detone", type=int, default=None, metavar="K",
        help="Remove K leading eigenvectors after denoising.",
    )
    check.add_argument(
        "--base-currency", help="Override the config's base currency for FX conversion."
    )
    check.add_argument(
        "--benchmark",
        help="Benchmark to check the mandate against: a kind or an asset name.",
    )
    check.add_argument("--max-tracking-error", type=float, default=None)
    check.add_argument("--max-active-share", type=float, default=None)

    _add_ingest_arguments(check)

    describe = sub.add_parser(
        "describe", help="Explain one optimizer: what it needs and what it assumes."
    )
    describe.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the result as JSON on stdout instead of a formatted report. Human-readable narration moves to stderr, so stdout stays parseable."
        ),
    )
    describe.add_argument("name", help="Optimizer name (see list-optimizers).")

    sub.add_parser("list-optimizers", help="List available optimizer names.")

    providers = sub.add_parser(
        "providers",
        help="List data providers, what each can serve, and whether its key is set.",
    )
    providers.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    providers.add_argument(
        "--env-file", help="Load API keys from this .env file before reporting."
    )

    ingest_cmd = sub.add_parser(
        "ingest",
        help="Fetch a price panel from a provider and write it to disk.",
    )
    _add_ingest_arguments(ingest_cmd)
    ingest_cmd.add_argument(
        "--output", default="prices.csv",
        help="Where to write the close panel (.csv, .xlsx or .parquet).",
    )
    ingest_cmd.add_argument(
        "--volume-output",
        help=(
            "Also write the volume panel here. Skipped with a note when the "
            "universe carries no volume, which is the norm for indices."
        ),
    )

    fred = sub.add_parser("fred", help="Fetch one or more FRED series and write to disk.")
    fred.add_argument("series", help="Comma- or space-separated series ids (e.g. 'DGS10,VIXCLS').")
    fred.add_argument("--start", help="Start date (YYYY-MM-DD).")
    fred.add_argument("--end", help="End date (YYYY-MM-DD).")
    fred.add_argument("--output", default="fred_data.csv", help="Output CSV path.")

    sample = sub.add_parser("sample-data", help="Write a synthetic price panel to disk.")
    sample.add_argument("--output", default="data/sample/sample_prices.csv")
    sample.add_argument("--periods", type=int, default=252 * 8)

    return parser


def _cmd_optimize(args: argparse.Namespace) -> int:
    from optimization_engine.optimizers.feasibility import InfeasibleConstraintsError

    inputs = _prepare_inputs(args)
    if isinstance(inputs, int):
        return inputs
    config, returns, quality = inputs.config, inputs.returns, inputs.quality
    for issue in quality.errors:
        print(f"Data error — {issue.describe()}", file=sys.stderr)
    for issue in quality.warnings:
        print(f"Data warning — {issue.describe()}", file=sys.stderr)
    if quality.errors and args.strict:
        print(
            "Refusing to optimize on data with errors. Drop --strict to "
            "proceed anyway.",
            file=sys.stderr,
        )
        return 2

    try:
        run = run_engine(
            returns,
            config,
            build_frontier=args.frontier,
            n_frontier_points=args.frontier_points,
            raise_on_infeasible=args.strict,
        )
    except InfeasibleConstraintsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except SolverFailure as exc:
        print(f"Optimization failed: {exc}", file=sys.stderr)
        if config.max_tracking_error is not None or config.max_active_share is not None:
            print(
                "  A tracking-error or active-share budget is in force. A "
                "benchmark holding an asset your bounds cap below its index "
                "weight sets a floor on tracking error that no allocation can "
                "go below — raise the limit, or relax the bound.",
                file=sys.stderr,
            )
        return 2

    for warning in run.warnings:
        print(f"Warning — {warning}", file=sys.stderr)

    walk_forward = None
    if args.walk_forward:
        try:
            walk_forward = run.walk_forward(
                lookback=args.lookback,
                rebalance_every=args.rebalance_every,
                transaction_cost_bps=args.cost_bps,
            )
        except ValueError as exc:
            print(f"Walk-forward skipped: {exc}", file=sys.stderr)

    uncertainty = None
    if args.resample:
        if not args.frontier:
            print(
                "--resample needs --frontier; skipping the uncertainty band.",
                file=sys.stderr,
            )
        else:
            from optimization_engine.resampling import bootstrap_frontier

            try:
                uncertainty = bootstrap_frontier(
                    returns, config,
                    n_draws=int(args.resample),
                    n_points=max(args.frontier_points // 2, 6),
                )
                print(f"  {uncertainty.summary()}")
            except ValueError as exc:
                print(f"Frontier resampling skipped: {exc}", file=sys.stderr)

    print(
        f"{config.optimizer.name}: expected return "
        f"{run.result.expected_return:.2%}, volatility "
        f"{run.result.expected_volatility:.2%}, Sharpe "
        f"{run.result.sharpe_ratio:.2f}"
    )
    if run.diagnostics is not None:
        print(
            f"  {run.diagnostics.n_positions} position(s) · effective N "
            f"{run.diagnostics.effective_n:.1f} · diversification ratio "
            f"{run.diagnostics.diversification_ratio:.2f}"
        )
    _report_layer_exposures(run)
    performance = _report_versus_benchmark(run, config)

    sheets = run_sheets(
        run,
        riskfree_rate=config.optimizer.risk_free_rate,
        data_quality=quality,
        walk_forward=walk_forward,
        frontier_uncertainty=uncertainty,
        performance=performance,
    )
    out = write_excel_report(args.output, sheets)
    if walk_forward is not None:
        comparison = run.in_vs_out_of_sample(
            walk_forward, config.optimizer.risk_free_rate
        )
        gap = float(comparison.loc["Sharpe Ratio", "Degradation"])
        print(
            f"  Walk-forward: {walk_forward.n_resolves} re-solve(s) over "
            f"{walk_forward.n_trade_dates} trade date(s), "
            f"Sharpe falls {gap:.2f} out of sample"
        )
        _report_deflated_sharpe(walk_forward.returns, args, config)

    if args.mcos:
        from optimization_engine.resampling import monte_carlo_optimization_selection

        methods = tuple(
            m.strip() for m in str(args.mcos_methods).replace(" ", ",").split(",") if m.strip()
        )
        try:
            selection = monte_carlo_optimization_selection(
                returns, config, methods=methods, n_simulations=int(args.mcos)
            )
        except ValueError as exc:
            print(f"MCOS skipped: {exc}", file=sys.stderr)
        else:
            print(f"  {selection.describe()}")
            for name, row in selection.ranking().iterrows():
                print(
                    f"    {name:<18} weight RMSE {row['weight_rmse']:.2%} · "
                    f"worst position {row['max_weight_drift']:.2%}"
                )
    print(f"Wrote {out} ({len(sheets)} sheets)")
    _capture(args, optimization_payload(run, output_path=str(out)))
    return 0


def _apply_benchmark_flags(
    config, args: argparse.Namespace, assets: list[str]
) -> None:
    """Let ``--benchmark`` and the two limits override the config's block.

    ``--benchmark`` accepts a kind or an asset name, because on the command
    line "compare this against SPY" is the common case and forcing the
    ``kind: single_asset`` YAML for it would be ceremony.

    Raises:
        BenchmarkError: When the argument names neither a known kind nor an
            asset in the universe — better than silently comparing against
            something the caller did not ask for.
    """
    raw = getattr(args, "benchmark", None)
    if raw:
        value = str(raw).strip()
        if value.lower() in ("none", "off"):
            config.benchmark = BenchmarkSpec(kind="none")
            config.benchmark_weights = None
        elif value.lower() in ("equal_weight", "equal-weight", "ew", "1/n"):
            config.benchmark = BenchmarkSpec(kind="equal_weight")
        elif value in assets:
            config.benchmark = BenchmarkSpec(kind="single_asset", asset=value)
        else:
            raise BenchmarkError(
                f"--benchmark {value!r} is neither a benchmark kind nor an "
                f"asset in the universe ({', '.join(assets[:8])}"
                f"{' …' if len(assets) > 8 else ''}). Use 'equal_weight', an "
                "asset name, or define the benchmark in the config."
            )
    if getattr(args, "max_tracking_error", None) is not None:
        config.max_tracking_error = float(args.max_tracking_error)
    if getattr(args, "max_active_share", None) is not None:
        config.max_active_share = float(args.max_active_share)


def _report_layer_exposures(run) -> None:
    """Print each layer's bucket exposures next to their caps.

    Only the bucket that is *binding* explains the allocation, so the marker
    goes there: an allocator scanning the output should be able to see which
    line of the policy produced the portfolio without opening the workbook.
    """
    exposures = run.layer_exposures()
    if exposures.empty:
        return
    for layer_name, block in exposures.groupby("layer", sort=False):
        print(f"  {layer_name}:")
        for _, row in block.iterrows():
            limits = ""
            if pd.notna(row["effective_max"]):
                floor = (
                    f"{row['effective_min']:.1%}"
                    if pd.notna(row["effective_min"]) and row["effective_min"] > 0
                    else "0%"
                )
                limits = f" / limit {floor}–{row['effective_max']:.1%}"
                if row["basis"] == "parent":
                    limits += (
                        f" ({row['min']:.0%}–{row['max']:.0%} of {row['parent']})"
                    )
            mark = "  ←binding" if row["binding"] else ""
            print(f"    {row['bucket']:<24} {row['weight']:>7.2%}{limits}{mark}")


def _report_versus_benchmark(run, config):
    """Print the relative headline and return the report for the workbook.

    Returns ``None`` when the run has no benchmark, which is also what tells
    :func:`run_sheets` there is nothing relative to write.
    """
    if run.benchmark is None:
        return None
    try:
        report = run.performance(riskfree_rate=config.optimizer.risk_free_rate)
    except ValueError as exc:
        print(f"  Relative performance skipped: {exc}", file=sys.stderr)
        return None
    h = report.headline()
    print(
        f"  vs {run.benchmark_label}: excess {h['excess_return']:+.2%} · "
        f"T.E. {h['tracking_error']:.2%} · IR {h['information_ratio']:.2f} · "
        f"beta {h['beta']:.2f}"
        + (
            f" · active share {h['active_share']:.1%}"
            if "active_share" in h
            else ""
        )
    )
    return report


def _report_deflated_sharpe(returns, args: argparse.Namespace, config) -> None:
    """Print the walk-forward Sharpe deflated for the number of trials.

    An out-of-sample Sharpe is still a selected number when the configuration
    that produced it was itself chosen by looking at results. ``--trials``
    is how the analyst declares that search; the default of 1 is a claim, and
    the printed line says so.
    """
    from optimization_engine.analytics.selection import (
        deflated_sharpe_ratio,
        minimum_track_record_length,
    )

    try:
        deflated = deflated_sharpe_ratio(
            returns,
            n_trials=max(int(args.trials), 1),
            riskfree_rate=config.optimizer.risk_free_rate,
            periods_per_year=config.periods_per_year,
        )
    except ValueError as exc:
        print(f"  Deflated Sharpe skipped: {exc}", file=sys.stderr)
        return
    print(f"  {deflated.describe()}")
    try:
        needed = minimum_track_record_length(
            returns,
            benchmark_sharpe=deflated.benchmark_sharpe,
            riskfree_rate=config.optimizer.risk_free_rate,
            periods_per_year=config.periods_per_year,
        )
    except ValueError:
        return
    if needed == float("inf"):
        print(
            "  Minimum track record: unreachable — this Sharpe does not "
            "exceed the selection-bias threshold at any sample length."
        )
    else:
        print(
            f"  Minimum track record to call it significant at 95%: "
            f"{needed / config.periods_per_year:.1f} year(s) "
            f"({needed:.0f} periods); this run has "
            f"{len(returns)}."
        )


def _cmd_list_optimizers() -> int:
    width = max(len(n) for n in available_optimizers())
    for name in available_optimizers():
        print(f"{name:<{width}}  {requirements_for(name).summary}")
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    try:
        req = requirements_for(args.name)
    except KeyError as exc:
        print(str(exc).strip("\""), file=sys.stderr)
        return 2
    print(f"{req.display_name}  ({req.name})")
    print(f"\n  {req.summary}")
    print(f"\nUse it when:\n  {req.when_to_use}")
    if req.assumptions:
        print("\nAssumptions:")
        for a in req.assumptions:
            print(f"  - {a}")
    needs = [
        label
        for flag, label in (
            (req.requires_mu, "expected returns"),
            (req.requires_cov, "a covariance matrix"),
            (req.requires_returns, "the full return history"),
        )
        if flag
    ]
    print(f"\nInputs: {', '.join(needs) if needs else 'none beyond the universe'}")
    supports = [
        label
        for flag, label in (
            (req.supports_target_return, "target return"),
            (req.supports_target_volatility, "target volatility"),
            (req.supports_risk_aversion, "risk-aversion utility"),
            (req.supports_group_bounds, "group bounds"),
            (req.supports_turnover, "turnover budget"),
            (req.supports_frontier, "efficient frontier"),
        )
        if flag
    ]
    print(f"Supports: {', '.join(supports) if supports else 'no optional settings'}")
    print(f"Bounds: {req.bounds_note}")
    _capture(args, describe_payload(req))
    return 0


def _apply_estimator_flags(config, args: argparse.Namespace) -> None:
    """Let command-line estimator flags override the config file.

    Kept in one place so ``check`` and ``optimize`` cannot diverge: a
    pre-flight that reports the conditioning of a matrix the solve will not
    use is worse than no pre-flight at all.
    """
    if getattr(args, "denoise", False):
        config.denoise = True
    detone = getattr(args, "detone", None)
    if detone is not None:
        config.detone = int(detone)




#: Field presets exposed on the command line. ``close`` is the default because
#: it is all the optimizer needs; ``ohlcv`` is what a capacity-aware backtest
#: needs, and asking for it from a provider that has no volume fails loudly at
#: preflight instead of quietly returning a short panel.
_FIELD_PRESETS = {
    "close": ingest_fields.PRICE_ONLY,
    "ohlc": ingest_fields.OHLC,
    "ohlcv": ingest_fields.OHLCV,
}


def _add_ingest_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the multi-provider ingest flags to a subcommand.

    Added to every command that needs a price panel, so switching a run from a
    spreadsheet to a live provider is one flag rather than a different
    workflow.
    """
    group = parser.add_argument_group("data ingest (multi-provider)")
    group.add_argument(
        "--provider",
        help=(
            "Data provider to fetch from: "
            f"{', '.join(_provider_names())}. Requires --identifiers."
        ),
    )
    group.add_argument(
        "--identifiers",
        help="Comma- or space-separated universe, e.g. 'SPY,AGG,GLD' or 'SP500,IPC'.",
    )
    group.add_argument("--ingest-start", help="Inclusive start date (YYYY-MM-DD).")
    group.add_argument("--ingest-end", help="Inclusive end date (YYYY-MM-DD).")
    group.add_argument(
        "--ingest-period", default="5y",
        help="Window when no start date is given (1y, 2y, 3y, 5y, 10y, 20y).",
    )
    group.add_argument(
        "--ingest-interval", default="1d", choices=["1d", "1wk", "1mo"],
        help="Bar size.",
    )
    group.add_argument(
        "--ingest-fields", default="close", choices=sorted(_FIELD_PRESETS),
        help=(
            "Which fields to fetch. 'close' is enough to optimize and "
            "backtest; 'ohlcv' adds the volume a capacity-aware cost model "
            "needs, where the provider publishes it."
        ),
    )
    group.add_argument(
        "--ingest-currency",
        help="Convert every series into this ISO currency (e.g. USD, MXN).",
    )
    group.add_argument(
        "--require-volume", action="store_true",
        help=(
            "Fail when an instrument that should report volume does not. Off "
            "by default: indices have no volume and are backtested from a "
            "fixed participation rate instead."
        ),
    )
    group.add_argument(
        "--cache-dir",
        help="Directory for the on-disk panel cache. Omit to disable caching.",
    )
    group.add_argument(
        "--file-path",
        help="Path read by the 'file' provider (CSV, Excel or Parquet).",
    )
    group.add_argument(
        "--env-file", help="Load API keys from this .env file before fetching."
    )


def _provider_names() -> tuple[str, ...]:
    from optimization_engine.ingest import available_providers

    return available_providers()


def _ingest_request_from(args: argparse.Namespace) -> IngestRequest:
    """Build an :class:`IngestRequest` from the shared ingest flags."""
    return IngestRequest(
        identifiers=getattr(args, "identifiers", "") or "",
        provider=args.provider,
        start=getattr(args, "ingest_start", None),
        end=getattr(args, "ingest_end", None),
        period=None if getattr(args, "ingest_start", None) else args.ingest_period,
        interval=args.ingest_interval,
        fields=_FIELD_PRESETS[args.ingest_fields],
        currency=getattr(args, "ingest_currency", None),
        require_volume=bool(getattr(args, "require_volume", False)),
        cache_dir=getattr(args, "cache_dir", None),
    )


def _run_ingest(args: argparse.Namespace):
    """Fetch a panel and print the per-identifier outcome.

    Returns the :class:`~optimization_engine.ingest.IngestResult` so callers
    can take both the prices and the volume, which is what makes a
    capacity-aware backtest possible from the command line.
    """
    if getattr(args, "env_file", None):
        loaded = load_dotenv(args.env_file)
        print(f"Loaded {loaded} variable(s) from {args.env_file}")

    options = {}
    if getattr(args, "file_path", None):
        options["path"] = args.file_path

    result = ingest(_ingest_request_from(args), **options)
    print(f"Ingest: {result.summary()}")
    for outcome in result.failed:
        print(f"  ! {outcome.identifier}: {outcome.status} — {outcome.message}")
    for note in result.warnings:
        print(f"  · {note}")
    return result


def _load_prices_for(args: argparse.Namespace):
    """Resolve the price panel from whichever legacy flag names it.

    ``--provider`` is handled by :func:`_prepare_inputs`, which needs the
    whole ingest result rather than the prices alone.
    """
    if getattr(args, "yahoo", None):
        if args.yahoo_start:
            return load_prices_yahoo(
                args.yahoo, start=args.yahoo_start, end=args.yahoo_end
            )
        return load_prices_yahoo(args.yahoo, period=args.yahoo_period)
    if args.sample or not args.prices:
        return sample_dataset()
    return load_prices(args.prices, sheet_name=args.sheet)


@dataclass
class _Inputs:
    """What every solving command starts from, built one way."""

    config: object
    prices: pd.DataFrame
    returns: pd.DataFrame
    quality: object
    volumes: pd.DataFrame | None


def _fail(args: argparse.Namespace, message: str, code: int = 2) -> int:
    """Report a failure on stderr — and into the JSON payload, when there is one."""
    print(message, file=sys.stderr)
    sink = getattr(args, "_json_sink", None)
    if sink is not None:
        sink.setdefault("error", message)
    return code


def _prepare_inputs(args: argparse.Namespace) -> _Inputs | int:
    """Load the config and the panel, and shape both the way the solve needs.

    ``check``, ``optimize`` and ``backtest`` all start here, which is what
    makes a pre-flight worth running: it validates the mandate the solve
    goes on to see — same panel, same currency, same universe, same
    benchmark flags — rather than a mandate assembled slightly differently.
    Each used to build its inputs by hand, and they drifted: ``optimize``
    refused a config with no ``expected_returns`` block that ``check`` had
    just called ready, ``check`` never saw ``--base-currency`` or
    ``--benchmark``, and ``backtest`` seeded zero expected returns that
    ``resolve_expected_returns`` would otherwise have estimated.

    The order of operations: estimator flags, then the panel, then currency
    conversion, then the universe, then the benchmark flags. Currency before
    universe because a conversion needs every column it is told about;
    universe before benchmark because a single-asset benchmark has to name
    an asset that survived.

    Returns:
        The inputs, or an exit code when something the caller can fix is
        wrong — printed on stderr, and carried into the ``--json`` payload.
    """
    from optimization_engine.data.quality import analyze_prices

    config = load_config(args.config)
    _apply_estimator_flags(config, args)

    volumes = None
    ingested_currency = None
    try:
        if getattr(args, "provider", None):
            # One request serves both prices and volume, so a capacity-aware
            # backtest cannot price impact off a different panel than the
            # one it traded.
            ingested = _run_ingest(args)
            prices, volumes = ingested.prices, ingested.volumes
            ingested_currency = getattr(args, "ingest_currency", None)
        else:
            prices = _load_prices_for(args)
    except YahooFinanceError as exc:
        return _fail(args, f"Yahoo Finance error: {exc}")
    except IngestError as exc:
        return _fail(args, f"Ingest error: {exc}")

    if getattr(args, "base_currency", None):
        config.base_currency = args.base_currency.upper()
    if ingested_currency:
        # The ingest step converted from the provider's own currency
        # metadata. Applying ``config.currencies`` on top would convert the
        # panel twice, so the config's map is set aside and the base is the
        # one the panel was actually converted into.
        if config.currencies:
            print(
                f"  Currency: the panel was converted to {ingested_currency.upper()} "
                "on ingest, so the config's currencies map is not applied again.",
                file=sys.stderr,
            )
        config.base_currency = ingested_currency.upper()
    elif config.currencies:
        try:
            prices = apply_fx_conversion(prices, config)
        except FXError as exc:
            return _fail(args, f"FX conversion failed: {exc}")

    if config.expected_returns:
        common = [c for c in prices.columns if c in config.expected_returns]
        if not common:
            return _fail(
                args, "Config has no expected returns matching the price columns."
            )
        prices = prices[common]

    quality = analyze_prices(prices, periods_per_year=config.periods_per_year)
    returns = prices_to_returns(prices).dropna(how="any")
    if returns.empty:
        return _fail(args, "No usable returns after alignment.")
    if config.expected_returns:
        config.expected_returns = {a: config.expected_returns[a] for a in returns.columns}

    try:
        _apply_benchmark_flags(config, args, list(returns.columns))
    except (BenchmarkError, ValueError) as exc:
        return _fail(args, f"Benchmark error: {exc}")

    return _Inputs(
        config=config, prices=prices, returns=returns, quality=quality, volumes=volumes
    )


def _parse_sweep_arguments(raw: list[str] | None):
    """``--sweep path=a,b`` into the grid the sweep runner expects.

    Values are parsed as JSON when they can be, so numbers and booleans reach
    the config as numbers and booleans rather than as strings a validator
    would later reject.
    """
    import json as _json

    if not raw:
        return None
    params: dict[str, list] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(
                f"--sweep expects PATH=V1,V2 but got {entry!r}. "
                "Example: --sweep optimizer.name=min_variance,risk_parity"
            )
        path, _, values = entry.partition("=")
        parsed = []
        for token in values.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                parsed.append(_json.loads(token))
            except ValueError:
                parsed.append(token)
        if not parsed:
            raise ValueError(f"--sweep {path!r} lists no values.")
        params[path.strip()] = parsed
    from optimization_engine.backtest import SweepSpec

    return SweepSpec(params=params)


def _cmd_backtest(args: argparse.Namespace) -> int:
    """Walk the process forward, price the trading, and count the trials.

    The three things this prints that a plain optimize run cannot: what the
    strategy earned out of sample, what the trading cost to get it, and how
    much of the remaining Sharpe survives being deflated for the size of the
    search that produced it.
    """
    from optimization_engine.backtest import (
        BacktestSpec,
        CostSpec,
        final_holdout_run,
        gate_returns,
        run_backtest,
    )
    from optimization_engine.backtest.spec import SpecValidationError

    inputs = _prepare_inputs(args)
    if isinstance(inputs, int):
        return inputs
    config, prices, returns, volumes = (
        inputs.config, inputs.prices, inputs.returns, inputs.volumes
    )

    try:
        spec = BacktestSpec(
            frequency=args.rebalance,
            costs=CostSpec(
                commission_bps=args.commission_bps,
                slippage_bps=args.slippage_bps,
                impact_coefficient=args.impact_eta,
                impact_participation=args.impact_participation,
                impact_participation_source=args.impact_participation_source,
                impact_adv_share=args.impact_adv_share,
                impact_adv_lookback=args.impact_adv_lookback,
            ),
            execution_lag=args.execution_lag,
            periods_per_year=config.periods_per_year,
            initial_capital=args.initial_capital,
            name=Path(args.config).stem,
        )
    except SpecValidationError as exc:
        print(f"{exc} Pass --initial-capital.", file=sys.stderr)
        return 2
    print(spec.describe())
    if spec.costs.uses_volume and volumes is None:
        print(
            "  Liquidity: no volume panel available, so ADV-based impact "
            "falls back to the fixed participation rate of "
            f"{spec.costs.impact_participation:.1%}. Every affected trade is "
            "listed in the run's degradation notes."
        )

    evaluation = returns
    if args.holdout:
        evaluation = gate_returns(returns, args.holdout)
        print(
            f"  Holdout: walk-forward sees {len(evaluation)} of {len(returns)} "
            f"observations; everything after {args.holdout} is withheld."
        )

    try:
        run = run_engine(evaluation, config, check_feasibility=False)
    except SolverFailure as exc:
        return _fail(args, f"The initial solve failed: {exc}")
    try:
        walk = run.walk_forward_run(
            lookback=args.lookback,
            rebalance_every=args.rebalance_every,
            spec=spec,
            expanding=args.expanding,
            prices=prices,
            volumes=volumes,
        )
    except ValueError as exc:
        print(f"Walk-forward failed: {exc}", file=sys.stderr)
        return 2

    print(f"  {walk.run.describe()}")
    print(f"  {walk.describe()}")
    if walk.n_failures:
        print(f"  {walk.n_failures} solve(s) failed; the previous book was carried forward.")

    sweep_results = None
    overfitting = None
    n_trials = 1
    trial_sharpes = None
    try:
        sweep_spec = _parse_sweep_arguments(args.sweep)
    except ValueError as exc:
        print(f"Sweep skipped: {exc}", file=sys.stderr)
        sweep_spec = None
    if sweep_spec is not None:
        sweep_results = run.sweep(
            sweep_spec,
            lookback=args.lookback,
            rebalance_every=args.rebalance_every,
            spec=spec,
            expanding=args.expanding,
            prices=prices,
            volumes=volumes,
        )
        print(f"  {sweep_results.describe()}")
        n_trials = max(sweep_results.n_ok, 1)
        trial_sharpes = sweep_results.trial_sharpes()
        try:
            overfitting = sweep_results.overfitting_report()
        except ValueError as exc:
            print(f"  Overfitting analysis skipped: {exc}", file=sys.stderr)
        else:
            print(f"  {overfitting.describe()}")

    sheet = run.tearsheet(
        walk.run,
        n_trials=n_trials if sweep_results is not None else None,
        trial_sharpes=trial_sharpes,
        overfitting=overfitting,
    )
    print(f"  {sheet.tca.describe()}")
    if sheet.deflated_sharpe is not None:
        print(f"  {sheet.deflated_sharpe.describe()}")
    for caveat in sheet.caveats:
        print(f"  ! {caveat}")

    if args.holdout:
        # The last book the gated walk-forward chose is what a desk would
        # actually have been holding when the boundary arrived. Replaying it
        # forward is the one honest question the held-out segment can answer.
        locked = walk.weights_history.iloc[-1]
        holdout_spec = spec.with_(is_out_of_sample=True, name=f"{spec.name}-holdout")
        outcome = final_holdout_run(
            returns,
            args.holdout,
            # The held-out replay is a real run and must be priced the same
            # way: without the volume panel it would silently fall back to the
            # fixed participation rate, so the one segment that is supposed to
            # be the honest answer would be the cheapest.
            lambda segment: run_backtest(
                segment,
                locked,
                holdout_spec,
                prices=prices.reindex(segment.index) if volumes is not None else None,
                volumes=(
                    volumes.reindex(index=segment.index, columns=segment.columns)
                    if volumes is not None
                    else None
                ),
            ).returns,
            strategy={"config": args.config, "spec_hash": spec.spec_hash},
            audit_path=args.audit_log,
            periods_per_year=config.periods_per_year,
        )
        print(f"  {outcome.describe()}")
        print(
            "  Held-out Sharpe: "
            f"{float(outcome.summary.loc['holdout', 'Sharpe Ratio']):.2f}"
        )

    out = None
    if args.output:
        frames = sheet.to_frames()
        if sweep_results is not None:
            frames["sweep"] = sweep_results.frame
        out = write_excel_report(args.output, frames)
        print(f"Wrote {out} ({len(frames)} sheets)")
    _capture(
        args,
        backtest_payload(
            walk.run,
            tearsheet=sheet,
            output_path=str(out) if out is not None else None,
        ),
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Pre-flight the inputs and constraints, and say what would go wrong."""
    from optimization_engine.data.covariance import (
        covariance_diagnostics,
        covariance_from_config,
    )
    from optimization_engine.optimizers.factory import (
        constraints_from_config,
        effective_expected_returns,
    )
    from optimization_engine.optimizers.feasibility import analyze_feasibility

    inputs = _prepare_inputs(args)
    if isinstance(inputs, int):
        return inputs
    config, prices, returns, quality = (
        inputs.config, inputs.prices, inputs.returns, inputs.quality
    )

    print("== Data quality ==")
    print(quality.describe())
    print(
        f"\nCommon history: {quality.n_common_periods} period(s) "
        f"across {prices.shape[1]} asset(s)."
    )

    print("\n== Estimation ==")
    cov = covariance_from_config(returns, config)
    diag = covariance_diagnostics(
        cov, len(returns), config.covariance_method, config.ewma_lambda
    )
    print(
        f"T/N = {diag.observations_per_asset:.1f} · "
        f"condition number = {diag.condition_number:.3g}"
    )
    denoise_report = cov.attrs.get("denoise_report")
    if denoise_report is not None:
        print(f"  {denoise_report.describe()}")
    for w in diag.warnings:
        print(f"  ! {w}")
    if not diag.warnings:
        print("  Covariance estimate looks well conditioned.")

    print("\n== Constraints ==")
    # The same vector the solve will use. Deriving it differently here —
    # zeros, when the config carries no expected_returns block — would have
    # this command validate a mandate the optimizer never sees.
    mu = resolve_expected_returns(config, returns, cov)
    report = analyze_feasibility(
        list(returns.columns),
        constraints_from_config(config, list(returns.columns)),
        expected_returns=effective_expected_returns(config, cov, mu),
        cov_matrix=cov,
    )
    print(report.describe())
    if report.min_return is not None:
        line = (
            f"Reachable expected return: {report.min_return:.2%} to "
            f"{report.max_return:.2%}"
        )
        if report.min_variance_return is not None:
            line += f" (efficient above {report.min_variance_return:.2%})"
        print(line)

    _capture(args, check_payload(quality, report, diag))
    if quality.errors or not report.is_feasible:
        print("\nNot ready to optimize.", file=sys.stderr)
        return 1
    print("\nReady to optimize.")
    return 0


def _cmd_providers(args: argparse.Namespace) -> int:
    """Show every provider, what it serves, and whether it is usable now.

    The single most common failure in a multi-provider setup is asking a
    provider for something it does not publish, or forgetting a key. Both are
    visible here in one screen, before any run.
    """
    if getattr(args, "env_file", None):
        load_dotenv(args.env_file)

    rows = describe_providers()
    if args.json:
        import json

        print(json.dumps(list(rows), indent=2, default=str))
        return 0

    print(f"{len(rows)} data providers\n")
    for row in rows:
        mark = "ready" if row["ready"] else "needs key"
        print(f"  {row['provider']:<8} [{mark}]  {row['description']}")
        print(
            f"    fields:    {', '.join(f.removeprefix('m_') for f in row['fields'])}"
        )
        print(f"    intervals: {', '.join(row['intervals'])}")
        print(
            f"    volume:    {'yes' if row['serves_volume'] else 'no — index-style levels only'}"
        )
        print(f"    key:       {row['key_label']}")
        if row["signup_url"]:
            print(f"    sign up:   {row['signup_url']}")
        if row["notes"]:
            print(f"    note:      {row['notes']}")
        print()
    print(
        "Set a key with the provider's environment variable, or put it in a "
        ".env file and pass --env-file."
    )
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Fetch a panel and write it out, prices and volume separately."""
    if not args.provider:
        print("--provider is required for `ingest`.", file=sys.stderr)
        return 2
    if not args.identifiers:
        print("--identifiers is required for `ingest`.", file=sys.stderr)
        return 2

    try:
        result = _run_ingest(args)
    except IngestError as exc:
        print(f"Ingest error: {exc}", file=sys.stderr)
        return 2

    written = _write_panel(Path(args.output), result.prices)
    if written is None:
        return 2
    print(f"Wrote {args.output} ({result.prices.shape[0]} rows × "
          f"{result.prices.shape[1]} series)")

    if args.volume_output:
        volumes = result.volumes
        if volumes is None:
            print(
                "No volume to write: this universe carries none. The backtest "
                "will price impact from a fixed participation rate."
            )
        elif _write_panel(Path(args.volume_output), volumes) is not None:
            print(f"Wrote {args.volume_output} ({volumes.shape[1]} series)")

    print()
    print(result.panel.coverage().to_string())
    return 0 if result.is_complete else 1


def _write_panel(path: Path, frame: pd.DataFrame) -> Path | None:
    """Write a frame in the format its extension names.

    Parquet is the one format that needs a package the project does not depend
    on, so a missing engine is reported as the install it needs rather than as
    pandas' own two-paragraph ImportError.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame.to_excel(path, sheet_name="Precios")
    elif suffix == ".parquet":
        try:
            frame.to_parquet(path)
        except ImportError:
            print(
                "Writing Parquet needs pyarrow, which is not installed. "
                'Install it with: pip install -e ".[data]" — or write .csv '
                "or .xlsx instead.",
                file=sys.stderr,
            )
            return None
    else:
        print(f"Unsupported output extension: {suffix}", file=sys.stderr)
        return None
    return path


def _cmd_sample_data(args: argparse.Namespace) -> int:
    prices = sample_dataset(n_periods=args.periods)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        prices.to_csv(out)
    elif out.suffix.lower() in {".xlsx", ".xls"}:
        prices.to_excel(out, sheet_name="Precios")
    elif out.suffix.lower() == ".parquet":
        prices.to_parquet(out)
    else:
        print(f"Unsupported output extension: {out.suffix}", file=sys.stderr)
        return 2
    print(f"Wrote {out} ({prices.shape[0]} rows × {prices.shape[1]} cols)")
    return 0


def _cmd_fred(args: argparse.Namespace) -> int:
    try:
        df = load_fred_series(args.series, start=args.start, end=args.end)
    except FREDError as exc:
        print(f"FRED error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df.to_csv(out)
    elif out.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(out)
    elif out.suffix.lower() == ".parquet":
        df.to_parquet(out)
    else:
        print(f"Unsupported output extension: {out.suffix}", file=sys.stderr)
        return 2
    print(f"Wrote {out} ({df.shape[0]} rows × {df.shape[1]} series)")
    return 0


def _emit_json(
    args: argparse.Namespace, command: Callable[[argparse.Namespace], int]
) -> int:
    """Run a command in JSON mode: payload on stdout, narration on stderr.

    The commands below print as they work — data-quality findings, solver
    fallbacks, constraint warnings — and that narration is useful even to a
    machine caller, but not on the stream it is parsing. So stdout is
    redirected to stderr for the duration and the payload is written to the
    real stdout afterwards, which keeps every existing ``print`` where it is
    instead of threading a formatter through several hundred lines.

    A command that fails before producing a payload still emits one. A
    caller parsing stdout should never have to distinguish "no JSON" from
    "JSON I could not read"; an error object with the exit code is
    unambiguous, and the human-readable reason is on stderr.

    That promise covers a raised exception as much as a non-zero return.
    An unreadable config or an unwritable output directory raises rather
    than returning, and letting it propagate would print a traceback and
    leave stdout empty — precisely the case this mode exists to remove.
    The traceback still goes to stderr, where the human debugging it looks;
    the caller parsing stdout gets the exception's type and message.
    """
    import json

    sink: dict[str, object] = {}
    args._json_sink = sink
    failure: str | None = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            code = command(args)
    except Exception as exc:  # noqa: BLE001 — the payload *is* the contract
        traceback.print_exc()
        failure = f"{type(exc).__name__}: {exc}"
        code = 1
    payload = sink.get("payload")
    if failure is not None or payload is None:
        # A run that raised reports the failure even if it had already
        # captured a payload: emitting that payload under a non-zero exit
        # code would describe a result the command did not finish producing.
        # A run that *returned* a code carries the reason it printed.
        payload = {
            "schema_version": SCHEMA_VERSION,
            "command": args.command,
            "error": (
                failure
                or sink.get("error")
                or "the command exited before producing a result"
            ),
            "exit_code": code,
        }
    print(json.dumps(payload, indent=2, default=str))
    return code


def _capture(args: argparse.Namespace, payload: dict[str, object]) -> None:
    """Hand a payload to :func:`_emit_json`, if this run is in JSON mode."""
    sink = getattr(args, "_json_sink", None)
    if sink is not None:
        sink["payload"] = payload


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and dispatch to the requested subcommand.

    Args:
        argv: Arguments to parse, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code. ``0`` on success; ``1`` when the command ran and
        the answer is negative — an infeasible mandate from ``check``, an
        incomplete panel from ``ingest``, or, under ``--json``, a command that
        raised; ``2`` when the command could not run at all. See
        ``docs/ERRORS.md`` for the full contract.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        if args.json:
            return _emit_json(args, _cmd_optimize)
        return _cmd_optimize(args)
    if args.command == "list-optimizers":
        return _cmd_list_optimizers()
    if args.command == "sample-data":
        return _cmd_sample_data(args)
    if args.command == "fred":
        return _cmd_fred(args)
    if args.command == "check":
        if args.json:
            return _emit_json(args, _cmd_check)
        return _cmd_check(args)
    if args.command == "backtest":
        if args.json:
            return _emit_json(args, _cmd_backtest)
        return _cmd_backtest(args)
    if args.command == "describe":
        if args.json:
            return _emit_json(args, _cmd_describe)
        return _cmd_describe(args)
    if args.command == "providers":
        return _cmd_providers(args)
    if args.command == "ingest":
        return _cmd_ingest(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
