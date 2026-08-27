"""Command-line entrypoint: ``optengine``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from optimization_engine.benchmark import BenchmarkError, BenchmarkSpec
from optimization_engine.config import load_config
from optimization_engine.data.fred import FREDError, load_fred_series
from optimization_engine.data.fx import FXError
from optimization_engine.data.loader import load_prices, prices_to_returns, sample_dataset
from optimization_engine.data.yahoo import YahooFinanceError, load_prices_yahoo
from optimization_engine.engine import apply_fx_conversion, run_engine
from optimization_engine.optimizers._cvxpy_helpers import SolverFailure
from optimization_engine.optimizers.factory import available_optimizers
from optimization_engine.optimizers.requirements import requirements_for
from optimization_engine.reporting.exporters import run_sheets, write_excel_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optengine",
        description="Multi-asset portfolio optimization engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    optimize = sub.add_parser("optimize", help="Run an optimization end-to-end.")
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
        help="Periods between re-solves. Defaults to one quarter.",
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

    check = sub.add_parser(
        "check",
        help="Validate data and constraints without solving. Reports data "
             "quality, covariance conditioning and constraint feasibility.",
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

    describe = sub.add_parser(
        "describe", help="Explain one optimizer: what it needs and what it assumes."
    )
    describe.add_argument("name", help="Optimizer name (see list-optimizers).")

    sub.add_parser("list-optimizers", help="List available optimizer names.")

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
    from optimization_engine.data.quality import analyze_prices
    from optimization_engine.optimizers.feasibility import InfeasibleConstraintsError

    config = load_config(args.config)
    try:
        prices = _load_prices_for(args, config)
    except YahooFinanceError as exc:
        print(f"Yahoo Finance error: {exc}", file=sys.stderr)
        return 2

    _apply_estimator_flags(config, args)
    if args.base_currency:
        config.base_currency = args.base_currency.upper()
    if config.currencies:
        try:
            prices = apply_fx_conversion(prices, config)
        except FXError as exc:
            print(f"FX conversion failed: {exc}", file=sys.stderr)
            return 2

    common = set(prices.columns) & set(config.expected_returns.keys())
    if not common:
        print("Config has no expected returns matching the price columns.", file=sys.stderr)
        return 2
    prices = prices[sorted(common)]

    quality = analyze_prices(prices, periods_per_year=config.periods_per_year)
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

    returns = prices_to_returns(prices).dropna(how="any")
    if returns.empty:
        print("No usable returns after alignment.", file=sys.stderr)
        return 2
    config.expected_returns = {
        a: config.expected_returns[a] for a in returns.columns
    }

    try:
        _apply_benchmark_flags(config, args, list(returns.columns))
    except (BenchmarkError, ValueError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
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
            f"  Walk-forward: {walk_forward.n_rebalances} re-solve(s), "
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


def _load_prices_for(args: argparse.Namespace, config):
    """Resolve the price panel from whichever source the flags name."""
    if getattr(args, "yahoo", None):
        if args.yahoo_start:
            return load_prices_yahoo(
                args.yahoo, start=args.yahoo_start, end=args.yahoo_end
            )
        return load_prices_yahoo(args.yahoo, period=args.yahoo_period)
    if args.sample or not args.prices:
        return sample_dataset()
    return load_prices(args.prices, sheet_name=args.sheet)


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

    config = load_config(args.config)
    _apply_estimator_flags(config, args)
    prices = _load_prices_for(args, config)
    returns = prices_to_returns(prices)
    if not config.expected_returns:
        config.expected_returns = {asset: 0.0 for asset in returns.columns}

    spec = BacktestSpec(
        costs=CostSpec(
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            impact_coefficient=args.impact_eta,
            impact_participation=args.impact_participation,
        ),
        execution_lag=args.execution_lag,
        periods_per_year=config.periods_per_year,
        name=Path(args.config).stem,
    )
    print(spec.describe())

    evaluation = returns
    if args.holdout:
        evaluation = gate_returns(returns, args.holdout)
        print(
            f"  Holdout: walk-forward sees {len(evaluation)} of {len(returns)} "
            f"observations; everything after {args.holdout} is withheld."
        )

    run = run_engine(evaluation, config, check_feasibility=False)
    try:
        walk = run.walk_forward_run(
            lookback=args.lookback,
            rebalance_every=args.rebalance_every,
            spec=spec,
            expanding=args.expanding,
        )
    except ValueError as exc:
        print(f"Walk-forward failed: {exc}", file=sys.stderr)
        return 2

    print(f"  {walk.run.describe()}")
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
            lambda segment: run_backtest(segment, locked, holdout_spec).returns,
            strategy={"config": args.config, "spec_hash": spec.spec_hash},
            audit_path=args.audit_log,
            periods_per_year=config.periods_per_year,
        )
        print(f"  {outcome.describe()}")
        print(
            "  Held-out Sharpe: "
            f"{float(outcome.summary.loc['holdout', 'Sharpe Ratio']):.2f}"
        )

    if args.output:
        frames = sheet.to_frames()
        if sweep_results is not None:
            frames["sweep"] = sweep_results.frame
        out = write_excel_report(args.output, frames)
        print(f"Wrote {out} ({len(frames)} sheets)")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Pre-flight the inputs and constraints, and say what would go wrong."""
    from optimization_engine.data.covariance import (
        covariance_diagnostics,
        covariance_from_config,
    )
    from optimization_engine.data.quality import analyze_prices
    from optimization_engine.optimizers.factory import (
        constraints_from_config,
        effective_expected_returns,
    )
    from optimization_engine.optimizers.feasibility import analyze_feasibility

    config = load_config(args.config)
    _apply_estimator_flags(config, args)
    prices = _load_prices_for(args, config)
    common = [c for c in prices.columns if c in config.expected_returns]
    if common:
        prices = prices[common]

    print("== Data quality ==")
    quality = analyze_prices(prices, periods_per_year=config.periods_per_year)
    print(quality.describe())
    print(
        f"\nCommon history: {quality.n_common_periods} period(s) "
        f"across {prices.shape[1]} asset(s)."
    )

    returns = prices_to_returns(prices).dropna(how="any")
    if returns.empty:
        print("\nNo usable returns after alignment.", file=sys.stderr)
        return 2

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
    mu = pd.Series(config.expected_returns).reindex(returns.columns).fillna(0.0)
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

    if quality.errors or not report.is_feasible:
        print("\nNot ready to optimize.", file=sys.stderr)
        return 1
    print("\nReady to optimize.")
    return 0


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        return _cmd_optimize(args)
    if args.command == "list-optimizers":
        return _cmd_list_optimizers()
    if args.command == "sample-data":
        return _cmd_sample_data(args)
    if args.command == "fred":
        return _cmd_fred(args)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "backtest":
        return _cmd_backtest(args)
    if args.command == "describe":
        return _cmd_describe(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
