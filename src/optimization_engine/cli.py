"""Command-line entrypoint: ``optengine``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from optimization_engine.config import load_config
from optimization_engine.data.fred import FREDError, load_fred_series
from optimization_engine.data.fx import FXError
from optimization_engine.data.loader import load_prices, prices_to_returns, sample_dataset
from optimization_engine.data.yahoo import YahooFinanceError, load_prices_yahoo
from optimization_engine.engine import apply_fx_conversion, run_engine
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
        "--strict", action="store_true",
        help="Refuse to run when the constraints are infeasible, naming the "
             "constraint instead of letting the solver fail.",
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

    sheets = run_sheets(
        run,
        riskfree_rate=config.optimizer.risk_free_rate,
        data_quality=quality,
        walk_forward=walk_forward,
    )
    out = write_excel_report(args.output, sheets)

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
    if walk_forward is not None:
        comparison = run.in_vs_out_of_sample(
            walk_forward, config.optimizer.risk_free_rate
        )
        gap = float(comparison.loc["Sharpe Ratio", "Degradation"])
        print(
            f"  Walk-forward: {walk_forward.n_rebalances} re-solve(s), "
            f"Sharpe falls {gap:.2f} out of sample"
        )
    print(f"Wrote {out} ({len(sheets)} sheets)")
    return 0


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


def _cmd_check(args: argparse.Namespace) -> int:
    """Pre-flight the inputs and constraints, and say what would go wrong."""
    from optimization_engine.data.covariance import covariance_diagnostics, covariance_matrix
    from optimization_engine.data.quality import analyze_prices
    from optimization_engine.optimizers.factory import (
        constraints_from_config,
        effective_expected_returns,
    )
    from optimization_engine.optimizers.feasibility import analyze_feasibility

    config = load_config(args.config)
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
    cov = covariance_matrix(
        returns,
        method=config.covariance_method,
        periods_per_year=config.periods_per_year,
        ewma_lambda=config.ewma_lambda,
    )
    diag = covariance_diagnostics(
        cov, len(returns), config.covariance_method, config.ewma_lambda
    )
    print(
        f"T/N = {diag.observations_per_asset:.1f} · "
        f"condition number = {diag.condition_number:.3g}"
    )
    for w in diag.warnings:
        print(f"  ! {w}")
    if not diag.warnings:
        print("  Covariance estimate looks well conditioned.")

    print("\n== Constraints ==")
    mu = pd.Series(config.expected_returns).reindex(returns.columns).fillna(0.0)
    report = analyze_feasibility(
        list(returns.columns),
        constraints_from_config(config),
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
    if args.command == "describe":
        return _cmd_describe(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
