"""Command-line entrypoint: ``optengine``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from optimization_engine.config import load_config
from optimization_engine.data.loader import load_prices, prices_to_returns, sample_dataset
from optimization_engine.engine import run_engine
from optimization_engine.optimizers.factory import available_optimizers
from optimization_engine.reporting.exporters import write_excel_report


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
    optimize.add_argument("--output", default="outputs.xlsx", help="Output Excel path.")
    optimize.add_argument("--frontier", action="store_true", help="Also compute the frontier.")
    optimize.add_argument("--frontier-points", type=int, default=25)

    sub.add_parser("list-optimizers", help="List available optimizer names.")

    sample = sub.add_parser("sample-data", help="Write a synthetic price panel to disk.")
    sample.add_argument("--output", default="data/sample/sample_prices.csv")
    sample.add_argument("--periods", type=int, default=252 * 8)

    return parser


def _cmd_optimize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.sample or not args.prices:
        prices = sample_dataset()
    else:
        prices = load_prices(args.prices, sheet_name=args.sheet)

    returns = prices_to_returns(prices)
    common = set(returns.columns) & set(config.expected_returns.keys())
    if not common:
        print("Config has no expected returns matching the price columns.", file=sys.stderr)
        return 2
    returns = returns[sorted(common)]
    config.expected_returns = {a: config.expected_returns[a] for a in returns.columns}

    run = run_engine(
        returns,
        config,
        build_frontier=args.frontier,
        n_frontier_points=args.frontier_points,
    )

    sheets: dict[str, pd.DataFrame] = {
        "weights": run.result.weights.to_frame("weight"),
        "summary": pd.DataFrame(
            [
                {
                    "expected_return": run.result.expected_return,
                    "expected_volatility": run.result.expected_volatility,
                    "sharpe_ratio": run.result.sharpe_ratio,
                }
            ]
        ),
        "risk_contributions": run.risk_contributions().to_frame("share_of_variance"),
        "expected_returns": run.expected_returns.to_frame("annualized"),
        "cov_matrix": run.cov_matrix,
    }
    if run.frontier is not None:
        sheets["frontier_summary"] = run.frontier.summary
        sheets["frontier_weights"] = run.frontier.weights
        if run.frontier.group_weights is not None and not run.frontier.group_weights.empty:
            sheets["frontier_groups"] = run.frontier.group_weights

    out = write_excel_report(args.output, sheets)
    print(f"Wrote {out}")
    return 0


def _cmd_list_optimizers() -> int:
    for name in available_optimizers():
        print(name)
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        return _cmd_optimize(args)
    if args.command == "list-optimizers":
        return _cmd_list_optimizers()
    if args.command == "sample-data":
        return _cmd_sample_data(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
