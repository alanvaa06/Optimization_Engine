"""End-to-end script: run an optimization, write Excel + plots.

Usage::

    python scripts/run_optimization.py --config config/example_multi_asset.yaml --sample
    python scripts/run_optimization.py --config config/legacy_optluis.yaml \\
                                       --prices data/Precios_OptLuis_USD.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from optimization_engine.config import load_config  # noqa: E402
from optimization_engine.data.loader import (  # noqa: E402
    load_prices,
    prices_to_returns,
    sample_dataset,
)
from optimization_engine.engine import run_engine  # noqa: E402
from optimization_engine.reporting.exporters import write_excel_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prices", default=None)
    parser.add_argument("--sheet", default="Precios")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--output", default="outputs/optimization_report.xlsx")
    parser.add_argument("--frontier", action="store_true", default=True)
    parser.add_argument("--frontier-points", type=int, default=25)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.sample or args.prices is None:
        prices = sample_dataset()
    else:
        prices = load_prices(args.prices, sheet_name=args.sheet)

    returns = prices_to_returns(prices)
    common = sorted(set(returns.columns) & set(config.expected_returns))
    if not common:
        print("No overlap between prices and config.expected_returns", file=sys.stderr)
        return 2
    returns = returns[common]
    config.expected_returns = {a: config.expected_returns[a] for a in common}

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
        "absolute_summary": run.absolute_summary(
            riskfree_rate=config.optimizer.risk_free_rate
        ),
        "cov_matrix": run.cov_matrix,
    }
    if run.frontier is not None:
        sheets["frontier_summary"] = run.frontier.summary
        sheets["frontier_weights"] = run.frontier.weights
        if run.frontier.group_weights is not None and not run.frontier.group_weights.empty:
            sheets["frontier_groups"] = run.frontier.group_weights

    out = write_excel_report(args.output, sheets)
    print(f"Wrote {out}")
    print(f"Weights:\n{run.result.weights.round(4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
