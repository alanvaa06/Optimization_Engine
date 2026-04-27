# Optimization Engine

A multi-asset portfolio optimization engine with a clean API, a Streamlit
UI, and a CLI. Built on top of `cvxpy`, `scipy`, `pandas`, and `plotly`.

## What's inside

**Optimization techniques**

| Method | Class | When to use |
| --- | --- | --- |
| Mean-variance (target return / vol / utility) | `MeanVarianceOptimizer` | Classic Markowitz with full constraints |
| Global minimum variance | `MinVarianceOptimizer` | Risk minimization without return targets |
| Maximum Sharpe ratio | `MaxSharpeOptimizer` | Tangency portfolio (excess-return / vol) |
| Risk parity / risk budgeting (ERC) | `RiskParityOptimizer` | Equal or specified risk contributions |
| Hierarchical Risk Parity (HRP) | `HRPOptimizer` | Many assets, ill-conditioned covariance |
| Black-Litterman | `BlackLittermanOptimizer` | Combine equilibrium + subjective views |
| Mean-CVaR (Rockafellar-Uryasev) | `CVaROptimizer` | Tail-risk-aware allocation |
| Maximum diversification | `MaxDiversificationOptimizer` | Maximize correlation benefit |
| Inverse volatility | `InverseVolatilityOptimizer` | Cheap baseline |
| Equal weight (1/N) | `EqualWeightOptimizer` | Hard-to-beat baseline |

**Constraints**

* Per-asset weight bounds.
* Group / asset-class bounds (e.g. equity 30–70%).
* Long-only or leveraged.
* Target return, target volatility, or risk-aversion utility.
* Turnover budget vs. previous weights (mean-variance / CVaR).

**Covariance estimators**

`sample`, `ledoit_wolf`, `oas`, `ewma` (RiskMetrics), `semi`,
`shrink` (riskfolio passthrough when installed).

**Analytics**

Drawdown, VaR/CVaR (historic & Cornish-Fisher), Sharpe, Sortino,
information ratio, beta, up/down capture, summary stats vs. benchmark.

## Layout

```
src/optimization_engine/
├── analytics/        # performance · risk · relative metrics
├── data/             # loaders + covariance estimators
├── optimizers/       # one file per technique
├── reporting/        # Excel exporter + Plotly figures
├── config.py         # YAML/JSON-driven config
├── engine.py         # high-level façade (run_engine)
├── frontier.py       # efficient frontier sweep
└── cli.py            # `optengine` entrypoint
app/streamlit_app.py  # interactive UI
config/               # example configs (multi-asset + legacy)
notebooks/            # quickstart notebook
scripts/              # batch runners
tests/                # pytest smoke tests
```

## Install

```bash
pip install -e .[ui,extras,dev]
```

The `extras` pull `riskfolio-lib`; `ui` pulls Streamlit and ipywidgets.
Without `extras` the engine falls back to scikit-learn's Ledoit-Wolf for
the `shrink` covariance method.

## Use it

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Tabs walk you through the workflow:

1. **Overview** — wealth, correlations.
2. **Assets** — per-asset stats and drawdowns.
3. **Expected returns & constraints** — fully editable tables for
   expected returns, bounds, groups, group bounds, plus optimizer-specific
   inputs (risk budgets / Black-Litterman views).
4. **Compare** — pick up to 5 saved scenarios; renders a stacked-weight
   chart, a per-asset grouped bar chart, a wealth-overlay backtest, and a
   summary table with annualized return / vol / Sharpe / active positions.
5. **What-if** — anchor on a saved scenario and drag per-asset weight-bound
   range sliders; the optimizer re-solves live (or behind a Recompute button
   for `cvar` or universes >25 assets) and KPIs update immediately. "Save
   these as new scenario" forks the result.
6. **Optimize** — KPI cards, weights, risk contributions, frontier, backtest.
7. **Report** — one-click Excel export and the YAML config the run used.

The sidebar's **📚 Scenarios** block lets you save, update, load, rename,
delete, and download/upload named profiles (YAML) — so you can switch
between a "Conservative", "Black-Litterman bull case", and "Risk parity"
view without retyping the constraints.

### CLI

```bash
optengine list-optimizers
optengine sample-data --output data/sample/sample_prices.csv
optengine optimize --config config/example_multi_asset.yaml --sample --frontier
```

### Python

```python
from optimization_engine import (
    EngineConfig, OptimizerSpec, run_engine,
    sample_dataset, prices_to_returns,
)

prices = sample_dataset()
returns = prices_to_returns(prices)

config = EngineConfig(
    expected_returns={c: 0.05 for c in returns.columns},
    bounds={c: [0.0, 0.4] for c in returns.columns},
    optimizer=OptimizerSpec(name="risk_parity"),
)

run = run_engine(returns, config, build_frontier=True)
print(run.result.weights.round(3))
print(run.absolute_summary())
```

## Migration from the original notebook

The original notebook used a single `optimize_portfolio` function with
hard-coded constraints and a target-return sweep. Equivalents:

| Old | New |
| --- | --- |
| `optimize_portfolio(0.057, ...)` | `MeanVarianceOptimizer(...).optimize()` with `target_return=0.057` |
| `efficient_frontier(start, end, steps)` | `efficient_frontier(config, cov, returns, n_points=...)` or `run_engine(..., build_frontier=True)` |
| `summary_stats`, `summary_relative` | unchanged names in `optimization_engine.analytics` |
| `risk_contribution(w, Σ)` | `optimization_engine.analytics.risk_contribution` |
| Excel export at the end | `optimization_engine.reporting.write_excel_report` |

A drop-in YAML for the original universe lives at
`config/legacy_optluis.yaml` — drop `Precios_OptLuis_USD.xlsx` into
`data/` and run:

```bash
optengine optimize --config config/legacy_optluis.yaml \
                   --prices data/Precios_OptLuis_USD.xlsx \
                   --frontier
```

## Tests

```bash
pytest -q
```

Covers covariance estimators, every optimizer, frontier monotonicity,
ERC properties, and Black-Litterman blending.

## License

MIT — see [LICENSE](LICENSE).
