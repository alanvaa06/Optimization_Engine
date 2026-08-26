# Optimization Engine

A multi-asset portfolio optimization engine with a clean API, a Streamlit
UI, and a CLI. Built on top of `cvxpy`, `scipy`, `pandas`, and `plotly`.

The engine is opinionated about one thing: **an allocation is not a result
until you can see what it rests on.** Every solve returns the weights *and*
the evidence — which solver answered, whether the constraints were actually
respected, how well-conditioned the covariance estimate was, how concentrated
the book is in risk rather than capital, and how much of the backtest
survives out of sample.

## What's inside

**Optimization techniques**

| Method | Class | When to use |
| --- | --- | --- |
| Mean-variance (target return / vol / utility) | `MeanVarianceOptimizer` | Classic Markowitz with full constraints |
| Global minimum variance | `MinVarianceOptimizer` | You don't trust any expected-return estimate |
| Maximum Sharpe ratio | `MaxSharpeOptimizer` | Tangency portfolio, sized separately against cash |
| Risk parity / risk budgeting (ERC) | `RiskParityOptimizer` | Spread *risk* evenly, not capital |
| Hierarchical Risk Parity (HRP) | `HRPOptimizer` | Many assets, ill-conditioned covariance, small T/N |
| Black-Litterman | `BlackLittermanOptimizer` | A few specific views on top of equilibrium |
| Mean-CVaR (Rockafellar-Uryasev) | `CVaROptimizer` | Skewed or fat-tailed returns; variance is the wrong measure |
| Maximum diversification | `MaxDiversificationOptimizer` | Correlation benefit as the objective |
| Inverse volatility | `InverseVolatilityOptimizer` | Cheap risk-parity approximation |
| Equal weight (1/N) | `EqualWeightOptimizer` | The baseline everything else has to beat |

`optengine describe <name>` prints what a method needs, what it supports, and
what it assumes — the same text the UI shows next to the method picker.

**Constraints**

* Per-asset weight bounds.
* Group / asset-class bounds (e.g. equity 30–70%).
* Long-only or long-short with a gross-exposure cap.
* Target return, target volatility, or risk-aversion utility.
* Turnover budget against a previous allocation.

Every constraint is honoured by every method, but not always the same way.
Convex solves impose them directly; the allocate-then-constrain methods (1/N,
inverse volatility, HRP) project onto the *closest* feasible allocation and
report how far the mandate moved their answer — because a 1/N book that had
to shift 15% of its weight is no longer really 1/N. Where a constraint truly
cannot bind (a turnover budget on the homogeneous max-Sharpe,
max-diversification and risk-parity solves), the method warns rather than
ignoring it silently. After every solve the weights are checked against every
constraint and any breach is reported.

**Covariance estimators**

`sample`, `ledoit_wolf`, `oas`, `ewma` (RiskMetrics), `semi`,
`shrink` (riskfolio passthrough when installed).

Every estimate is symmetrized and PSD-repaired, and comes with diagnostics:
observations per asset, condition number, smallest eigenvalue, and effective
sample size — with a warning when the estimate is too thin or too
ill-conditioned to optimize against.

**Expected returns**

`historical_mean`, `ema`, `capm`, and `shrunk_mean` (Jorion Bayes-Stein
shrinkage toward the minimum-variance portfolio's return, which keeps the
cross-sectional ranking while pulling in the extremes that drive
mean-variance to corner solutions).

**Analytics**

Drawdown episodes with recovery timing, Ulcer index, VaR/CVaR (historic and
Cornish-Fisher), Sharpe, probabilistic Sharpe, Sortino, Calmar, Omega, tail
ratio, hit rate, rolling metrics, Newey-West annualized volatility, the full
Euler risk decomposition in volatility units, concentration measures
(Herfindahl, effective N, effective N of *risk*, diversification ratio), and
benchmark-relative statistics: tracking error, information ratio, CAPM alpha
with its t-statistic and R², up/down capture, conditional (up/down) beta, and
active share.

**Backtesting**

Two things a constant-weight in-sample replay hides, and this doesn't:

* `backtest_weights()` lets positions drift between rebalances, trades the
  book back on a chosen cadence, and charges transaction costs on traded
  notional — reporting turnover and the annualized cost drag.
* `walk_forward_backtest()` re-estimates and re-solves on a rolling (or
  expanding) window and holds each solution forward over returns the
  optimizer never saw. `compare_in_and_out_of_sample()` puts the two track
  records side by side with the degradation between them.

## Layout

```
src/optimization_engine/
├── analytics/        # performance · risk · relative · backtest
├── data/             # loaders · covariance · data-quality analysis
├── optimizers/       # one file per technique + diagnostics + feasibility
├── reporting/        # Excel exporter + Plotly figures
├── config.py         # YAML/JSON-driven config
├── engine.py         # high-level façade (run_engine)
├── frontier.py       # efficient frontier sweep
└── cli.py            # `optengine` entrypoint
app/
├── streamlit_app.py  # interactive UI
└── components.py     # reusable render blocks
config/               # example configs
notebooks/            # quickstart notebook
scripts/              # batch runners
tests/                # pytest suite
```

## Install

```bash
pip install -e ".[ui,extras,dev]"
```

The `extras` pull `riskfolio-lib`; `ui` pulls Streamlit and ipywidgets.
Without `extras` the engine falls back to scikit-learn's Ledoit-Wolf for
the `shrink` covariance method.

## Use it

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

The sidebar walks a numbered path — **Data → Currency → Method →
Assumptions → Objective → Exposure → Frontier** — and each step surfaces what
could make the next one wrong.

1. **Data** — data-quality report before anything is estimated: interior
   gaps, stale feeds that read to an optimizer as low volatility, unadjusted
   splits, and how many periods actually have every asset present. The
   missing-data policy is an explicit choice, and the app logs exactly what it
   did to the panel.
2. **Assets** — per-asset statistics (extended metrics on a toggle) plus a
   drawdown-episode table with peak, trough, recovery and time underwater.
3. **Assumptions & constraints** — editable expected returns, weight bounds,
   groups, group budgets, per-method inputs (risk budgets, Black-Litterman
   views both absolute and relative), and a **live feasibility check** that
   names the constraint making the problem impossible and what to change,
   before you ever press solve.
4. **Optimize** — a compliance banner, KPI cards, concentration and
   diversification measures, a capital-vs-risk chart, the full Euler
   decomposition, and a frontier marked with the minimum-variance and
   tangency portfolios, the capital allocation line, and where your portfolio
   actually sits.
5. **Backtest** — choose a rebalancing cadence and transaction costs; see
   weight drift, cost drag, rolling performance, the return distribution with
   its VaR/CVaR cuts, benchmark-relative statistics, and a walk-forward run
   that says in words how much of the result was hindsight.
6. **Compare / What-if** — saved scenarios side by side, and live re-solving
   as you drag weight bounds.
7. **Report** — one-click Excel export carrying assumptions, diagnostics,
   data-quality findings and the walk-forward comparison alongside the
   weights.

The sidebar's **📚 Scenarios** block saves, updates, loads, renames, deletes,
and downloads/uploads named profiles (YAML).

### CLI

```bash
optengine list-optimizers                    # each method with a one-line summary
optengine describe hrp                       # what it needs, supports, assumes
optengine sample-data --output data/sample/sample_prices.csv

# Pre-flight the inputs and constraints without solving.
optengine check --config config/example_multi_asset.yaml --sample

# Solve, and refuse rather than fail obscurely if the mandate is impossible.
optengine optimize --config config/example_multi_asset.yaml --sample \
    --frontier --walk-forward --cost-bps 10 --strict
```

`check` exits non-zero when the data has errors or the constraints have no
solution, so it drops straight into CI or a pre-commit hook.

### Python

```python
from optimization_engine import (
    EngineConfig, OptimizerSpec, run_engine,
    sample_dataset, prices_to_returns, analyze_prices,
)

prices = sample_dataset()
print(analyze_prices(prices).describe())

returns = prices_to_returns(prices)

config = EngineConfig(
    expected_returns={c: 0.05 for c in returns.columns},
    bounds={c: [0.0, 0.4] for c in returns.columns},
    optimizer=OptimizerSpec(name="risk_parity"),
)

run = run_engine(returns, config, build_frontier=True)
print(run.result.weights.round(3))
print(run.result.violations)          # empty when fully compliant
print(run.diagnostics.effective_n)    # concentration, not just position count
print(run.risk_decomposition())       # contributions in volatility units

# What survives out of sample?
wf = run.walk_forward(transaction_cost_bps=10)
print(run.in_vs_out_of_sample(wf))
```

Relative (spread) Black-Litterman views:

```python
from optimization_engine import View

config.optimizer.name = "black_litterman"
config.optimizer.bl_views = [
    View({"US_Equity": 1.0, "EM_Equity": -1.0}, 0.03, label="US over EM"),
]
```

Note that Black-Litterman optimizes against its equilibrium *posterior*,
which normally sits well below historical means — a return target that suits
mean-variance is often unreachable here. `run_engine` checks the target
against the posterior and says so.

## Tests

```bash
pytest -q
ruff check src app tests scripts
```

The suite covers the covariance estimators and their PSD repair, every
optimizer, frontier monotonicity and reachability, ERC properties,
Black-Litterman blending with absolute and relative views, constraint
compliance, feasibility diagnosis, backtest drift/costs, walk-forward
look-ahead safety, data-quality detection, and the CLI's exit codes.

## License

MIT — see [LICENSE](LICENSE).
