# Optimization Engine

A multi-asset portfolio optimization engine with a clean API, a Streamlit
UI, and a CLI. Built on top of `cvxpy`, `scipy`, `pandas`, and `plotly`.

The engine is opinionated about one thing: **an allocation is not a result
until you can see what it rests on.** Every solve returns the weights *and*
the evidence — which solver answered, whether the constraints were actually
respected, how well-conditioned the covariance estimate was, how concentrated
the book is in risk rather than capital, and how much of the backtest
survives out of sample.

![The Optimize tab: compliance banner, concentration diagnostics, allocation and risk decomposition](docs/images/app-optimize.png)

## The four questions it answers

### 1. Where does this portfolio sit, and what else was available?

The frontier is drawn with the portfolios worth marking on it — the
minimum-variance anchor where the efficient branch starts, the tangency
portfolio, the capital allocation line through it, and where your own
allocation actually landed. The sweep range comes from what the constraints
can reach, so a binding position cap shortens the curve instead of silently
failing half of it.

![Efficient frontier with minimum-variance, tangency, capital allocation line and the selected portfolio marked](docs/images/frontier.png)

### 2. How much of that curve is real?

The frontier is a point estimate of a curve. Resample the return history and
re-trace it, and the curve moves a great deal: on the sample panel the
expected return at a typical risk level spans a **6.3-percentage-point band**.
Differences narrower than that band are not distinguishable from estimation
noise, which is a useful thing to know before defending a 20bp allocation
difference in a meeting.

![The same frontier resampled 60 times, drawn as a confidence band around the point estimate](docs/images/frontier-uncertainty.png)

### 3. Where is the risk, as opposed to the money?

Capital weight and risk share are different quantities, and optimizers are
happy to let them diverge. Here a 26% position in EM equity carries **71% of
the portfolio's risk** — the gap between the two bars is the entire argument
for risk budgeting, and it is invisible in a weights table.

![Capital weight beside share of risk per asset, showing a 26% position carrying 71% of risk](docs/images/capital-vs-risk.png)

### 4. Would any of this have worked?

The optimizer estimated its inputs from the same returns a naive backtest
replays, so a fitted track record is a description of the past, not a
forecast. The walk-forward re-estimates and re-solves on a rolling window and
holds each solution over returns the optimizer never saw. Same rebalancing
cadence and the same 15bps of trading cost on both lines, so the gap is
overfitting rather than a cost artefact: **Sharpe 0.74 fitted against 0.23
walk-forward.**

![In-sample and out-of-sample wealth curves diverging over time](docs/images/walk-forward.png)

*Every figure above is produced by `python scripts/render_docs_images.py` from
the built-in sample dataset, using the same plotting code the app calls — so
what the README shows is what the library draws.*

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

**Estimation error**

The engine spends a lot of effort saying that expected returns and
covariances are noisy — condition numbers, T/N ratios, walk-forward
degradation — so it does not then draw the frontier as a single crisp line and
leave it there:

* `bootstrap_frontier()` resamples the return history (block, IID or
  parametric), re-estimates and re-traces the frontier on each draw, and
  returns the resulting confidence band plus the per-asset weight dispersion.
  On the sample panel the frontier's expected return spans a 6.3-percentage-
  point band at a typical risk level — differences narrower than that are not
  distinguishable from noise.
* `resampled_efficient_frontier()` implements Michaud-style resampling:
  averaging weights across draws at each frontier rank, which lifts the mean
  effective N from 3.3 to 5.2 on the sample panel because the optimizer stops
  acting on differences the data cannot resolve.

**Backtesting**

Two things a constant-weight in-sample replay hides, and this doesn't:

* `backtest_weights()` lets positions drift between rebalances, trades the
  book back on a chosen cadence, and charges transaction costs on traded
  notional — reporting turnover and the annualized cost drag.
* `walk_forward_backtest()` re-estimates and re-solves on a rolling (or
  expanding) window and holds each solution forward over returns the
  optimizer never saw. `compare_in_and_out_of_sample()` puts the two track
  records side by side with the degradation between them.

  Expected returns are re-derived inside each window by default
  (`reestimate_expected_returns=True`). This matters more than it sounds:
  `config.expected_returns` is normally populated — the UI seeds that table
  from the *full* history — so reusing it would hand every "out-of-sample"
  window an estimate built partly from its own future. On the sample panel
  that look-ahead lifts walk-forward Sharpe from 0.46 to 0.89. Turn it off
  only when your expected returns are genuine forward-looking assumptions
  rather than estimates from the same data.

## How a run flows

```mermaid
flowchart TD
    A[Prices] --> B{Data quality}
    B -->|gaps, stale feeds,<br/>splits, thin samples| B1[Reported before<br/>anything is estimated]
    B --> C[Align panel<br/>explicitly, with a log]
    C --> D[Covariance<br/>+ conditioning diagnostics]
    C --> E[Expected returns<br/>historical · EMA · CAPM · shrunk]
    D --> F{Feasibility}
    E --> F
    F -->|impossible| F1[Names the constraint<br/>and the fix]
    F -->|solvable| G[Optimize]
    G --> H{Constraints respected?}
    H -->|breach| H1[Reported, never silent]
    H --> I[Weights + diagnostics<br/>effective N · risk decomposition]
    I --> J[Frontier<br/>+ resampled confidence band]
    I --> K[Backtest<br/>drift · costs · walk-forward]
    J --> L[Excel report<br/>carrying its own assumptions]
    K --> L
```

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
├── resampling.py     # bootstrap bands + Michaud resampling
└── cli.py            # `optengine` entrypoint
app/
├── streamlit_app.py  # interactive UI
└── components.py     # reusable render blocks
config/               # example configs
docs/images/          # README figures (regenerate with scripts/)
notebooks/            # quickstart notebook
scripts/              # batch runners + docs-figure renderer
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

   ![The Data tab, leading with a data-quality verdict and per-asset coverage](docs/images/app-data.png)
2. **Assets** — per-asset statistics (extended metrics on a toggle) plus a
   drawdown-episode table with peak, trough, recovery and time underwater.
3. **Assumptions & constraints** — editable expected returns, weight bounds,
   groups, group budgets, per-method inputs (risk budgets, Black-Litterman
   views both absolute and relative), and a **live feasibility check** that
   names the constraint making the problem impossible and what to change,
   before you ever press solve.

   ![The constraints tab, with the method card and the live feasibility check](docs/images/app-constraints.png)
4. **Optimize** — a compliance banner, KPI cards, concentration and
   diversification measures, a capital-vs-risk chart, the full Euler
   decomposition, and a frontier marked with the minimum-variance and
   tangency portfolios, the capital allocation line, and where your portfolio
   actually sits — plus an opt-in panel that redraws the frontier as a
   confidence band and names the positions the sample cannot pin down.
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
    --frontier --resample 50 --walk-forward --cost-bps 10 --strict
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

The README figures are regenerated with:

```bash
python scripts/render_docs_images.py
```

The suite covers the covariance estimators and their PSD repair, every
optimizer, frontier monotonicity and reachability, ERC properties,
Black-Litterman blending with absolute and relative views, constraint
compliance, feasibility diagnosis, backtest drift/costs, walk-forward
look-ahead safety, data-quality detection, and the CLI's exit codes.

## License

MIT — see [LICENSE](LICENSE).
