# Code review — 0.5.2

Scope: the whole package (`src/optimization_engine`, `app/`, `scripts/`, `config/`,
CI). Method: every module was read in full; every finding marked **verified**
was reproduced with a snippet or the CLI against the sample panel (pandas 3.0.5,
numpy 2.4.6, cvxpy 1.9.2, CLARABEL/SCS/OSQP/HIGHS installed, ECOS absent).
The suite passes at the reviewed commit: 824 passed, 4 skipped, 53 s.

**Status.** Fixed on this branch, with tests, and listed under *Unreleased*
in the changelog: the five headline items (B1, B2, B4, O1, O2+O3, O5) and
the second tier (E1, E2, E3, E4, E6, I1, I2, I3, I6, I8, the unknown-key
half of E8, and the `indices.yaml` currency block). Everything else below
is still open.

Things that checked out and are *not* flagged: HRP quasi-diagonalisation and
recursive bisection match the reference implementation exactly; the ERC
log-barrier, Rockafellar-Uryasev CVaR and Chekhlov-Uryasev CDaR programs are
correctly formulated; He-Litterman Ω, Marchenko-Pastur denoising/detoning,
EWMA and semicovariance are correct; Meucci minimum-torsion matches the
reference iteration; `benchmark.py`, `scenarios.py`, `active.py` are sound;
the Tiingo token really does travel in a header; the MCP server really does
not write files.

---

## 1. Bugs, ranked

Severity: **H** changes numbers a user would act on or silently violates the
mandate; **M** wrong in an edge case or misleads the reader; **L** cosmetic.

### Backtest core

| # | Sev | Where | What |
|---|-----|-------|------|
| B1 | H | `backtest/runner.py:287-289` | **Drift renormalises weights to sum to 1.** `held = grown / grown.sum()` forces full investment after one bar. A 50%-invested book is 100% invested from period 2; a 60/−40 long-short (net 0.2) becomes 264/−164. Correct denominator is `1 + held @ period` (cash earns zero and stays). Verified. Also compounds with `_as_schedule` (`:75`) dropping columns absent from `returns` without renormalising. |
| B2 | H | `backtest/runner.py:279-289` | **A NaN return in an *unheld* asset poisons the whole NAV path** (`0 * NaN`). Any late-listing or delisted name yields an all-NaN backtest. `positions.py:123` and `benchmark.py:336` fill zero; the runner has no NaN policy. Verified. |
| B3 | H | `backtest/runner.py:178-180` | **Schedule rows dated off the returns index are never traded.** Under `frequency="none"` only `index[0]` trades, so a weekly schedule stamped on Sundays becomes buy-and-hold with no warning. `searchsorted` at `:185` already knows how to map to the next bar. Verified. |
| B4 | H | `analytics/performance.py:128-130, :267`; `analytics/risk.py:125, :179`; `analytics/backtest.py:401-413`; `analytics/report.py:500-518` | **NaN-padded columns bias CAGR and hit rate and NaN-out VaR/CVaR/beta.** `annualize_returns` uses `prod()` (skips NaN) over `shape[0]` (counts NaN). `compare_in_and_out_of_sample` and `compare_performance` concatenate on the union index, so the shorter OOS column is NaN-padded: OOS CAGR −5.8% reported as −2.9%, hit rate 0.48 → 0.24, VaR/CVaR/beta NaN, and the headline `Degradation` is wrong in the direction that flatters a losing OOS run. Verified. |
| B5 | M | `backtest/walkforward.py:243-247, :265` | A failed *first* solve drops periods from the evaluation (evaluation starts at the first successful solve), contrary to the module docstring's promise. Verified: 20 of 25 OOS periods evaluated. |
| B6 | M | `backtest/results.py:28-33` | Result hash rounds to 12 *absolute* decimals; at NAV ≈ 1e7 that is below float64 resolution, so one ulp of BLAS noise changes the hash. Verified (stable at NAV=1, unstable at NAV=1e7). Round relatively (`f"{v:.12g}"`). Hash also omits the weight path and `notes`. |
| B7 | M | `backtest/sweep.py:261-267` vs `:319-324` | `n_failed` docstring: "these still count as trials"; `deflated_sharpe()` passes `n_trials=self.n_ok`. Also `return_matrix()` inner-joins and `dropna`s, so when the grid sweeps `lookback` the PBO matrix is truncated to the shortest cell while `trial_sharpes` uses full streams. |
| B8 | M | `backtest/holdout.py:208-218, :249, :266` | `final_holdout_run` hands `evaluate` only the post-boundary segment, so a walk-forward cannot warm up without either a dead first `lookback` periods or a closure that quietly holds full history. `family_hash` of `strategy=None` equals that of `{}`, so unrelated unlabeled strategies trip `SHIFTED_HOLDOUT` on each other. Audit row is written after `evaluate` returns, so an `evaluate` that raises leaves no record. |

### Optimizers and estimators

| # | Sev | Where | What |
|---|-----|-------|------|
| O1 | H | `optimizers/_cvxpy_helpers.py:477-514`; `mean_variance.py:173-181`; `max_diversification.py:464-471`; `frontier.py:197` | **`leverage` and `fully_invested=False` are silently dropped by every ray-space solve** (max_sharpe, max_diversification, and the frontier's tangency anchor). `build_scaled_constraints` never translates `‖y‖₁ ≤ L·κ` (one line) and cannot express an open budget. Verified: leverage cap 1.2 → gross 2.495, `ignored_constraints=None`. The `fully_invested=False` case cannot even be caught post-solve. |
| O2 | H | `optimizers/black_litterman.py:209, :410-416`; `mean_variance.py:124` | **BL with no views does not return the market portfolio.** Prior is `π = δΣw` (FOC of `μ'w − (δ/2)w'Σw`), but the sub-solve is `MeanVarianceOptimizer(risk_aversion=δ)` whose utility is `μ'w − λw'Σw` with no ½. Effective aversion is 2δ; the no-view solution is exactly `½·w_mkt + ½·w_GMV`. Verified numerically. Also the factory feeds `spec.risk_aversion` (default 1.0) as δ while the class documents 2.5. |
| O3 | H | `optimizers/black_litterman.py:427-428` | **`optimize()` is not idempotent**: `_solve` assigns `self.cov_matrix = post_cov; self.expected_returns = post_mean`, so the next call reverse-optimises from the posterior. Verified: weight A 0.5315 → 0.5217 → 0.5121 over three calls on one instance. |
| O4 | H | `optimizers/max_diversification.py:484-528` | **`except Exception` swallows `SolverFailure(infeasible)`**, solves unconstrained, projects (dropping the TE budget), and reports `solver_status="optimal"`. Verified: TE 26.55% vs limit 0.10%, status optimal, while `min_variance` on the same inputs raises correctly. `bounds_mode` flips to `soft_iterated` on the instance while `requirements.py:460` advertises `hard`. |
| O5 | H | `data/covariance.py:466-494, :548-553` | **Bayes-Stein intensity computed on annualised moments is ~250× too weak.** Jorion's `λ = (N+2)/((N+2)+T·q)` needs the per-period quadratic form; `shrunk_mean` passes annualised μ and Σ. Verified: intensity 0.748 per-period vs 0.012 as shipped; on pure noise the cross-sectional spread of means barely moves (0.0787 → 0.0779). Docstring also claims `risk_free_rate` feeds the target; it does not. |
| O6 | M | `frontier.py:298-301`; `engine.py:1034-1041` vs `:1007` | For a BL config the frontier is traced on the raw μ, not the posterior the optimizer used (feasibility two lines earlier correctly uses `effective_expected_returns`). `effective_expected_returns` swallows every exception (`factory.py:286-287`). |
| O7 | M | `frontier.py:137-139, :317-318` | `sweep="risk_aversion"` clears `target_return` but not `target_volatility`, and CVaR ignores `risk_aversion`, so both cases produce N identical points with no warning. Verified. |
| O8 | M | `frontier.py:151-208, :303, :341-367` | The mean-CVaR frontier uses GMV/tangency anchors and a variance-based `is_efficient` test; the dominated branch of a CVaR frontier starts at the min-CVaR return, not the GMV's. |
| O9 | M | `optimizers/benchmark_relative.py:301, :329-337` | Active-MV's α shift `μ − μ·b` is objective-neutral only if `1'b = 1`; `benchmark_vector` allows a subset benchmark. Reported `expected_active_return` 0.0178 vs true 0.0288. Verified. |
| O10 | M | `optimizers/mean_variance.py:165-171` | Max-Sharpe refuses whenever all excess returns ≤ 0 regardless of `long_only`; a long-short book with unequal negative excess returns has a positive-Sharpe solution. |
| O11 | M | `optimizers/black_litterman.py:258-291` | Posterior covariance is Σ with no views and `(1+τ)Σ` with any view, however vague — a discontinuity the sub-solve inherits. |
| O12 | M | `data/covariance.py:373-379` | EWMA + denoise passes nominal `len(returns)` to the Marchenko-Pastur ratio; the effective sample is `1/(1−λ)` (~17 at 0.94), so most eigenvalues are classed as signal. `covariance_diagnostics` already computes effective observations. |
| O13 | M | `resampling.py:198-233, :363-365` | `bootstrap_frontier` raises without `config.expected_returns`; when it runs, the centre curve uses the configured vector while every draw re-estimates μ historically, so the band is centred on a different estimator than the line. Rank alignment by column position misaligns after any failed point. |
| O14 | L | `optimizers/cdar.py:361, :408` | `cdar_solver_objective` stores ζ (the DaR threshold), not the CDaR objective. CVaR names the same quantity `cvar_solver_zeta`. |
| O15 | L | `frontier.py:183-206`; `feasibility.py:473-512`; `_bounds.py:144-148` | More silent fallbacks: `_anchor_portfolios` `except: pass`; feasibility conflates a solver crash with "LP infeasible" and uses the default solver rather than the chain; the fast projection path enforces `sum=1` even when `fully_invested=False` and ignores `leverage` (the path HRP/HERC/NCO/naive take when no layers are set). |

### Engine, CLI, MCP, config

| # | Sev | Where | What |
|---|-----|-------|------|
| E1 | H | `cli.py:383-387` vs `:1105-1107`; `:374-381`; `:411` | **`check` and `optimize` do not validate the same mandate.** `optimize` refuses a config whose `expected_returns` do not overlap the panel; `check` proceeds. `optimize` applies FX, `--benchmark`, `--max-tracking-error`, `--max-active-share`; `check` accepts none of them. Verified: `optimizer: risk_parity` alone → `check` "Ready to optimize", `optimize` "Config has no expected returns matching the price columns." Contradicts AGENTS.md, ERRORS.md and the comment at `cli.py:724`. |
| E2 | H | `mcp_server.py:311-320` | MCP `optimize` calls `run_engine` with `raise_on_infeasible=False`, so the documented `ToolError` path for an infeasible mandate is unreachable; `SolverFailure` escapes and the SDK discards its message. Verified with a stubbed SDK. |
| E3 | H | `mcp_server.py:152-153` | MCP `optimizer=` override replaces the whole `OptimizerSpec`, dropping `risk_free_rate`, `target_return`, `risk_budget`, `bl_views`… A max-Sharpe run against rf=0 instead of the config's 0.04. Verified. |
| E4 | H | `mcp_server.py:365-378` | MCP `backtest` builds `BacktestSpec` without `periods_per_year` (default 252 on a monthly config) and without `frequency="none"` (spec default `monthly`, whereas engine and CLI use `none`); tearsheet drops the config's rf; lookback/step are hardcoded daily numbers. The "same payloads as `--json`" claim is false in the numbers. |
| E5 | M | `cli.py:366-371, :928-977, :1102-1104` | Only `YahooFinanceError` is caught around price loading; `FileNotFoundError`, `IngestError`, `LayerConfigurationError`, `ConfigurationError` and (in `backtest`) `SolverFailure` print tracebacks and exit 1. ERRORS.md:224-231 says otherwise. Verified. |
| E6 | M | `cli.py:937-938` | `backtest` seeds `expected_returns = {asset: 0.0}` for the initial solve, overriding `resolve_expected_returns`; a `target_return` config with no explicit μ is infeasible before the walk-forward starts. Leftover from before `resolve_expected_returns` existed. |
| E7 | M | `config.py:234-235` vs `engine.py:787-793, :992` | Benchmark precedence differs: `benchmark_weight_map` (explicit vector wins) drives the TE/active-share *constraints*; `resolve_benchmark(config.benchmark)` (spec wins) drives the *report*. Verified disagreement with `benchmark=equal_weight` + `benchmark_weights={US_Equity:1}`. |
| E8 | M | `config.py:302-373`; `constraints.py:125-128` | Validation gaps: unknown top-level keys silently dropped (`max_tracking_eror`, `group_bound`, and `asset_currency:` in the shipped `config/indices.yaml` — verified `currencies == {}` so DAX/NIKKEI/FTSE are never converted); unknown optimizer keys raise a bare `TypeError`; `periods_per_year: 0`, `covariance_method: bogus`, `Literal` fields never checked; bounds not checked for `lo <= hi`. |
| E9 | M | `constraints.py:298-309, :237-240` | `legacy_group_layer` and `ConstraintLayer.from_dict` document `group -> max` but crash with `'float' object is not subscriptable`. Only `layer_from_mapping` implements it. |
| E10 | M | `constraints.py:390-403` | `effective_layers` docstring claims parent validation it does not perform; a dangling `parent:` surfaces from inside the optimizer. ERRORS.md:108 repeats the claim. |
| E11 | M | `constraints.py:759-762` | `currency_layer` leaves assets absent from `currencies` *uncovered*, while `apply_fx_conversion` and `ui_state` treat absence as base currency; a "local ≤ 70%" cap silently excludes the cash line. |
| E12 | M | `engine.py:685-699`; `analytics/backtest.py:401-413` | `in_vs_out_of_sample` compares a costless constant-weight in-sample stream against a costed, drifting walk-forward. Immaterial on the sample panel (Sharpe 0.883 vs 0.882 matched) but the README's "same cost on both lines" sentence quotes numbers from this method. |
| E13 | L | `reporting/plots.py:595, :1040` | `plot_rolling_metrics`/`plot_rolling_relative` import `plotly.subplots` directly, bypassing `MissingDependencyError`. |
| E14 | L | `reporting/payloads.py:64, :149, :230` | `--json` `issues` arrays contain dataclass reprs (`DataIssue(severity='warning', …)`) instead of `describe()`. |
| E15 | L | `cli.py:957-959`; `reporting/exporters.py:71-89` | Every `SpecValidationError` in `backtest` gets "Pass --initial-capital." appended; `unique_sheet_name` does not strip the Excel-illegal characters it documents. |

### Ingestion, data, app

| # | Sev | Where | What |
|---|-----|-------|------|
| I1 | H | `ingest/service.py:258-283, :237-239` | **Partially-failed panels are cached**; warm hits re-label a transient failure as `missing` for the TTL and never retry. Verified. |
| I2 | H | `ingest/service.py:272-283, :540-603` | **An FX failure poisons the cache**: the unconverted panel is stored under a `currency=USD` fingerprint and served later with only "Served from cache". Verified. |
| I3 | H | `ingest/spec.py:212-229` | `fingerprint()` ignores `provider_options` (`path=`, `sheet_name=`, `seed=`), so with `--cache-dir` the `file` provider serves file A's panel for file B. Verified: identical hash for two paths. |
| I4 | M | `ingest/service.py:619-651`; `fields.py:97-107` | `require_volume=True` is a no-op for `InstrumentKind.UNKNOWN` (Tiingo always, FMP equities, file), which is then described as "index or rate levels". Verified. |
| I5 | M | `app/streamlit_app.py:452-486, :1024-1042`; `data_sources.py:363-375` | Ingest currency metadata never reaches the app's currency layer; the sidebar seeds every asset as base currency. The shipped "Mexico + US" preset with "Leave as quoted" yields MXN and USD series both labelled USD. Editing the table after an ingest-side conversion double-converts. |
| I6 | M | `cli.py:365-380`; `engine.py:93-107` | `--ingest-currency` plus `config.currencies` double-converts. |
| I7 | M | `app/streamlit_app.py:339-382`; `data_sources.py:545-549` | The uploader advertises long format but reads wide only: a long CSV loses half its rows to "duplicate dates" then crashes in `prices_to_returns`. `ingest/providers/file.py` already has the right reader with layout detection and validation. Verified. |
| I8 | M | `data/fx.py:215-221` | `ffill().bfill()` back-fills leading FX gaps with a *future* rate (look-ahead); `fill="bfill"` calls `fillna(method=)`, removed in pandas 3 (`pyproject` pins `pandas>=2.0` unbounded). Verified. |
| I9 | M | `ingest/providers/fred.py:134-139`; `data/fred.py:86-158` | One bad FRED id fails the whole batch (up to 25 ids); FRED and Yahoo bypass `_get_text`'s retry/redirect/status handling, so `errors.py:6-10` ("the service retries transient errors") is not true for them. Verified. |
| I10 | M | `ingest/service.py:568-574`; `app/streamlit_app.py:474-486`; `data_sources.py:189-190` | FX rates are fetched once per price field (5 FRED calls for OHLC+raw) and the app re-fetches on every Streamlit rerun with no cache. |
| I11 | M | `app/streamlit_app.py:1681-1715, :1957-2004` | `last_run` survives data/config changes and is captioned with the *current* settings; the Backtest tab then mixes the old run with the new panel. `frontier_uncertainty` is never reset on a new solve. |
| I12 | M | `app/streamlit_app.py:2146-2174, :2417-2427, :2573-2616, :3475` | Backtest tab, walk-forward, Performance tab and Report tab each replay under a different cost model (full spec / commission+slippage only / own slider / defaults), and the "In-sample vs out-of-sample" table compares across them. `EngineRun.walk_forward` has no `spec`/`volumes`/lag argument, so impact pricing vanishes exactly where it matters. |
| I13 | L | `ingest/providers/yahoo.py:187-197` | `end = today + 1` captures today's partial bar as a close during market hours, and the 24h TTL freezes it. Not network-verified. |

---

## 2. Conventions that disagree with each other

These are not single bugs but three-way inconsistencies; each one has already produced at least one of the bugs above.

1. **Risk-free rate units.** Code everywhere treats `riskfree_rate` as *annual* and converts (`performance.py:133`). Docstrings on `EngineRun.absolute_summary/in_vs_out_of_sample/tearsheet`, `RunResult.summary`, `BacktestResult.summary`, `compare_in_and_out_of_sample`, `compare_performance`, `performance.py:391`, and `optimizers/base.py:261-263` say "per-period". A library user following the docstring gets a wrong Sharpe. `summary_stats` defaults 0.03 while every wrapper defaults 0.0.
2. **Cost and turnover units.** The number is applied per traded leg (`|Δw|·bps`), i.e. one-way. `CostSpec.from_bps` (`spec.py:276`) and `EngineRun.backtest` (`engine.py:346`) call it "round-trip". Turnover is `Σ|Δw|` (two-sided) but every docstring says "one-way" and presents it as "the number a desk budgets" — desks quote one-way.
3. **Three Sharpe definitions side by side.** `summary_stats["Sharpe Ratio"]` is geometric (CAGR of excess / vol); `Prob. Sharpe > 0`, `deflated_sharpe_ratio().sharpe`, `minimum_track_record_length` and `rolling_sharpe` are arithmetic. `SweepResults.deflated_sharpe` feeds geometric `trial_sharpes` into a deflation of an arithmetic selected-cell Sharpe. On a daily series with mean 0.1%/vol 3% the two differ by 2× (−0.388 vs −0.189).
4. **Mean-variance risk aversion.** `μ'w − λw'Σw` (no ½) in MV; `π = δΣw` (½ convention) in BL. Pick one, document it, and make BL consistent (O2).
5. **Historical mean.** `expected_returns_from_history("mean")` is geometric; CVaR/CDaR annualise as `(1+m)^T − 1` (arithmetic-compounded); `_historical_mu_cached` in the app is a third copy.
6. **Decision date vs first holding date.** `walk_forward_run` stamps the schedule at the first holdable bar; the runner documents the schedule index as the *decision* date and `calendar.py` calls `execution_lag=1` "the desk's default". Following that advice in a walk-forward gives a two-bar gap; leaving the default `0` on a user-built signal-dated schedule is same-bar lookahead. The two entry points need different lag semantics but share one parameter and one default.
7. **Benchmark precedence** (E7). **Currency-absent assets** (E11). **`schema_version`** is `"1.0"` (str) in `payloads.py` and `1` (int) in `scenarios.py`.

---

## 3. Portfolio-construction methods: what would add the most value

Ranked by value per unit of work, and excluding what `docs/RESEARCH.md §5` already lists (EVaR, nonlinear shrinkage, entropy pooling, factor model, full ONC), which remains a good list.

1. **Fix before adding.** O1, O2, O3, O4, O5 are each a few lines and change results people are already looking at. Do these first; nothing below is worth more.
2. **Transaction-cost term in the objective.** Today there is only a hard L1 turnover cap, and the ray-space methods cannot honour even that. A linear `− c'|w − w_prev|` penalty in MV/CVaR/CDaR is trivial in cvxpy and is the single largest practical improvement for an engine whose second half is a backtester. Add `cost_aversion` to `OptimizerSpec`, and report the cost-adjusted expected return alongside the raw one.
3. **Robust mean-variance.** The SOC machinery in `build_scaled_constraints` already exists; `max μ'w − κ‖Σ_μ^{½}w‖₂ − λw'Σw` is one more constraint and one estimator of `Σ_μ` (the Bayes-Stein posterior variance falls out of O5's fix). This is the principled answer to the README's "6.3-point band" and pairs naturally with `bootstrap_frontier`.
4. **Cardinality, minimum position and round-lot.** HIGHS is already installed; cvxpy dispatches MIQP to it. A `max_positions` / `min_position` pair in `EngineConfig` covers the most common real mandate the engine cannot express today.
5. **Black-Litterman completeness.** Idzorek confidence-to-Ω mapping (confidences today are raw variances only), per-view τ, a warning when a view names an asset outside the universe (`black_litterman.py:138-149` drops it silently), and a continuous posterior covariance (O11).
6. **HERC per Raffinot.** `_cluster_risk(leaves(branch))` measures the branch at inverse-variance weights across all leaves, ignoring the cluster partition; the paper sums the risk of the constituent clusters each at its own intra-cluster weights. The current implementation is "HRP with tree-shaped splits" and the CVaR/CDaR variants evaluate the tail of a portfolio never held.
7. **Backtest realism.** No purge/embargo hooks in the walk-forward; no CPCV over paths (only CSCV over strategies); the impact model prices but never caps participation or carries partial fills; the initial purchase from cash is charged full turnover and is inseparable from steady-state turnover (10×/yr on a 25-period sample). Add `embargo` to `walk_forward_run`, a `max_participation` that carries the residual forward, and report the initial-purchase cost as its own line.
8. **Frontier for non-variance objectives** (O8): min-risk and tangency anchors defined on the objective's own risk measure.
9. **Annualising CVaR by √T** (`cvar.py:171-179`) is only valid for iid symmetric returns; report per-period or aggregate scenarios.

---

## 4. Modularity

1. **The pre-flight pipeline is hand-copied four times** — cov → diagnostics → μ → constraints → effective μ → feasibility — in `engine.run_engine`, `cli._cmd_check`, `mcp_server.check_mandate`, and `app/streamlit_app.py:1610`. E1 is the direct consequence. Extract `engine.preflight(returns, config, expected_returns=None) -> Preflight(cov, diagnostics, mu, constraints, feasibility)` and make `run_engine` consume it; then `check` cannot diverge from `optimize` because they are the same call.
2. **The CLI input pipeline is triplicated and divergent** (config → flags → prices → FX → universe filter → returns) across `_cmd_optimize/_cmd_check/_cmd_backtest`, each with a different subset. One `_prepare_inputs(args)` removes E1, E5, E6, I6. `_load_prices_for(args, config)` never uses `config`.
3. **`cli.py` (1417 lines)** splits cleanly into `cli/parser.py`, `cli/inputs.py`, `cli/json_mode.py`, and one module per command group. Three copies of the extension-dispatch writer live at `:1245-1310`.
4. **`engine.py` (1059 lines): `EngineRun` is a god object** mixing the result dataclass, allocation views, benchmark analytics, three backtest flavours, sweep, tearsheet and summaries. Split into `engine/run.py`, `engine/evaluation.py`, `engine/pipeline.py`. `_window_solver` and `sweep.evaluate.solve` are the same closure written twice.
5. **`ui_state.py` is a Streamlit module inside the core package**, with two mid-file imports that needed a dedicated E402 ignore. Move it under `app/` or `optimization_engine.ui`.
6. **`streamlit_app.py` (3538 lines)** is a module-level script whose functions close over ~25 globals (`_build_config` reads `returns`, `optimizer_name`, `base_currency`, … from module scope). Package the sidebar into one state object first; the tabs then split into `pages/` mechanically. I11 and I12 are symptoms of the globals.
7. **Legacy `data/` loaders vs `ingest/` providers** have diverged: two Yahoo ticker grammars, two error-message policies, three file readers (only one validates), FRED's own HTTP stack without retry. Make `data/yahoo.py` and `data/fred.py` private helpers of the providers, route the app uploader through `LocalFile`, and drive `_PASSTHROUGH_PROVIDERS` from `ProviderCapabilities`.
8. **Optimizer-level duplication.** The "project, record `projection_distance`, write `bounds_note`" block is pasted five times (`naive`, `hrp`, `herc`, `nco`, `max_diversification`); `_ProjectedOptimizer` exists but only the naive methods use it. `hrp.py:34-66` re-implements what `_clustering.py` provides. Risk contributions are implemented three times. `factory._REGISTRY`, `requirements.REQUIREMENTS` and `BaseOptimizer.name` are three hand-maintained copies of the method list with no assertion they agree. CVaR/CDaR/HERC take `returns` and override `assets`; everything else derives the universe from `cov_matrix`, so the factory special-cases them.
9. **Result duplication.** `RunResult`/`BacktestResult` and `WalkForwardRun`/`WalkForwardResult` duplicate turnover, cost, wealth, summary and stability properties; `rebalance_dates` means "scheduled marks" in one and "actual traded dates" in the other. PSR and rf-per-period are implemented in four places each.
10. **Serialisation is not single-sourced.** `SCHEMA_VERSION` differs in type between `payloads` and `scenarios`; `list_optimizers` and `providers --json` build their own shapes without `schema_version`; `OptimizationResult.as_dict()` claims JSON-serialisable and is not (Series, dataclasses, int-keyed dicts in `extras`).

---

## 5. Documentation

The docstring coverage push in 0.5.2 was thorough on *presence*; the review turned up a set of docstrings that are now confidently wrong, which is worse than absent because the API reference is generated from them.

- Units: every "per-period risk-free rate" (§2.1); "round-trip" cost (§2.2); "one-way turnover" (§2.2).
- `rolling_metrics.rolling_drawdown` is not rolling (`performance.py:383, :417`).
- `effective_layers` claims validation it does not do; `legacy_group_layer` documents a form that crashes; `save_config` says it does not create parents and does.
- `EngineRun.assumptions()` claims "every modelling choice" and omits `denoise`, `denoise_method`, `denoise_alpha`, `detone`.
- `EngineConfig` docstring omits `currencies`/`base_currency`; `run_engine` "Raises" omits `SolverFailure`, `ConfigurationError`, `LayerConfigurationError`, `BenchmarkError`.
- `derive_widget_state` documents `{"disabled","help"}` and returns `{"enabled","tooltip"}`; `plot_weights_bar` documents `min`/`max` columns and reads `Min Weight`/`Max Weight`.
- `ingest/panel.py:9-15` says validation catches split jumps; it checks positivity, finiteness and OHLC ordering only. `errors.py:6-10` says the service retries; it does not. `credentials.py:191-202` says a key is never partially rendered; `mask()` renders seven characters.
- `requirements.py:122` still describes the old `τ·σ_i²` BL confidence default; code uses He-Litterman.
- `docs/ERRORS.md:224-231` "does not leak tracebacks" and "solver failure → 2" (E5). `AGENTS.md` and README: `check` "runs the same analysis" as `optimize` (E1); "same 15 bps on both lines" (E12); README's provider table omits that `stooq` closes are price-return only, which the adapter itself states prominently.
- `optengine optimize --json` still writes `outputs.xlsx` to the CWD and so needs the `excel` extra even in JSON mode — unstated.

---

## 6. Test gaps that let the above through

- No `check`-vs-`optimize` agreement test on the CLI (the MCP suite has one; the CLI would fail it).
- No backtest test with a target that does not sum to one, a NaN column, or a schedule dated off the index (B1-B3) — every existing runner test uses a fully-invested, complete panel.
- No `compare_*` test where the two streams have different lengths (B4).
- No BL test asserting "no views ⇒ market weights", nor a two-call idempotence test (O2, O3).
- No max-Sharpe / max-diversification test with `leverage` or `fully_invested=False` (O1), nor an infeasible max-diversification test (O4).
- No shrinkage-intensity test against a known per-period value (O5).
- MCP: no infeasible `optimize`, no `optimizer=` spec-preservation, no `periods_per_year`/`frequency` assertion (E2-E4).
- Constraints: no `lo > hi`, bare-number limits, dangling parent, or currency-absent asset (E8-E11).
- Ingest: no partially-failed or FX-fallback cache test; no fingerprint-vs-`provider_options` test; `require_volume` only tested with `EQUITY`; `test_fx.py` covers trailing gaps only and never a non-default `fill=`; no FRED batch-with-one-bad-id test (I1-I4, I8, I9).
- App: `AppTest` runs cover the sample provider only; nothing for the uploader, a mixed-currency panel, stale-run invalidation, or `frontier_uncertainty` reset.

---

## 7. Suggested order of work

Three pull requests, each independently shippable and each with the tests from §6 that cover it.

**PR 1 — numbers that are wrong today** (small diff, large effect):
B1, B2, B3, B4, O1, O2, O3, O4, O5, E2, E3, E4, I1, I2, I3, plus the unit docstrings in §2.1-2.2. A `0.5.3` patch; the changelog should say which reported numbers move (BL weights, Bayes-Stein means, any non-fully-invested backtest, NaN-padded comparisons).

**PR 2 — one pre-flight, one input pipeline** (structural):
§4.1 and §4.2, which close E1, E5, E6, E7, E8, I6 as a by-product; then `cli.py` and `engine.py` splits (§4.3-4.4). Behaviour-preserving apart from the bugs it removes; a `0.6.0` because `EngineRun` moves modules.

**PR 3 — construction methods** (§3.2-3.6):
cost term in the objective, robust MV, cardinality via HIGHS, BL completeness, HERC per the paper. Each is its own optimizer-level change with a `describe` entry and a README row.

The app work (§4.5-4.6, I5, I7, I11, I12) is a fourth track that can proceed in parallel once PR 2 has given it a `preflight` to call.
