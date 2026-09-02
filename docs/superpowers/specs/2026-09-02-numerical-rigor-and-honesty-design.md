# Optimization Engine: numerical rigor, honest failures, and the pre-trade layer — Design

**Status:** Drafted 2026-09-02. Follows the 0.5.3 review (`docs/reviews/2026-09-01-code-review.md`) and closes the items it left open, plus a second pass over the estimation and simulation code.

**Goal:** Every number the engine reports must be (a) the number the method actually computed, (b) in a unit the caller can name, and (c) accompanied by a loud failure whenever the method could not deliver. Then extend the engine with the pre-trade capabilities it lacks: structural feasibility diagnosis, a stress-scenario layer, and a point-in-time universe layer.

**Scope:** Three increments. Increment 1 (0.6.0) fixes numbers and conventions. Increment 2 (0.6.1) hardens engineering and CI. Increment 3 (0.7.0) adds the new capabilities. Each is independently shippable; 3 depends on 1.

---

## Decisions captured

1. **One Sharpe.** The per-period **arithmetic** Sharpe is canonical because it is the quantity PSR, DSR and MinTRL are derived on. The geometric variant survives behind an explicit `method="geometric"`.
2. **Arithmetic expected returns.** `expected_returns_from_history("mean")` returns the arithmetic annualized mean. Mean-variance is a single-period model; pairing a geometric μ with an arithmetic Σ was a silent inconsistency. Geometric survives as `"geometric_mean"`.
3. **The risk-aversion convention stays as it is** (`μ'w − λ·w'Σw`, no ½). Changing it would silently rescale every user's `risk_aversion`. Black-Litterman keeps passing `δ/2`. The convention is documented once, in `OptimizerSpec`, and cross-referenced from `docs/ERRORS.md`.
4. **CVaR `alpha` remains the tail probability.** No change; recorded here so the next reader does not reopen it.
5. **Infeasible and unbounded are properties of the problem, never of the solver.** No fallback path may catch them. Only numerical failure (`solver_error`, `unknown`) earns a fallback, and a fallback never reports `optimal`.
6. **`optimal_inaccurate` is refused by default.** The fallback chain stays; it is what gives the engine a second opinion. What changes is the default when the whole chain comes back inaccurate.
7. **Failed draws, failed cells and failed solves are rows, never drops.** Wherever the code already says this, it now does it; wherever it silently dropped, it now counts and reports.
8. **Hashes round relatively, not absolutely.**
9. **`py.typed` means mypy runs in CI.** Zero silencing is the target; the first pass may land with a baseline file, which is then burned down.
10. **Every labelled quantity is what its label says.** `annualized` means an annualization that holds under the method's assumptions; `objective` means the objective; a `bounds_mode` in the registry means the mode the optimizer actually ran in.

---

## Increment 1 — numerical correctness (0.6.0)

Every item: problem → change → acceptance → migration. File paths are relative to `src/optimization_engine/`.

### 1.1 Max-diversification masks infeasibility as `optimal`

**Problem.** `optimizers/max_diversification.py:_solve` wraps `solve_problem` in `except Exception` and routes *every* failure — including `SolverFailure("infeasible")` — into `_fallback_projection`, which re-solves unconstrained, projects onto bounds (dropping any tracking-error budget, which projection cannot see), and reports the *unconstrained* solve's `solver_status="optimal"`. `requirements.py` advertises `bounds_mode="hard"` for a method that can silently run in `soft_iterated`.

**Change.**
- Catch `SolverFailure` specifically. If `exc.status in {"infeasible", "unbounded"}`, re-raise. Otherwise call the fallback.
- In `_fallback_projection`, write `solver_status="fallback_projection"` into the diagnostics, never the inner solve's status. Add `dropped_constraints` listing every constraint the projection cannot represent (today: `max_tracking_error`, `max_active_share`, `leverage`).
- `requirements.py`: `bounds_mode="hard"` → `"hard_or_projected"`; the UI's compliance banner reads the actual mode from `result.extras["bounds_mode"]`, which the optimizer must set on the instance *and* on the result.

**Acceptance.**
- `test_max_div_infeasible_mandate_raises`: a `max_tracking_error` below the reachable minimum raises `SolverFailure` with `status="infeasible"`; nothing is returned.
- `test_max_div_fallback_never_reports_optimal`: force a numerical failure (mock `solve_problem` to raise `SolverFailure("solver_error", ...)`); result `extras["solver_status"] == "fallback_projection"`, `extras["bounds_mode"] == "soft_iterated"`, `dropped_constraints` non-empty when a TE budget was set.
- Registry test: for every optimizer, run once with a binding mandate and assert `result.extras["bounds_mode"] in requirements_for(name).allowed_bounds_modes`.

**Migration.** A config that used to "succeed" with a violated mandate now raises. That is the fix. CHANGELOG under **Fixed** with the exact wording: "a max-diversification mandate the solver could not honour was returned violated with status optimal; it now raises."

### 1.2 Deflated Sharpe undercounts trials

**Problem.** `backtest/sweep.py:SweepResults.deflated_sharpe` passes `n_trials=self.n_ok`. The docstring of `n_failed` five lines earlier says failed cells "still count as trials". Additionally `return_matrix()` inner-joins and `dropna(how="any")`, while `trial_sharpes()` uses full-length streams — so when the grid sweeps `lookback`, PBO and DSR see different samples.

**Change.**
- `n_trials=self.n_cells`.
- `trial_sharpes()` gains `aligned: bool = True`; when true it computes Sharpes on the columns of `return_matrix()` so DSR and CSCV share a sample. `deflated_sharpe` and `overfitting_report` both call the aligned form.
- Both Sharpes go through the single `sharpe_ratio` of §1.3.

**Acceptance.**
- `test_dsr_counts_failed_cells`: a grid with 4 cells, 1 forced failure; the `DeflatedSharpe.n_trials == 4`.
- `test_dsr_and_pbo_share_a_sample`: sweep `lookback` over two values; assert the trial-Sharpe index equals `return_matrix().columns`.

**Migration.** Every `deflated_sharpe` result moves toward zero (more trials, lower deflated value). CHANGELOG **Fixed**.

### 1.3 One Sharpe definition

**Problem.** Three definitions coexist: geometric in `analytics/performance.py:sharpe_ratio` (`annualize_returns(excess)/annualize_volatility`), arithmetic in `rolling_metrics` (`excess.mean()·ppy / ann_vol`), and arithmetic per-period in `analytics/selection.py:_period_sharpe`. `SweepResults.deflated_sharpe` deflates an arithmetic Sharpe against a distribution of geometric ones. On a daily series with mean 0.1% and vol 3% the two differ by ~2×.

**Change.**
```python
def sharpe_ratio(r, riskfree_rate=0.0, periods_per_year=252, *, method="arithmetic"):
    rf = _rf_per_period(riskfree_rate, periods_per_year)
    excess = r - rf
    ann_vol = annualize_volatility(r, periods_per_year)
    if method == "geometric":
        return annualize_returns(excess, periods_per_year) / ann_vol
    if method == "arithmetic":
        return excess.mean() * periods_per_year / ann_vol
    raise ValueError(...)
```
- `selection._period_sharpe` becomes `sharpe_ratio(..., periods_per_year=1)`.
- `rolling_metrics["rolling_sharpe"]` and `SweepResults.trial_sharpes` call `sharpe_ratio` directly.
- `summary_stats["Sharpe Ratio"]` becomes arithmetic; a new row `"Sharpe Ratio (geometric)"` keeps the old number visible for one release.

**Acceptance.**
- `test_sharpe_single_source`: `sharpe_ratio(r, periods_per_year=1)` equals `_period_sharpe(r)` to 1e-12; `rolling_metrics` at the last row equals `sharpe_ratio` over the same window.
- `test_dsr_uses_arithmetic_trials`: the trial distribution fed to `deflated_sharpe_ratio` equals `sharpe_ratio(method="arithmetic")` per cell.
- Documented example: on the daily series above, the two methods differ and the summary shows both.

**Migration.** Headline Sharpe changes for every user. CHANGELOG **Changed**, with the formula for both and the reason.

### 1.4 `pct_change` depends on the pandas version

**Problem.** `data/loader.py:prices_to_returns` and `data/quality.py:analyze_prices` call `pct_change()` with no `fill_method`. pandas 2.0–2.2 default to `pad` (a gap becomes a 0% return followed by a compounded jump); 3.0 does not fill. `pyproject.toml` pins `pandas>=2.0` unbounded, and only 3.0 is exercised in CI. Same file, two supported versions, two return series.

**Change.** `pct_change(fill_method=None)` at both sites. `pandas>=2.1` in `pyproject.toml` (first version where `fill_method=None` is silent). Grep the tree for any other bare `pct_change()`; there must be none.

**Acceptance.** `test_returns_do_not_pad_gaps`: a price panel with one interior NaN yields NaN at that position and the *next* position (no compounded jump), on whatever pandas is installed. CI matrix adds a `pandas==2.1.*` cell for the core tests.

**Migration.** Users on pandas < 3 with gappy data see different returns. CHANGELOG **Fixed**; the pin bump under **Changed**.

### 1.5 Schedule dates off the returns index are silently discarded

**Problem.** `backtest/runner.py:run_backtest` builds `decisions` from schedule dates that are *in* the index; any other date vanishes. A weekly schedule stamped on Sundays, or a holiday, becomes buy-and-hold under `frequency="none"` with no message.

**Change.** Map every off-index schedule date to the next bar via `searchsorted(side="left")`; drop only dates after the last bar. Record the mapping in `meta.notes["schedule_dates_moved"]` as `{original: executed}`. Emit one `logging.warning` naming the count.

**Acceptance.**
- `test_offindex_schedule_dates_trade_next_bar`: three Sunday-dated rows on a business-day index produce three trades on the following Mondays; `notes["schedule_dates_moved"]` has three entries.
- `test_schedule_after_last_bar_is_dropped_and_noted`.

**Migration.** Backtests that were silently buy-and-hold now trade. CHANGELOG **Fixed**.

### 1.6 A failed first solve shortens the walk-forward evaluation

**Problem.** `backtest/walkforward.py:walk_forward_run` only writes `schedule[decision_date]` when a previous book exists, and sets `evaluation = returns.loc[weights_history.index[0]:]`. A failed *first* solve therefore drops every period until the first success, contradicting the module docstring ("a failed solve is a row, never a drop").

**Change.**
- On failure with no prior book: `schedule[decision_date] = pd.Series(0.0, index=returns.columns)` (hold cash). Status string unchanged.
- `evaluation = returns.loc[returns.index[min_lookback]:]` — the window starts at the first *decision*, not the first success.
- `notes["periods_in_cash_after_failed_solve"]` counts the cash-held periods so the tearsheet can say so.

**Acceptance.** `test_failed_first_solve_holds_cash_not_dropped`: force the first window to raise; assert `len(evaluation) == n - min_lookback`, the first rebalance row exists with zero weights and a `failed:` status, and the tearsheet caveat mentions it.

### 1.7 Result hash is unstable at realistic NAV

**Problem.** `backtest/results.py` rounds NAV, traded weights and costs to `_HASH_DECIMALS = 12` *absolute* decimals before hashing. At NAV ≈ 1e7 that is below float64 resolution, so one ulp of BLAS noise flips the hash. The hash also omits the weight path, so two runs with different holdings and identical NAV/trades collide.

**Change.**
```python
_HASH_SIG_FIGS = 12
def _q(value: float) -> str:
    return f"{float(value):.{_HASH_SIG_FIGS}g}"
```
Use `_q` everywhere the hash reads a float. Add the weight path (`date|asset|weight` per non-zero holding, mergesort-ordered) to the digest. Bump `RunMeta.hash_version` to 2 so old hashes are not compared to new ones.

**Acceptance.**
- `test_result_hash_stable_at_large_nav`: `initial_capital=1e7`, run twice, hashes equal; perturb one NAV value by 1 ulp, still equal; perturb by 1e-9 relative, different.
- `test_result_hash_sees_weight_path`: two runs with identical NAV/trades but different held weights (constructed by hand) hash differently.

### 1.8 Michaud resampling discards failed draws without counting them

**Problem.** `resampling.py:resampled_efficient_frontier` has `except Exception: continue` inside the draw loop and averages over whatever survived. `bootstrap_frontier` reports `n_failed`; this path does not. The average is biased toward the draws where the mandate did not bind.

**Change.**
- Count failures; keep the first error string.
- Raise if `n_failed > n_draws_ok` ("an average over the minority that solved is not a resampled portfolio").
- Return a `ResampledFrontier` dataclass (`weights`, `n_draws`, `n_failed`, `first_error`) instead of a bare frame. The frame stays reachable as `.weights` for the app.

**Acceptance.** `test_michaud_reports_failed_draws`: mock every third draw to raise; `n_failed` equals the mocked count. `test_michaud_refuses_minority_average`: mock two thirds to raise; `ValueError`.

### 1.9 Frontier anchors vanish silently

**Problem.** `frontier.py:_anchor_portfolios` has two `except Exception: pass` blocks. A GMV or tangency solve that fails simply disappears from the chart.

**Change.** Catch, record `anchor_failures[name] = str(exc)`, and surface it on `FrontierResult.anchor_failures: dict[str, str]`. `plot_frontier` renders a footnote when it is non-empty.

**Acceptance.** `test_frontier_reports_anchor_failure`: a mandate that makes max-Sharpe infeasible produces `anchor_failures["tangency"]` naming the status; the GMV anchor is still drawn.

### 1.10 Covariance and expected-return estimators

Three independent fixes in `data/covariance.py`.

**(a) EWMA + denoising uses the nominal sample size.** `covariance_matrix` passes `n_observations=len(returns)` to `denoise_covariance` regardless of method. Under EWMA the effective sample is `1/(1−λ)` (~17 at λ=0.94), so most eigenvalues are classed as signal. `covariance_diagnostics` already computes the effective count.
Change: `n_effective = round(1/(1−λ))` when `method == "ewma"`, else `len(returns)`; pass that. Record both on `DenoiseReport`.
Acceptance: `test_denoise_ewma_uses_effective_observations` — same panel, `method="ewma"` vs `"sample"`, the MP edge differs and the report shows `n_observations_effective < n_observations`.

**(b) Bayes-Stein returns intensity 1.0 with unshrunk means.** `james_stein_shrinkage`: when the quadratic form is non-positive it returns `(mu, 1.0)`. Nothing was shrunk; report `0.0`.
Acceptance: `test_bayes_stein_degenerate_reports_zero_intensity`.

**(c) Historical mean is geometric.** `expected_returns_from_history("mean")` returns `((1+r).prod())**(ppy/T) − 1`. Per Decision 2, `"mean"` becomes `r.mean() * ppy`; the old formula moves to `"geometric_mean"`. `"shrunk_mean"` shrinks the arithmetic mean. The app's `_historical_mu_cached` and any other copy call this function; the copies are deleted.
Acceptance: `test_mean_is_arithmetic`, `test_geometric_mean_matches_old_formula`, `test_no_other_mu_implementation` (grep-style test asserting a single definition site).
Migration: every `max_sharpe` / `target_return` run with historical μ moves. CHANGELOG **Changed**, with both formulas.

### 1.11 Black-Litterman input validation

**(a) Views on assets outside the universe are dropped silently.** `optimizers/black_litterman.py:_pick_matrix` skips a view whose weights touch no asset in the universe (`if not touched: continue`), and silently zeroes the coefficients of assets outside the universe inside a basket view.
Change: raise `ValueError` naming every referenced asset not in the universe. A view on a name you cannot hold is not a view.
Acceptance: `test_bl_view_outside_universe_raises` (absolute and basket forms).

**(b) A degenerate view is floored to 1e-12.** `default_omega = np.maximum(default_omega, 1e-12)` silently gives a view with zero prior variance a tiny variance, i.e. near-infinite confidence.
Change: raise `ValueError` naming the view whose pick row projects onto numerically zero prior variance (threshold `1e-300`).
Acceptance: `test_bl_degenerate_view_raises` with a pick row on two perfectly collinear assets.

**(c) Posterior-covariance discontinuity (decision required, see Open decisions).** Not changed in this increment.

### 1.12 Target return as an equality

**Problem.** `optimizers/mean_variance.py`: `mu @ w == target_return`. A target below the GMV return returns a point on the *inefficient* lower branch with no warning; a target above the reachable maximum is infeasible rather than clamped.

**Change.** `mu @ w >= target_return`. Under min-variance the optimum sits on the efficient branch by construction. Feasibility of an unreachable target already raises via `analyze_feasibility`; keep that.

**Acceptance.** `test_target_return_below_gmv_returns_gmv`: with target < GMV return, weights equal the GMV weights to 1e-8 and `extras["target_return_binding"] is False`.

### 1.13 Inverse-volatility zero-variance handling

**Problem.** `optimizers/naive.py:InverseVolatilityOptimizer` gives a zero-variance asset zero weight and only raises if *every* asset has zero variance. A name silently vanishes from the book.

**Change.** Raise `ValueError` naming every zero-variance asset. Same posture as max-diversification.

**Acceptance.** `test_inverse_vol_zero_variance_raises`.

### 1.14 Labels that lie

- `optimizers/cvar.py`: `cvar_annualized = cvar × √T` is an iid-Gaussian heuristic, not an annualization. Rename to `cvar_sqrt_t_scaled` / `var_sqrt_t_scaled`; the note already attached to it stays. Keep the old keys for one release, emitting `DeprecationWarning` on read via a small `extras` shim.
- `optimizers/cdar.py`: `cdar_solver_objective` stores ζ (the DaR threshold), not the CDaR objective. Rename to `cdar_solver_zeta`, mirroring `cvar_solver_zeta`. Add `cdar_solver_objective` holding the actual objective value.

**Acceptance.** `test_cvar_extras_keys`, `test_cdar_extras_keys`; the deprecation shim is covered by `pytest.warns`.

---

## Increment 2 — engineering and CI (0.6.1)

### 2.1 Cache write is not Windows-safe under concurrency

**Problem.** `ingest/cache.py:PanelCache.store` finishes with `os.replace(staging, target)`. On Windows, `os.replace` raises `PermissionError` (`WinError 5`) when another process holds `target` open. `test_concurrent_writers_of_one_key_all_succeed` fails deterministically on Windows (1–3 of 16 writers). CI runs Ubuntu only, so it was never observed.

**Change.** Retry `os.replace` up to five times with linear backoff (10 ms × attempt) on `PermissionError`. Losing the race to another writer of the *same* key is success by the cache's own contract, so if `target` exists after the retries, return `True` and log at debug. Anything else returns `False` as today.

**Acceptance.** The existing concurrency test passes on the Windows CI runner added in §2.3.

### 2.2 `nco.py` bypasses `optimize()`

**Problem.** Both NCO layers call `optimizer._solve()` directly, skipping the weight cleaning, bounds recording and diagnostics that `optimize()` performs.

**Change.** Call `optimize().weights`. Sub-optimizers are built with `constraints=_sub_constraints(...)` as today.

**Acceptance.** Existing NCO tests pass; add `test_nco_layers_go_through_optimize` asserting each sub-result carries `extras["solver"]`.

### 2.3 CI: type checking and a Windows runner

- New job `typecheck`: `pip install -e ".[all,dev]" mypy pandas-stubs && mypy src/optimization_engine`. `mypy` and `pandas-stubs` join the `dev` extra. A `[tool.mypy]` section in `pyproject.toml`: `python_version = "3.9"`, `warn_unused_ignores = true`, `no_implicit_optional = true`. The first run may land with a `mypy-baseline.txt` consumed by `mypy --baseline` (or an equivalent allowlist) so the job is green from day one; the baseline count is reported in the job summary and must not grow.
- `test` matrix gains `os: [ubuntu-latest, windows-latest]`. Streamlit/MCP `extras` job stays Ubuntu-only.
- Core tests add a `pandas==2.1.*` cell (see §1.4).

**Acceptance.** All jobs green on `main`; `py.typed` is now backed by a check.

### 2.4 `dropna(how="any")` in the CLI

**Problem.** `cli.py:_cmd_backtest`: `prices_to_returns(prices).dropna(how="any")`. One late-listing asset truncates the sample for every other asset, with no message. `align_panel` exists and returns an action log for exactly this.

**Change.** Replace with `align_panel(...)`; print its action log under `--verbose` and include it in `--json` output as `alignment`.

**Acceptance.** `test_cli_backtest_reports_alignment`: a panel with one late listing produces a non-empty `alignment` entry and the same start date as `align_panel` alone.

### 2.5 Process-wide warnings filter at import time

**Problem.** `optimizers/_cvxpy_helpers.py` executes `warnings.filterwarnings("ignore", message="Solution may be inaccurate")` at module import. The thread-safety reasoning is sound; the side effect on `import optimization_engine` is not.

**Change.** Move the call into `solve_problem`, guarded by a module-level `threading.Lock` and an `_installed` flag, so it runs once on first solve rather than on import. Behaviour otherwise unchanged.

**Acceptance.** `test_import_has_no_warning_side_effects`: `importlib.reload` the module and assert `warnings.filters` is unchanged until the first `solve_problem` call.

---

## Increment 3 — pre-trade capabilities (0.7.0)

### 3.1 Solver status contract: refuse `optimal_inaccurate` by default

**Problem.** `solve_problem(..., accept_inaccurate=True)` walks the chain, prefers an exact answer, and settles for an inaccurate one with a logged warning. A degraded answer returned by default is how wrong books ship.

**Change.**
- Default `accept_inaccurate=False`. When the whole chain ends inaccurate, raise `SolverFailure(status="optimal_inaccurate", attempts=...)` with the message pointing at `accept_inaccurate=True` and at `analyze_feasibility`.
- `OptimizerSpec.accept_inaccurate: bool = False` threads the opt-in from config to every solve. The CLI exposes `--accept-inaccurate`; the app exposes a checkbox under Advanced, off by default.
- The fallback chain is unchanged.

**Acceptance.** `test_inaccurate_refused_by_default` (mock the chain to return inaccurate everywhere); `test_inaccurate_accepted_on_opt_in` asserts the warning and `SolveInfo.status == "optimal_inaccurate"`.

**Migration.** Problems that used to return with a warning now raise. CHANGELOG **Changed**; ERRORS.md gains the new status.

### 3.2 Post-solve mandate audit

**Problem.** `layer_breaches` and `bounds_note` *report* violations; nothing fails on them. A projection or a loose solver can return a book that violates the mandate, and only a reader of the diagnostics finds out.

**Change.**
- New `optimizers/audit.py`: `audit_weights(weights, assets, constraints, cov_matrix=None, *, tolerance=1e-5) -> AuditReport` checking sum/budget, box bounds, layer limits, leverage, turnover (when `previous_weights` given), tracking error and active share (when the covariance / benchmark are given). Every breach is a `ConstraintViolation(kind, name, limit, value, magnitude)` with `describe()`.
- `BaseOptimizer.optimize()` runs the audit after cleaning and attaches `result.audit`.
- `EngineConfig.strict_mandate: bool = False`. When true, any violation above tolerance raises `MandateViolationError(AuditReport)` (new, listed in ERRORS.md as recoverable: loosen the mandate or pick a `bounds_mode="hard"` method).

**Acceptance.** `test_audit_catches_projected_breach` (HRP with a layer cap that projection cannot fully satisfy → report non-empty; strict mode raises). `test_audit_passes_hard_solve` (constrained MV → empty report). Registry cross-check: every `bounds_mode="hard"` optimizer must produce an empty audit on every fixture mandate.

### 3.3 Structural feasibility diagnosis

**Problem.** `optimizers/feasibility.py` answers feasibility with an LP and conflates a solver crash with "infeasible" (review O15); it also uses cvxpy's default solver rather than the chain. It cannot say *which* constraint is impossible.

**Change.** Restructure `analyze_feasibility` into two stages:
1. **Structural arithmetic, no solver.** Box capacity vs budget (`Σ lb ≤ 1 ≤ Σ ub`), per-layer bucket capacity (`Σ_{i∈bucket} ub_i ≥ bucket_min`, `Σ lb_i ≤ bucket_max`), parent-basis coherence (a child bucket's limit cannot exceed its parent's), bounds on assets outside the universe, layers naming assets outside the universe. Each finding is a `FeasibilityIssue(code, severity, message, suggestion)`; any `fatal` issue stops here.
2. **Reachable-return range.** Box-and-budget only → fractional-knapsack closed form (no solver). With layers → two LPs (min and max `μ'w`) through `solve_problem` on the shared constraint translation; a solver crash is reported as `solver_error`, an LP infeasible as the specific finding "the box, budget and layers are each satisfiable but jointly impossible", naming the binding layer.

`FeasibilityReport` gains `issues`, `reachable_return: tuple[float, float] | None`, `stage_reached`. The CLI `check` command prints issues with suggestions and exits 2 on any fatal.

**Acceptance.**
- `test_feasibility_box_capacity` (Σ ub < 1 → fatal, names the shortfall).
- `test_feasibility_layer_capacity`.
- `test_feasibility_parent_coherence`.
- `test_feasibility_jointly_impossible` (box, budget, layers each fine alone, jointly infeasible → the specific message).
- `test_feasibility_knapsack_matches_lp` (box-and-budget: closed form equals LP to 1e-9).
- `test_feasibility_solver_crash_is_not_infeasible` (mock `solve_problem` to raise; report says `solver_error`).

### 3.4 Stress scenarios

**Problem.** The engine has no shock-based stress test. `scenarios.py` is configuration persistence (save/load/rename named configs); the name invites confusion.

**Change.**
- Rename `scenarios.py` → `presets.py` (`save_scenario` → `save_preset`, etc.), keeping `scenarios` as a deprecated re-export for one release.
- New `stress.py`:
  - `Shock(name, returns: Mapping[str, float], covariance_scale: float | Mapping | None = None, notes="")` — a one-period return shock per asset (assets absent default to 0), optionally a stressed covariance (scalar multiplier or a full matrix).
  - `stress_test(weights, shocks, cov_matrix=None) -> StressReport` with per-scenario P&L, per-asset P&L contributions (summing to the scenario P&L by identity), stressed volatility when a covariance is given, and `worst` (scenario name, P&L, largest contributor).
  - `StressReport.describe()` orders scenarios worst-first.
- `EngineConfig.stress: tuple[Shock, ...] = ()`; `run_engine` attaches `run.stress` when non-empty. `optengine optimize --stress shocks.yaml`; a Stress tab in the app; a stress section in the tearsheet.

**Acceptance.** `test_stress_contributions_sum` (identity to 1e-12); `test_stress_worst_named`; `test_stress_missing_asset_is_zero_shock`; `test_stress_covariance_scale`.

### 3.5 Universe layer

**Problem.** The engine's universe is "whatever columns are in the returns frame". There is no notion of an investable set that changes through time, no way to express "eligible if ADV > x for the last 63 days, with hysteresis", and no way to attach a classification (sector, country) that is only known as of a date. Every backtest is therefore run on the survivors of the final panel.

**Change.** New package `optimization_engine/universe/`, pure pandas/NumPy, no solver dependency.

- **`Signal`** — a `date × asset` boolean frame with three states: `True`, `False`, and missing (`pd.NA` on a `boolean` dtype). Kleene `&`, `|`, `~`. Missing means "not evaluable on that date", never `False`.
- **`Eligibility`** — a `Signal` plus membership rules:
  - `from_threshold(series_frame, op, value)`, `from_rank(series_frame, top_n)`, `from_rolling(frame, window, agg, op, value)` (windows strictly *prior* to the evaluation date; the first `window−1` rows are missing, not `False`).
  - `with_hysteresis(entry: Signal, exit: Signal, initial: bool | None)` — `member[t] = entry[t] | (member[t−1] & ~exit[t])` under Kleene logic; `initial=None` propagates "unknown" until the first evaluable date.
  - `hold_through(dates)` — membership evaluated only on reconstitution dates and held constant between them; rows before the first evaluation are missing.
  - `to_mask(policy: Literal["exclude", "include", "raise"])` — the single place missing collapses to a hard boolean, under an explicit policy.
  - `breadth()`, `turnover()` (entries + exits per date), `explain(date, asset) -> str` naming the rule that admitted or excluded the name.
- **`Classification`** — point-in-time labels. `Classification.static(mapping)` for labels with no history; `Classification.from_history(frame[asset, label, effective_from])` for dated labels. `label(asset, as_of)` **requires** `as_of` when the entry is dated and raises without it; returns `None` before the first record. `group_matrix(as_of)` yields the membership matrix layers consume.
- **Integration.**
  - `run_backtest(..., universe: Eligibility | None)`: at each decision date, assets not eligible as of that date are excluded from the target (weights renormalised by the optimizer, never by the runner) and, if held, liquidated at the next bar. `meta.notes["universe"]` records breadth at each decision and the policy used.
  - `walk_forward_run(..., universe=...)`: the solve window is restricted to assets eligible at the decision date; delisted names (no data after date `d`) are liquidated at their last print and recorded in `notes["delistings"]`. No look-ahead: eligibility and delisting are evaluated only on data up to and including the decision date.
  - `ConstraintLayer.from_classification(classification, as_of, limits)` builds a layer from a point-in-time classification.
  - CLI: `--universe rules.yaml`; app: a Universe tab showing the eligibility heatmap and breadth.

**Acceptance.**
- `test_signal_kleene_truth_table` (all nine combinations for `&` and `|`).
- `test_rolling_rule_first_rows_missing_not_false`.
- `test_hysteresis_unknown_initial_propagates`.
- `test_hold_through_between_reconstitutions`.
- `test_classification_dated_requires_as_of` (raises) and `test_classification_before_first_record_is_none`.
- `test_backtest_excludes_ineligible_and_liquidates_next_bar`.
- `test_walk_forward_universe_is_point_in_time`: an asset that becomes eligible at `t` is absent from every solve before `t`; a name delisted at `d` is liquidated at `d` and never traded after.
- `test_no_lookahead_in_delisting`: shuffle future data after `d`; decisions up to `d` are unchanged.

---

## Testing strategy

- Every §1 item ships with the acceptance tests named above; a fix without its test is not done.
- `tests/test_conventions.py` (new) pins Decisions 1–4 and 10: one Sharpe definition site, arithmetic `mean`, the documented risk-aversion formula, CVaR `alpha` semantics, and every `extras` key's meaning against its docstring.
- `tests/test_no_silent_swallow.py` (new): AST scan of `src/` asserting no `except Exception:` / `except:` whose body is only `pass` or `continue`; the allowlist is empty after §1.8–1.9.
- Registry cross-checks (§1.1, §3.2) are parametrized over every optimizer so a new method cannot enter the registry without declaring its `bounds_mode` truthfully.
- Increment 3 e2e: `tests/test_universe_to_backtest.py` runs rules → eligibility → walk-forward → tearsheet on the sample panel with two synthetic listings and one delisting, and asserts the trade log matches a hand-written expectation.

## Migration / compatibility

| Change | Version | Kind | What moves |
|---|---|---|---|
| Sharpe arithmetic by default | 0.6.0 | Changed | every headline Sharpe |
| `mean` arithmetic | 0.6.0 | Changed | every μ-driven optimizer with historical μ |
| DSR `n_trials` | 0.6.0 | Fixed | every deflated Sharpe, toward zero |
| Max-div infeasible raises | 0.6.0 | Fixed | previously violated books now raise |
| `pct_change(fill_method=None)`, `pandas>=2.1` | 0.6.0 | Fixed / Changed | gappy panels on pandas < 3 |
| Off-index schedule dates trade | 0.6.0 | Fixed | affected backtests trade instead of holding |
| Hash `hash_version=2` | 0.6.0 | Changed | stored hashes no longer comparable |
| `cvar_annualized` → `cvar_sqrt_t_scaled` | 0.6.0 | Deprecated (1 release) | extras key |
| `accept_inaccurate=False` default | 0.7.0 | Changed | inaccurate solves now raise |
| `scenarios` → `presets` | 0.7.0 | Deprecated (1 release) | module name |
| `strict_mandate`, `stress`, `universe` | 0.7.0 | Added | opt-in |

Each row is a CHANGELOG entry stating what moves and by how much on the sample panel.

## Open decisions

1. **Black-Litterman posterior covariance.** Today: `Σ` with no views, `Σ + M` with any view — a discontinuity at the first view. The predictive covariance is `Σ + M` always (with no views `M = τΣ`, giving `(1+τ)Σ`). That is continuous and correct, but it rescales the covariance handed to a risk-aversion solve under a budget, so the no-view book is no longer *exactly* the market portfolio under `risk_aversion`; it remains so under tangency normalization, which is scale-invariant.
   - a) `Σ + M` always; re-pin "no views reproduces the market" on the tangency road, and document that under a budgeted utility the no-view book sits at `w_mkt/(1+τ)` before renormalisation. **Recommended** — it is the definition, and the current test pins a coincidence of convention.
   - b) `Σ` always for sizing; `Σ + M` reported as a diagnostic only. Consistent and simple; understates risk where views are strongest.
   The deciding fact: whether the dominant use is budgeted `risk_aversion` (scale matters) or tangency (it does not).
2. **mypy baseline.** Land green with a baseline file that may only shrink, or block the merge until zero. Recommended: baseline, with the count printed in the job summary and a hard ceiling in the workflow.
3. **Universe collapse policy default.** `to_mask(policy=...)` has no safe default: `"exclude"` silently shrinks the book, `"include"` silently admits unknowns, `"raise"` stops every run with a warm-up period. Recommended: no default; the caller names one. The CLI defaults to `"exclude"` and prints the count.

## File-level changes summary

**Increment 1**
- `optimizers/max_diversification.py`, `optimizers/requirements.py` — §1.1
- `backtest/sweep.py` — §1.2
- `analytics/performance.py`, `analytics/selection.py` — §1.3
- `data/loader.py`, `data/quality.py`, `pyproject.toml` — §1.4
- `backtest/runner.py` — §1.5
- `backtest/walkforward.py` — §1.6
- `backtest/results.py` — §1.7
- `resampling.py` — §1.8
- `frontier.py`, `reporting/plots.py` — §1.9
- `data/covariance.py`, `app/streamlit_app.py` (delete `_historical_mu_cached`) — §1.10
- `optimizers/black_litterman.py` — §1.11
- `optimizers/mean_variance.py` — §1.12
- `optimizers/naive.py` — §1.13
- `optimizers/cvar.py`, `optimizers/cdar.py` — §1.14

**Increment 2**
- `ingest/cache.py` — §2.1
- `optimizers/nco.py` — §2.2
- `.github/workflows/ci.yml`, `pyproject.toml` — §2.3
- `cli.py` — §2.4
- `optimizers/_cvxpy_helpers.py` — §2.5

**Increment 3**
- `optimizers/_cvxpy_helpers.py`, `config.py`, `cli.py`, `app/` — §3.1
- `optimizers/audit.py` (new), `optimizers/base.py`, `engine.py`, `docs/ERRORS.md` — §3.2
- `optimizers/feasibility.py` (rewrite), `cli.py` — §3.3
- `presets.py` (renamed), `stress.py` (new), `config.py`, `engine.py`, `backtest/tearsheet.py`, `cli.py`, `app/` — §3.4
- `universe/` (new: `signal.py`, `eligibility.py`, `classification.py`), `backtest/runner.py`, `backtest/walkforward.py`, `constraints.py`, `cli.py`, `app/` — §3.5

## Out of scope

- Changing the risk-aversion convention (Decision 3).
- Transaction-cost terms inside the objective, robust mean-variance, cardinality constraints, Idzorek confidence mapping, HERC per-cluster risk measures beyond the current set — all listed in `docs/reviews/2026-09-01-code-review.md §3` and `docs/RESEARCH.md §5`; they follow this work, not precede it.
- Splitting `cli.py`, `engine.py` and `streamlit_app.py` (review §4). Increment 3 adds surfaces to them; the split is its own increment.
