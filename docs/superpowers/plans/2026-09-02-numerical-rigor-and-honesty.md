# Numerical rigor, honest failures, and the pre-trade layer — Implementation Plan

> **For agentic workers:** implement task-by-task. Steps use checkbox (`- [ ]`) syntax.
> Every task is TDD: write the failing test first, watch it fail for the right reason, then fix.

**Goal:** Every number the engine reports is the number the method computed, in a nameable
unit, with a loud failure when the method could not deliver. Then add the pre-trade layer:
structural feasibility, stress scenarios, a point-in-time universe.

**Spec:** [`../specs/2026-09-02-numerical-rigor-and-honesty-design.md`](../specs/2026-09-02-numerical-rigor-and-honesty-design.md)

**Baseline at plan time:** `main` @ 0.5.3, `881 passed, 1 skipped` in 134 s
(Python 3.11.15, pandas 3.0.5, numpy 2.4.6, cvxpy 1.9.2; CLARABEL/SCS/OSQP/HIGHS present, ECOS absent).

---

## Audit: what the spec gets wrong about the code

Every item below was verified against source before planning. **The spec is a design
document, not a survey of the tree** — these corrections are binding on the implementer.

### Already true — do not "fix" it

| Spec says | Reality |
| --- | --- |
| §1.1 "the optimizer must set `bounds_mode` on the instance *and* on the result" | Already both. `max_diversification.py:113` sets `self.bounds_mode = "soft_iterated"`; `base.py:331-335` copies it into `extras` *after* `_solve()` runs at `:311`. Nothing to do. |
| §3.1 add `accept_inaccurate` and raise `SolverFailure(..., attempts=...)` | The parameter exists (`_cvxpy_helpers.py:135-139`, default `True`) and the raise with `attempts` exists (`:228`, `SolverFailure.__init__` at `:82`). The work is **flipping the default and threading it**, nothing else. |
| §3.2 build `audit_weights` checking budget/box/layers/leverage/turnover/TE/active share | `check_constraints()` (`optimizers/diagnostics.py:52-119`) already checks **every one of those**, and `optimize()` already runs it (`base.py:346-360` `_post_solve_diagnostics`). §3.2 is a public wrapper plus the `strict_mandate` gate — not a new checker. |
| §3.3 `FeasibilityReport` gains `issues` | `issues: tuple[FeasibilityIssue, ...]` already exists (`feasibility.py:42-85`), as do `min_return`/`max_return` (the spec's `reachable_return`). Only `stage_reached` is new. |
| Decision 3 "documented once, in `OptimizerSpec`" | Already at `config.py:32`. Only the `docs/ERRORS.md` cross-reference is missing. |

### Factually wrong — the implementer must deviate

| § | Spec claim | Correction |
| --- | --- | --- |
| 1.1 | `requirements.py` has `allowed_bounds_modes` | No such field. It is `MethodRequirements.bounds_mode` (`requirements.py:64`) typed `BoundsMode = Literal["hard","soft_iterated","constrained"]` (`:14`). Adding a value requires editing the `Literal` **and** the `bounds_note` dict (`:93-106`), which does a total lookup and `KeyError`s on an unlisted mode. |
| 1.1 | the UI compliance banner reads `bounds_mode` | `render_compliance` (`app/components.py:193-234`) reads `fallback_reason`, `projection_distance`, `violations`, `ignored_constraints` — **never** `bounds_mode`. The fallback is mislabelled today, not silent. |
| 1.4 | "there must be no other bare `pct_change()`" | Two more: `analytics/performance.py:68` (`annualize_volatility`) and `:127` (`annualize_returns`), both in `if prices:` branches. Same hazard. |
| 1.5 | off-index rows "vanish" / become buy-and-hold | Overstated. The target still reaches the book at the next calendar mark via `searchsorted(side="right")` (`runner.py:191-196`); what is lost is the **decision timing**. Reachable only through a caller-supplied `weights` frame — `rebalance_dates` (`calendar.py:16-50`) derives every mark from `index` itself. |
| 1.7 | "bump `RunMeta.hash_version` to 2" | `hash_version` **does not exist** on `RunMeta` (`results.py:52-81`). It must be added with a default (the dataclass is frozen and built positionally in places). |
| 1.8 | return `.weights` "for the app" | The app never calls `resampled_efficient_frontier` — it imports `bootstrap_frontier` only (`app/streamlit_app.py:128, :1961`). Callers are `__init__.py` and `tests/test_resampling.py` alone. |
| 1.8 | one silent drop | Two. `resampling.py:366-367` (`except Exception: continue`) **and** `:361-362` (`if ok.sum() < 2: continue`), which drops without an exception. Count both. |
| 1.9 | anchor keys `gmv` / `tangency` | Fields are `min_variance` and `tangency` (`frontier.py:61-62`); row labels are `"Minimum variance"` / `"Maximum Sharpe"`. Use the **field** names as keys. |
| 1.11a | the function is `_pick_matrix` | It is the **public** `build_pick_matrix` (`black_litterman.py:119`), imported by `tests/test_analytical_rigor.py:357,380`, and its docstring (`:122,:130-132`) *documents* the dropping as intended. This reverses a documented behaviour. |
| 1.14 | a `dict` subclass warns on read | It does not. `base.py:222` `**self.extras` and `payloads.py:175` `dict(...)` both use the C-level fast path and never call `__getitem__`. The shim would be a false promise. |
| 2.2 | "both NCO layers call `_solve()`" | One call site — `nco.py:176`, reached from both layers via `_sub_solve`. |
| 2.3 | `mypy --baseline` | Not a mypy flag. `--baseline` belongs to the third-party **`mypy-baseline`** package. Mainline mypy would fail with an unrecognised argument. Also `dev` (`pyproject.toml:90`) has no mypy. |
| 2.4 | the `dropna` is in `_cmd_backtest`, fix under `--verbose` | It is `cli.py:942` inside **`_prepare_inputs`**, shared by `optimize`, `backtest` **and** `check`. And **there is no `--verbose` flag in `cli.py`** — zero matches. |
| Testing | "the allowlist is empty after §1.8–1.9" | The AST scan finds **five** silent-swallow sites, not two: `frontier.py:192`, `frontier.py:205`, `feasibility.py:652`, `hrp.py:202`, `resampling.py:366`. |

### Risks the spec does not mention

1. **§1.3 creates a *new* inconsistency.** `sortino_ratio` (`performance.py:226-231`) and
   `calmar_ratio` (`:248-253`) build their numerators from `annualize_returns` — geometric.
   An arithmetic Sharpe next to a geometric Sortino in the same `summary_stats` is the same
   class of bug §1.3 exists to kill. **Decision: leave them geometric (out of scope) and say
   so in their docstrings and in the summary-stats docstring.** Measured on the sample panel,
   equal-weight: geometric Sharpe `0.5950`, arithmetic `0.6238` — **+4.8%**. That is the
   number the CHANGELOG quotes.
2. **§1.10(a) is a behaviour change, not a relabel.** `n_observations` sets the
   Marchenko-Pastur ratio `q = T/n` (`denoise.py:486`). Dropping T from ~2500 to 17 pushes
   `q` below 1 and trips the guard at `denoise.py:488-491`. The EWMA+denoise path will
   change its eigenvalue split, or refuse. Handle the `q < 1` case explicitly.
3. **§1.12 collapses the frontier's dominated branch.** Under `>=`, every target below GMV
   returns the GMV portfolio, so `is_efficient` (`frontier.py:365-368`) flips true for all of
   them and `plot_frontier`'s "Dominated (below min-variance)" trace
   (`reporting/plots.py:180-198`) goes permanently empty. Latent — no caller passes
   `efficient_only=False` — but `feasibility.py:605-616`'s `target_return_inefficient`
   message ("solvable but dominated") becomes **false**. **Decision: accept the collapse**
   (it is what §1.12's own acceptance test demands), remove the dominated trace, and rewrite
   the feasibility message to "below the minimum-variance return; you will get the
   minimum-variance portfolio".
4. **§2.2 × §3.2 collide.** `_sub_constraints()` (`nco.py:139-149`) deliberately drops the
   mandate's box and group bounds — NCO applies them by projection *after* both layers
   (`nco.py:234-236`). Routing sub-solves through `optimize()` while `optimize()` audits
   means every cluster is audited against a constraint set that intentionally omits the
   mandate. **The audit needs an explicit per-solve opt-out, used by NCO.**
5. **§2.2 moves numbers.** `_clean_weights` (`base.py:388-421`) zeroes |w| < 1e-6 at
   *cluster* scale, which is not book scale. `tests/test_clustered_optimizers.py:129`
   compares NCO weight stability numerically and may move.
6. **§1.10(c) has the largest test blast radius in the plan** — every μ fixture in the
   suite. Also: `covariance.py` accepts exactly four methods (`"mean"`, `"ema"`,
   `"shrunk_mean"`, `"capm"`), while `config.expected_returns_method` uses a *different*
   vocabulary translated at `resampling.py:220-225` and `:343-348`.
7. **§1.7 weight-path hashing is O(periods × assets).** Row-by-row `f"{v:.12g}"` costs
   ~0.3–1 s per hash on a 20y × 50-asset run, paid once per sweep cell. Hash the rounded
   array with one `digest.update(arr.tobytes())` instead.
8. **§2.3 Windows runner:** the `smoke` job hard-codes `/tmp/ci_report.xlsx` and
   `/tmp/ci_backtest.xlsx`; `release.yml` uses `/tmp/report.xlsx`. Only the `test` job may
   gain `windows-latest`.

### Decisions taken (spec's "Open decisions")

1. **BL posterior covariance** — the spec itself defers it (§1.11(c), "Not changed in this
   increment"). **Out of scope.** No action.
2. **mypy baseline** — the spec's mechanism does not exist. **Use an explicit
   per-module allowlist in `[tool.mypy]`** (`[[tool.mypy.overrides]] ignore_errors = true`
   for modules not yet clean). No third-party CI dependency; the burn-down is "modules come
   off the override list and never go back on", enforced by a test that the list only shrinks.
3. **Universe collapse policy** — take the spec's recommendation: `to_mask(policy=...)` has
   **no default**; the CLI passes `"exclude"` and prints the count.
4. **§1.14 deprecation** — the read-shim cannot work. Instead: **write both key pairs** for
   one release and emit one `DeprecationWarning` per solve naming the old keys; migrate the
   in-tree consumer `frontier.py:378` in the same commit.

---

## Increment 1 — numerical correctness (0.6.0)

### Task 1.1 — Max-diversification: infeasible is not a fallback

**Files:** `optimizers/max_diversification.py`, `optimizers/requirements.py`, `tests/test_optimizers.py`, `tests/test_requirements.py`

- [ ] Test first: `test_max_div_infeasible_mandate_raises` — a `max_tracking_error` below the
      reachable minimum raises `SolverFailure` with `status="infeasible"`; nothing returned.
- [ ] Test: `test_max_div_fallback_never_reports_optimal` — monkeypatch `solve_problem` to
      raise `SolverFailure("solver_error", ...)`; assert `extras["solver_status"] ==
      "fallback_projection"`, `extras["bounds_mode"] == "soft_iterated"`, and
      `extras["dropped_constraints"]` non-empty when a TE budget was set.
- [ ] Replace `except Exception` (`max_diversification.py:70-73`) with `except SolverFailure`;
      re-raise when `exc.status in {"infeasible", "unbounded"}`; keep a separate
      `except Exception` for genuinely unexpected errors routed to the fallback.
- [ ] In `_fallback_projection`: **do not** `self._diagnostics.update(info.as_dict())` with the
      inner solve's status. Write `solver_status="fallback_projection"`, keep `solver`,
      `solve_seconds`, `solvers_attempted`, and add `dropped_constraints: list[str]` naming
      every constraint projection cannot represent (`max_tracking_error`, `max_active_share`,
      `leverage` — read them off `self.constraints`, list only those actually set).
- [ ] `requirements.py`: add `"hard_or_projected"` to the `BoundsMode` `Literal` (`:14`),
      add a matching entry to the `bounds_note` dict (`:93-106`) — **or it KeyErrors** — and
      change `max_diversification`'s declaration (`:460`).
- [ ] Update `tests/test_requirements.py:71-75`, which pins `"hard"`.
- [ ] Registry cross-check (parametrized over every optimizer): run once with a binding
      mandate; assert `result.extras["bounds_mode"]` is consistent with
      `requirements_for(name).bounds_mode` (`"hard_or_projected"` admits either `"hard"` or
      `"soft_iterated"`).

### Task 1.2 — One Sharpe, and the trials that feed it

**Files:** `analytics/performance.py`, `analytics/selection.py`, `analytics/relative.py`, `backtest/sweep.py`, `data/loader.py`, `data/quality.py`, `pyproject.toml`, tests

Bundled because §1.2, §1.3 and §1.4 all edit `analytics/performance.py`.

- [ ] Test: `test_sharpe_single_source` — `sharpe_ratio(r, periods_per_year=1)` equals
      `_period_sharpe(r)` to 1e-12; `rolling_metrics`'s last row equals `sharpe_ratio` over
      the same window.
- [ ] Test: `test_dsr_uses_arithmetic_trials`; `test_dsr_counts_failed_cells` (4 cells, 1
      forced failure → `n_trials == 4`); `test_dsr_and_pbo_share_a_sample` (sweep `lookback`
      over two values; trial-Sharpe index equals `return_matrix().columns`).
- [ ] Test: `test_returns_do_not_pad_gaps` — a price panel with one interior NaN yields NaN at
      that position **and** the next (no compounded jump).
- [ ] `sharpe_ratio(r, riskfree_rate=0.0, periods_per_year=252, *, method="arithmetic")` per
      the spec's snippet. `method="geometric"` reproduces today's number exactly.
- [ ] `selection._period_sharpe` → `sharpe_ratio(..., periods_per_year=1)`;
      `rolling_metrics["rolling_sharpe"]` and `sweep._cell_metrics` call `sharpe_ratio`.
- [ ] `summary_stats["Sharpe Ratio"]` becomes arithmetic; add
      `"Sharpe Ratio (geometric)"` for one release. Check `analytics/report.py:233,:330` and
      `app/streamlit_app.py:2640` still index correctly, and any test asserting row counts.
- [ ] **Document, do not change,** that `sortino_ratio` and `calmar_ratio` keep a geometric
      numerator — in both docstrings and in `summary_stats`'s.
- [ ] `sweep.py:321` `n_trials=self.n_cells`. `trial_sharpes(aligned: bool = True)` computes
      Sharpes on `return_matrix()`'s columns when true; `deflated_sharpe` and
      `overfitting_report` both use the aligned form.
- [ ] `pct_change(fill_method=None)` at **four** sites: `data/loader.py:95`,
      `data/quality.py:179`, `analytics/performance.py:68`, `analytics/performance.py:127`.
      Then grep: zero bare `pct_change(` left in `src/` and `app/`.
- [ ] `pyproject.toml`: `pandas>=2.1`.

### Task 1.3 — Backtest: schedule dates, failed first solves, stable hashes

**Files:** `backtest/runner.py`, `backtest/walkforward.py`, `backtest/results.py`, tests

- [ ] Tests: `test_offindex_schedule_dates_trade_next_bar`;
      `test_schedule_after_last_bar_is_dropped_and_noted`;
      `test_failed_first_solve_holds_cash_not_dropped`;
      `test_result_hash_stable_at_large_nav`; `test_result_hash_sees_weight_path`.
- [ ] `runner.py:184-186`: map off-index schedule dates to the next bar with
      `searchsorted(side="left")` (exact matches are idempotent); drop only dates past the
      last bar. Record `meta.notes["schedule_dates_moved"]` as `{original: executed}` and
      **record collapses** when two schedule dates land on one bar (the spec omits this).
      One `logging.warning` naming the count. `meta.notes` is not hashed — verified.
- [ ] `walkforward.py:243-247`: on failure with **no** prior book, write
      `schedule[decision_date] = pd.Series(0.0, index=returns.columns)`. Status string stays
      `f"failed: {exc}"` (note the f-string prefix — match with `.startswith`).
- [ ] `walkforward.py:265`: `evaluation = returns.loc[returns.index[min_lookback]:]`. This is
      exactly right, not off by one — the loop starts at `range(min_lookback, ...)` and
      `:219-223` guarantees `min_lookback < n`.
- [ ] `notes["periods_in_cash_after_failed_solve"]`; surface it in the tearsheet caveat.
- [ ] `results.py`: `_HASH_SIG_FIGS = 12`, `_q(v) = f"{float(v):.12g}"` everywhere the digest
      reads a float. Add the weight path — **vectorised**: round the frame, then one
      `digest.update(arr.tobytes())` plus the column/index labels; do not format per cell.
- [ ] **Add** `RunMeta.hash_version: int = 2` (it does not exist). Frozen dataclass, so give
      it a default and place it last.

### Task 1.4 — Frontier anchors, target return, inverse-vol, lying labels

**Files:** `frontier.py`, `reporting/plots.py`, `optimizers/mean_variance.py`, `optimizers/naive.py`, `optimizers/cvar.py`, `optimizers/cdar.py`, `optimizers/feasibility.py` (message only), tests

- [ ] Tests: `test_frontier_reports_anchor_failure`; `test_target_return_below_gmv_returns_gmv`;
      `test_inverse_vol_zero_variance_raises`; `test_cvar_extras_keys`; `test_cdar_extras_keys`.
- [ ] `frontier.py:192,:205`: record `anchor_failures["min_variance"|"tangency"] = str(exc)`
      on `FrontierResult.anchor_failures: dict[str, str]` (plain dataclass, add after
      `failures`). Use the **field** names, not `gmv`. `plot_frontier` renders a footnote via
      `fig.add_annotation(xref="paper", yref="paper", ...)` below the legend at `y=-0.18`.
- [ ] `mean_variance.py:106`: `mu @ w >= target_return`; set
      `extras["target_return_binding"]` (new key) from the solved slack.
- [ ] Remove `plot_frontier`'s now-empty "Dominated (below min-variance)" trace
      (`reporting/plots.py:180-198`) and rewrite `feasibility.py:605-616`'s
      `target_return_inefficient` message — "dominated" is no longer what happens.
      Update `tests/test_optimizers.py:96-107`, `test_analytical_rigor.py:448-453,:552-563`.
- [ ] `naive.py:57-63`: raise `ValueError` naming every zero-variance asset.
- [ ] `cvar.py:178-179`: add `cvar_sqrt_t_scaled` / `var_sqrt_t_scaled`; **also write** the old
      keys for one release and emit one `DeprecationWarning` per solve naming them (the
      read-shim cannot work — see the audit). Migrate `frontier.py:378`, the in-tree consumer.
- [ ] `cdar.py:221`: rename to `cdar_solver_zeta`; add a real `cdar_solver_objective` =
      `zeta + sum(z)/(alpha·T)` (the value at `cdar.py:162`).

### Task 1.5 — Resampling draws and covariance estimators

**Files:** `resampling.py`, `data/covariance.py`, `data/denoise.py`, `config.py`, `app/streamlit_app.py`, `scripts/render_docs_images.py`, tests

- [ ] Tests: `test_michaud_reports_failed_draws`; `test_michaud_refuses_minority_average`;
      `test_denoise_ewma_uses_effective_observations`;
      `test_bayes_stein_degenerate_reports_zero_intensity`; `test_mean_is_arithmetic`;
      `test_geometric_mean_matches_old_formula`; `test_no_other_mu_implementation`.
- [ ] `resampling.py`: count failures at **both** `:361-362` and `:366-367`; keep the first
      error string; raise when `n_failed > n_draws_ok`; return `ResampledFrontier(weights,
      n_draws, n_failed, first_error)` mirroring `FrontierUncertainty` (`:98-118`).
      Update `tests/test_resampling.py:151,:165,:182-184`. No app change needed.
- [ ] `covariance.py:375`: pass `n_effective = round(1/(1−ewma_lambda))` under EWMA, else
      `len(returns)`. Record **both** on `DenoiseReport` (frozen — add defaulted fields).
      **Handle `q < 1`**: at λ=0.94, T=17 against n assets pushes q below 1 and trips
      `denoise.py:488-491`. Decide and test the degenerate path explicitly.
- [ ] `covariance.py:499-500`: return intensity `0.0`, matching the `LinAlgError` exit at `:503`.
- [ ] `covariance.py:554-555`: `"mean"` → `r.mean() * ppy`; add `"geometric_mean"` with the old
      formula; `"shrunk_mean"` shrinks the arithmetic mean. Extend
      `ExpectedReturnMethod` (`:32`), the descriptions dict (`:421`), `config.py:194`, and the
      **two translation maps** at `resampling.py:220-225` and `:343-348`.
- [ ] Delete `app/streamlit_app.py:331` `_historical_mu_cached`'s body and the inline copies at
      `:2181`, `:2452`; also `scripts/render_docs_images.py:91`. All call
      `expected_returns_from_history`.
- [ ] `test_no_other_mu_implementation`: grep-style test asserting one definition site.

### Task 1.6 — Black-Litterman input validation

**Files:** `optimizers/black_litterman.py`, tests

- [ ] Tests: `test_bl_view_outside_universe_raises` (absolute **and** basket forms);
      `test_bl_degenerate_view_raises` (pick row on two perfectly collinear assets).
- [ ] `build_pick_matrix` (**public**, `:119`) raises `ValueError` naming every referenced asset
      not in the universe — at `:142` (basket legs silently zeroed) and `:145-146`. Rewrite the
      docstring at `:122,:130-132`, which currently documents the dropping as intended.
- [ ] `:271-272`: replace the `1e-12` floor with a `ValueError` naming the view whose pick row
      projects onto prior variance below `1e-300`. Leave the `confidence=0.0` path (`:280-284`)
      alone — a test pins it.
- [ ] **App regression guard:** `app/streamlit_app.py:227-240` rebuilds views from a saved
      config and can carry assets absent from the current panel; today they are dropped, now
      they raise. Catch it at the app boundary and show the message instead of crashing.

---

## Increment 2 — engineering and CI (0.6.1)

### Task 2.1 — Windows-safe cache write

**Files:** `ingest/cache.py`, tests

- [ ] `cache.py:236`: retry `os.replace` 5× with 10 ms × attempt backoff on `PermissionError`;
      if `target` exists after the retries, return `True` and log at debug.
- [ ] Do **not** broaden the "target exists ⇒ True" check above the write:
      `tests/test_ingest_service.py:774` asserts `store(...) is False` when `_write_archive`
      raises, and `:781` asserts the staging dir is empty — unlink staging on final failure.
- [ ] Note for the reviewer: no production caller reads the return value
      (`ingest/service.py:299` discards it); this contract is test-facing.

### Task 2.2 — NCO through `optimize()`, and a lazy warnings filter

**Files:** `optimizers/nco.py`, `optimizers/_cvxpy_helpers.py`, tests

- [ ] `nco.py:176` (the single `_solve()` site): call `optimize().weights`. Add
      `test_nco_layers_go_through_optimize` asserting each sub-result carries `extras["solver"]`.
- [ ] Guard the `long_only=False` path: `_clean_weights` bails at `abs(s) < 1e-9`
      (`base.py:407-408`), breaking the `Σ=1` invariant `nco.py:214-218` relies on.
- [ ] Check `tests/test_clustered_optimizers.py:129` — dust-zeroing now applies at cluster
      scale. If the numbers move materially, report before proceeding.
- [ ] `_cvxpy_helpers.py:37-39`: move the filter into `solve_problem` behind a module
      `threading.Lock` + `_installed` flag. Keep the `:22-36` comment and **extend it** — the
      filter stays process-wide, which is what that comment argues for; only the install is
      lazy. Test `test_import_has_no_warning_side_effects` via `importlib.reload`.

### Task 2.3 — CLI input pipeline

**Files:** `cli.py`, `mcp_server.py`, `scripts/run_optimization.py`, `reporting/payloads.py`, tests

- [ ] `cli.py:942` (in `_prepare_inputs`, shared by `optimize`/`backtest`/`check` — **not**
      `_cmd_backtest`): replace `dropna(how="any")` with `align_panel(...)`.
- [ ] `align_panel` (`data/quality.py:347`) takes **prices** and returns
      `tuple[DataFrame, list[str]]` where the log is prose. Thread it into `--json` as
      `alignment` via `_capture`/`_emit_json` (`cli.py:1387`).
- [ ] **Add a `--verbose` flag** — there is none today — or drop that half of the item and
      print the log to stderr unconditionally. Prefer stderr: stdout must stay parseable.
- [ ] Test `test_cli_backtest_reports_alignment`: one late listing produces a non-empty
      `alignment` and the same start date as `align_panel` alone.
- [ ] Mirror the fix at `mcp_server.py:139` and `scripts/run_optimization.py:61`, which carry
      the identical line. Leave `analytics/report.py:505` and `selection.py:436` (different
      intent) but note them.

### Task 2.4 — CI: type checking, Windows, pandas 2.1

**Files:** `.github/workflows/ci.yml`, `pyproject.toml`

- [ ] Add `mypy` and `pandas-stubs` to the `dev` extra (`pyproject.toml:90` has neither).
- [ ] `[tool.mypy]`: `python_version = "3.9"` (matches `requires-python`),
      `warn_unused_ignores = true`, `no_implicit_optional = true`,
      `ignore_missing_imports = true` (cvxpy ships no stubs).
- [ ] **Not** `mypy --baseline` — it does not exist. Use
      `[[tool.mypy.overrides]] module = [...] ignore_errors = true` listing modules not yet
      clean, and a test asserting the list only shrinks.
- [ ] New `typecheck` job next to `lint`, same shape.
- [ ] `test` job gains `strategy.matrix.os: [ubuntu-latest, windows-latest]`. **Only that
      job** — `smoke` hard-codes `/tmp/ci_report.xlsx`, `/tmp/ci_backtest.xlsx`.
- [ ] Add a `pandas==2.1.*` cell to the core tests (nothing pins pandas in CI today).

---

## Increment 3 — pre-trade capabilities (0.7.0)

### Task 3.1 — Refuse `optimal_inaccurate` by default

**Files:** `optimizers/_cvxpy_helpers.py`, `config.py`, `optimizers/base.py`, `optimizers/factory.py`, `cli.py`, `app/streamlit_app.py`, `docs/ERRORS.md`, tests

The flag and the raise already exist. The work is the default and the threading.

- [ ] Flip `solve_problem`'s default to `accept_inaccurate=False` (`_cvxpy_helpers.py:135-139`).
- [ ] Fix `:229`: raise with `status="optimal_inaccurate"` when the chain ended inaccurate —
      `last_status` currently reports whatever the **last** solver said.
- [ ] `OptimizerSpec.accept_inaccurate: bool = False`. `_OPTIMIZER_KEYS` derives from
      `__dataclass_fields__`, so it is auto-accepted; `to_dict()` drops only `None`, so `False`
      round-trips.
- [ ] Thread via `BaseOptimizer.__init__` + the factory's `common` dict
      (`factory.py:344-348`). **`CVaROptimizer` and `CDaROptimizer` bypass `common`** — add it
      explicitly at `factory.py:386` and `:396`.
- [ ] **Do not thread it into `_bounds.py:159`** — that is the dust-cleanup projection called
      from `_clean_weights`; making it refuse would break every soft-bounds method.
- [ ] `risk_parity.py:134` passes its own `solvers=` chain — preserve it.
- [ ] CLI: add `--accept-inaccurate` per solving subcommand (there is no shared parser helper).
      App: an Advanced checkbox, off by default.
- [ ] Note in the docstring that HRP/HERC/NCO/naive never call `solve_problem`, so the flag is
      a no-op for them — otherwise the CLI flag looks broken.
- [ ] Tests: `test_inaccurate_refused_by_default`, `test_inaccurate_accepted_on_opt_in`.
- [ ] `docs/ERRORS.md`: add the status, and the Decision-3 risk-aversion cross-reference.

### Task 3.2 — Post-solve mandate audit

**Files:** `optimizers/audit.py` (new), `optimizers/base.py`, `optimizers/nco.py`, `config.py`, `engine.py`, `docs/ERRORS.md`, tests

`check_constraints` already does the checking. This task is the public surface and the gate.

- [ ] `optimizers/audit.py`: `audit_weights(...) -> AuditReport` wrapping
      `check_constraints` (`diagnostics.py:52`). Note its `DEFAULT_TOLERANCE` is `1e-6`, not
      the spec's `1e-5` — keep `1e-6` and let the caller override.
- [ ] `AuditReport` with `violations`, `is_clean`, `describe()`. `ConstraintViolation` already
      exists — reuse it, do not redefine.
- [ ] `OptimizationResult.audit` (plain dataclass — appendable). Add it to
      `payloads.py:161-199`, which enumerates keys explicitly, or it will not reach `--json`.
- [ ] `EngineConfig.strict_mandate: bool = False` — must be added to the dataclass, `to_dict`,
      `from_dict` **and `_CONFIG_KEYS` (`config.py:97`)**, or loading a config that sets it
      raises `ConfigurationError`.
- [ ] `MandateViolationError(AuditReport)`; list it in `docs/ERRORS.md` as recoverable.
- [ ] **NCO opt-out:** `optimize(audit=False)` (or an instance flag) used by `nco.py`'s
      sub-solves, which run against `_sub_constraints` by design.
- [ ] Naming: `--audit-log`/`audit_path` already mean the *holdout* audit trail
      (`cli.py:264`, `backtest/holdout.py`). Do not reuse those names.
- [ ] Tests: `test_audit_catches_projected_breach`, `test_audit_passes_hard_solve`, plus the
      registry cross-check that every `bounds_mode="hard"` optimizer audits clean.

### Task 3.3 — Structural feasibility diagnosis

**Files:** `optimizers/feasibility.py`, `cli.py`, `reporting/payloads.py`, tests

- [ ] Stage 1, **no solver**: box capacity vs budget, per-layer bucket capacity, parent-basis
      coherence, bounds/layers naming assets outside the universe. Any `fatal` stops here.
- [ ] Stage 2: box-and-budget only → fractional-knapsack closed form; with layers → two LPs
      (min/max `μ'w`) **through `solve_problem`**, not the raw `problem.solve()` at `:473`.
- [ ] Fix the conflation: `reachable_return_range` (`:431`) returns `None` on any exception
      (`:474-475`) *and* any non-optimal status (`:476`), which `analyze_feasibility:546` turns
      into a **fatal** `lp_infeasible`. A solver crash must report `solver_error`; a genuinely
      infeasible LP reports "box, budget and layers are each satisfiable but jointly
      impossible", naming the binding layer. Also fix the `except Exception: pass` at `:652`.
- [ ] `FeasibilityReport` gains `stage_reached` only — `issues` and `min_return`/`max_return`
      already exist. Frozen with all-defaulted fields; no caller constructs it positionally.
- [ ] `feasibility_payload` (`payloads.py:133`) uses `getattr` defensively, so new fields are
      *ignored* until added. Add them.
- [ ] CLI `check` prints issues with suggestions and exits 2 on any fatal.
- [ ] Tests: the six named in the spec, plus `test_feasibility_solver_crash_is_not_infeasible`.

### Task 3.4 — Stress scenarios, and `scenarios` → `presets`

**Files:** `presets.py` (renamed), `scenarios.py` (shim), `stress.py` (new), `config.py`, `engine.py`, `backtest/tearsheet.py`, `cli.py`, `app/`, `__init__.py`, tests

- [ ] Give `presets.py` an explicit `__all__` **first** — `scenarios.py` has none, and
      `from x import *` will not re-export cleanly without it.
- [ ] `scenarios.py` becomes the shim: re-export + a module-level `DeprecationWarning`.
      `__init__.py` is fully eager (`:191-203`, `__all__` at `:304-314`) — update both.
      Beware `ui_state.py:244 LayerPreset` / `:332 preset_by_label` already own "preset".
- [ ] `stress.py`: `Shock`, `stress_test`, `StressReport` per the spec.
- [ ] `EngineConfig.stress` — **`EngineConfig` has no tuple fields today**; `to_dict` re-copies
      collections to lists for YAML. Match that, and register in `_CONFIG_KEYS`.
- [ ] `run_engine` gains `run_stress`; `EngineRun.stress`. Four internal callers pass
      `check_feasibility=False` (`engine.py:630,:678`, `cli.py:1056`, `mcp_server.py:392`) —
      those hot loops must not pay for stress by default.
- [ ] `optengine optimize --stress shocks.yaml`; a Stress tab; a tearsheet section
      (`Tearsheet` is a plain dataclass; `describe()` at `:83-89` is the extension pattern).
- [ ] Tests: the four named in the spec.

### Task 3.5 — Universe layer

**Files:** `universe/` (new: `signal.py`, `eligibility.py`, `classification.py`), `backtest/runner.py`, `backtest/walkforward.py`, `constraints.py`, `cli.py`, `app/`, tests

Genuinely new — nothing point-in-time exists anywhere in the tree.

- [ ] `Signal`: `date × asset` `boolean`-dtype frame, Kleene `&`, `|`, `~`. Missing means "not
      evaluable", never `False`.
- [ ] `Eligibility`: `from_threshold`, `from_rank`, `from_rolling` (windows strictly **prior**
      to the evaluation date; the first `window−1` rows missing, not `False`),
      `with_hysteresis`, `hold_through`, `to_mask(policy)` **with no default**, `breadth`,
      `turnover`, `explain`.
- [ ] `Classification`: `.static()`, `.from_history()`, `label(asset, as_of)` requiring `as_of`
      when dated, `group_matrix(as_of)`.
- [ ] `ConstraintLayer.from_classification` — note `layer_from_mapping` (`constraints.py:678`)
      already builds a layer from an assignment map; this is only worth adding if it is
      genuinely **point-in-time** (per-date assignments). Otherwise call the existing builder.
- [ ] `run_backtest(..., universe=None)`: filter targets at each decision
      (`runner.py:186-197`). There is **no liquidation concept today** — a dropped name is just
      a weight change priced by the cost model. Define it explicitly.
- [ ] `walk_forward_run(..., universe=None)`: mask `window`'s columns before `solve`
      (`walkforward.py:233-238`); `.reindex(returns.columns).fillna(0.0)` at `:238` already
      zero-fills for free. **`solve` is `Callable[[DataFrame], Series]`** — eligibility cannot
      be passed into it without widening that contract, which breaks `engine.py:678`'s closure.
      Mask the frame instead.
- [ ] Decide what a failed solve carrying the previous book forward (`:246-249`) does when that
      book holds a now-ineligible name.
- [ ] Tests: the eight named in the spec, plus `tests/test_universe_to_backtest.py` e2e.

---

## Cross-cutting

- [ ] `tests/test_conventions.py` — pins Decisions 1–4 and 10.
- [ ] `tests/test_no_silent_swallow.py` — AST scan of `src/`. **Five** sites exist today;
      §1.4 and §1.5 of this plan clear `frontier.py:192,:205` and `resampling.py:366`, and
      Task 3.3 clears `feasibility.py:652`. `hrp.py:202` (`# pragma: no cover`, scipy linkage
      edge case) is the one honest allowlist entry — or fix it too.
- [ ] `CHANGELOG.md` per the spec's migration table, each row quoting what moves on the sample
      panel. The Sharpe row quotes **0.5950 → 0.6238, +4.8%**.
- [ ] Version bumps: 0.6.0 after Increment 1, 0.6.1 after 2, 0.7.0 after 3.

## Definition of done

`pytest -q` green, `ruff check src app tests scripts` clean, and every acceptance test named
in the spec present and passing — a fix without its test is not done.

---

## Execution log

What actually landed, and what execution taught that the audit did not. Each
entry names the commit and the corrections found while implementing it — the
plan above was written from a read of the source, and a read is not a run.

| Task | Commit | Landed |
| --- | --- | --- |
| 2.1 cache retry | `7924434` | Retry on `PermissionError`; staging cleanup moved into a `finally` so the new lost-the-race path cannot leak a file |
| 1.1 max-diversification | `6e2ce4f` | Infeasible re-raises; fallback reports its own status; `hard_or_projected` |
| 1.6 Black-Litterman | `085ebca` | Out-of-universe and degenerate views raise |
| 1.3 backtest honesty | `79f39ba` | Off-index dates trade; failed first solve holds cash; relative hashing + weight path |
| 1.2 one Sharpe | `79dde38` | Arithmetic Sharpe; `n_cells` trials; `fill_method=None` ×4; `pandas>=2.1` |
| 1.2b second `n_ok` site | `54b77b9` | The CLI's walk-forward sweep carried the same defect |
| 2.2 NCO + lazy filter | `84e1dc4` | Sub-solves go through `optimize()`; warnings filter installs on first solve |
| 1.4 anchors, targets, labels | `57004da` | `anchor_failures`; target return is a floor; `cvar_sqrt_t_scaled`; `cdar_solver_zeta` |
| 2.4 CI | `ff2ff0b` | mypy over 89 modules with a 45-module allowlist; Windows and pandas-2.1 cells |

### Corrections the audit missed, found only by running the code

1. **`dropped_constraints` is narrower than three names.** `max_active_share`
   is *honoured* by the projection — setting it is what forces the CVXPY
   branch, and a 5% cap on a 70% weight comes back at exactly 5%. `leverage`
   is honoured on that branch and lost only on the clip-and-redistribute fast
   path, so it is reported conditionally. Only `max_tracking_error` is
   unconditionally dropped. Listing an enforced constraint as dropped would be
   a false diagnostic in the one field written to be honest.
2. **The hash was stable by default.** `initial_capital` defaults to `1.0`,
   not `1e6`, where twelve absolute decimals is well inside float64's
   resolution. The instability the design describes is real but only reachable
   once a caller sets realistic capital.
3. **§1.12's 1e-8 acceptance tolerance is not attainable.** Re-solving GMV as
   a min-variance QP with a slack inequality agrees with the direct GMV solve
   to 5e-5 on weights when a box constraint is active — genuine solver
   tolerance in the raw solve, not weight cleaning. The test asserts 5e-5 on
   weights and 1e-6 on the reported return, with the measurement in the
   comment.
4. **The efficiency flag survives §1.12.** The plan expected the dominated
   branch to become dead code. It is dead for a mean-variance sweep only: a
   mean-CVaR sweep minimises CVaR, so its portfolios can genuinely sit below
   the minimum-*variance* return. `show_dominated` is kept as a documented
   no-op because the app and the example notebook pass it.
5. **cvxpy 1.9.2 ships `py.typed`.** The audit said otherwise. mypy therefore
   parses cvxpy's source, which uses `match` and cannot be parsed at
   `python_version = "3.9"` — the run aborted before reaching this package.
6. **mypy 2.x refuses `python_version = "3.9"`** outright, so the dev
   dependency is capped below 2 until this package's floor rises.
7. **NCO's numbers barely moved** — cross-sample weight drift shifted 1.0e-07
   against a margin of 0.44 versus 0.75, because the sub-solves are
   hard-constrained and the dust threshold finds nothing. The cost is 20–24%
   per solve, all of it the per-layer diagnostics.
8. **A sixth swallow site**, invisible to the AST scan because its body is a
   `return`, not `pass`: `factory.py:286` `except Exception: return
   expected_returns` swallows Black-Litterman's new validation, so the preview
   falls back silently while the real solve raises.

### The rest of the increments

| Task | Commit | Landed |
| --- | --- | --- |
| 1.5 estimators | `da7a296` | Arithmetic mu with one definition site; counted Michaud draws; EWMA effective sample |
| 3.3 feasibility | `ac674d0` | Structural stage with no solver; a solver crash is no longer "infeasible" |
| 3.5 universe | `fd13abd` | `Signal`/`Eligibility`/`Classification`; both runners take a universe |
| 3.1 solver contract | `fde8126` | `accept_inaccurate=False`, carried by an ambient scope |
| 3.2 mandate audit | `eb44d3a` | `audit_weights`, `strict_mandate`, NCO opt-out |
| 3.4 stress + presets | `2b3f4a3` | `stress.py`; `scenarios` → `presets` with a shim |
| Wiring | `b485b67` | `--stress`, `--strict-mandate`, `--universe` with a rules loader |
| App | `7af50ec` | Stress and Universe tabs; stress in `--json`; example config files |

### More corrections execution turned up

9. **`accept_inaccurate` cannot thread through constructors.** NCO builds its
   sub-optimizers inside its own `_solve` and never tells them the caller's
   settings, and since §2.2 those sub-solves go through `optimize()`. A plain
   boolean default would have silently reset them mid-solve; the setting
   travels as an inherit-by-default `ContextVar` instead. The plan's note that
   the flag is a no-op for the clustered methods is wrong for NCO.
10. **The §3.2 acceptance test's premise is false.** HRP with a layer cap does
    *not* breach under projection — the projection puts every layer into its
    own program and meets a 20% bucket cap exactly. What a projection genuinely
    cannot represent is a turnover budget and a tracking-error budget.
11. **"Every `bounds_mode="hard"` method audits clean" is not true as stated.**
    `bounds_mode` is a claim about per-asset and group bounds; the audit checks
    everything the mandate carries, and `max_sharpe` honestly declares that it
    ignores a turnover budget. The cross-check is restricted to weight-only
    mandates.
12. **The silent-swallow count was one, not three.** Two of the sites reported
    during execution do not exist — those files contain no `except` at all.
    A sixth site does exist that a `pass`-body scan structurally cannot see:
    `factory.py`'s `return`-bodied handler, fixed in `911b6a4`.
13. **"Geometric Sharpe is below arithmetic" is false in general.** Compounding
    is convex, and its second-order term beats variance drag whenever
    volatility falls below about `sqrt(ppy-1)` times the mean. The docstring
    stated the shorthand as a rule; corrected in `686df53`.
14. **The projection honours `max_active_share`.** Setting it is what forces
    the CVXPY branch. See correction 1.

### Corrections CI found that local verification could not

15. **`pandas>=2.1` was never a floor this code could run on.** §1.4 chose it
    because 2.1 is the first release where `fill_method=None` is silent, without
    checking what else the tree needs — and `backtest/spec.py` and
    `analytics/report.py` use the `ME`/`QE`/`YE` aliases, which arrived in 2.2.
    The cell pinned to the declared floor failed 43 tests with
    `Invalid frequency: ME`. The floor is now 2.2, which is exactly what a
    version cell is for: nothing in local development or in a 3.x matrix could
    have caught a floor that was wrong on paper.
16. **The Bayes-Stein degenerate branch depended on an exact cancellation.**
    Testing the quadratic form for `<= 0` caught the case only when the BLAS
    cancelled exactly. When it left a few ulps — as 3.12's build does — the
    estimator returned intensity 1.0 beside an unshrunk vector, which is the
    claim §1.10(b) exists to prevent, reached through a different door. The
    guard is now on the deviation's own scale, so it does not vary by
    interpreter. §1.10(b)'s fix was close to unreachable on real data before.
17. **`typing.get_type_hints` cannot be used on `EngineConfig` at 3.9.** It
    evaluates every annotation on the class, several of which are PEP 604
    unions — a runtime `TypeError` there, even under
    `from __future__ import annotations`. The library never evaluates them, so
    only a test hit it; the expected-return vocabulary now has a name of its
    own to read instead.
18. **The typecheck job failed on its own first run**, on the three modules
    that import `yaml`. It passed locally only because a stub package had been
    installed by hand into the development environment and never reached the
    `dev` extra — the local green was not reproducible, which is the failure
    the job exists to catch.

### Still open

- `app/streamlit_app.py:1890,:1919` — the dominated-branch checkbox is inert.
  `show_dominated` can come out of `plot_efficient_frontier` after 0.7.x.
- `reporting/plots.py` still labels the mean-CVaR axis "annualized", the same
  false claim as the key renamed in §1.14; it coordinates with an app caption
  and a test, so it wants its own change.
- `app/components.py:404 render_frontier_health` shows `failures` but not
  `anchor_failures`. The chart footnote covers the visual case.
- The app's Excel export carries no stress sheet: the Report tab claims sheet
  parity with `optengine optimize`, so this belongs in `reporting/exporters.py`
  rather than in the app.
- Three nested solves — in `frontier.py`, `resampling.py` and
  `black_litterman.py` — still pay for the full post-solve pass. The §3.2
  opt-out applies to all three.
- `engine.py` keeps an inline expected-return vocabulary map that duplicates
  `expected_return_method_for_estimator`. It is correct, but it is a second
  copy.
- The mypy allowlist stands at 45 modules and may only shrink.

### Done

Every one of the design's 27 items is implemented, with the acceptance tests
it names. `docs/superpowers/specs/2026-09-02-numerical-rigor-and-honesty-design.md`
§1.11(c) — the Black-Litterman posterior-covariance discontinuity — is the one
item the design itself defers, and it remains open by its own instruction.
