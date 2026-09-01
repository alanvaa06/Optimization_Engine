# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, the public API may change in a minor
release. Every such change is listed here under **Changed** or **Removed**,
with what to do about it.

## [Unreleased]

## [0.5.3] — 2026-09-01

Five numerical fixes from a full-package review, and the interface fixes
that followed it; the review itself is in
`docs/reviews/2026-09-01-code-review.md`. Each numerical fix changes a
number a user may already be looking at, and each is listed with what moves.

### Fixed

- **Backtest drift no longer forces full investment.** The simulation core
  renormalized drifted weights by the sum of the positions, which is the
  book's growth only when the book is fully invested. Any cash residual was
  silently converted into positions after one period, and a 60/−40 long-short
  book became 264/−164 on its second bar. Weights now drift by the book's own
  growth, `1 + w·r`. A fully invested target replays exactly as before; every
  non-fully-invested backtest — `fully_invested=False`, a partial target, a
  long-short book — moves.
- **A missing return no longer poisons the NAV path.** `0 × NaN` made every
  period from the first gap onward NaN, even for a name the book never held.
  A gap is now a flat period for that asset alone; a gap on a *held* asset is
  recorded on `meta.notes["missing_returns"]` so it reads as the data problem
  it is.
- **Comparisons score both streams over their common window.**
  `compare_in_and_out_of_sample` and `compare_performance` padded the shorter
  stream with NaN, which `annualize_returns` and `hit_rate` counted as zero-
  return periods and `var_historic` turned into NaN. On the sample panel that
  reported a walk-forward CAGR of −2.9% for a stream that made −5.8%, and
  flattered the `Degradation` column whenever the out-of-sample run was
  losing. The two functions now align on the periods both streams cover, and
  the three metrics count observations rather than rows, so an isolated gap
  anywhere is neither a return nor a loss.
- **Black-Litterman with no views now returns the market portfolio.** The
  equilibrium prior `π = δΣw` is the first-order condition of
  `μ'w − (δ/2)·w'Σw`, but the mean-variance sub-solve it fed maximizes
  `μ'w − λ·w'Σw` with no half, so the effective aversion was doubled and the
  no-view answer sat exactly halfway between the market and the minimum-
  variance portfolio. The sub-solve now receives `λ = δ/2`. Every
  `black_litterman` allocation moves, toward the market portfolio.
- **`BlackLittermanOptimizer.optimize()` is idempotent.** It used to write the
  posterior back over `cov_matrix` and `expected_returns`, so a second call
  reverse-optimized from the first call's answer and the weights drifted on
  every solve. The posterior now lives on the optimizer privately; the
  result's return and risk are still reported against it, and the inputs are
  left as given. Code that read `optimizer.cov_matrix` *after* a solve and
  expected the posterior will now see the prior.
- **Max-Sharpe and max-diversification honour a `leverage` cap.** The
  homogeneous reformulation translated bounds, layers and the benchmark
  budgets into ray space and skipped the gross-exposure cap, so a 1.2× mandate
  could come back at 2.5× gross with nothing in `ignored_constraints`. The cap
  is now a hard constraint of the ray-space solve. `fully_invested=False`
  cannot be expressed on a ray at all; both optimizers now warn and record it
  in `result.extras["ignored_constraints"]` instead of silently returning a
  fully invested book.
- **Bayes-Stein shrinkage is computed on per-period moments.** Jorion's
  intensity `(N+2) / ((N+2) + T·q)` counts observations in `T`, so the
  quadratic form `q` has to be per-observation too; `shrunk_mean` handed it
  annualized means and covariance, inflating `q` by the annualization basis
  and leaving daily data some 250 times under-shrunk. `james_stein_shrinkage`
  takes a `periods_per_year` (default 252, matching
  `expected_returns_from_history`) and divides the quadratic form by it.
  Every `expected_returns_method="shrunk_mean"` vector moves, toward the
  minimum-variance portfolio's return.
- **`check`, `optimize` and `backtest` build their inputs the same way.** The
  three commands each assembled config, panel, currency, universe and
  benchmark by hand, and had drifted: `optimize` refused a config with no
  `expected_returns` block that `check` had just called ready; `check` never
  saw `--base-currency`, `--benchmark` or the two active-risk limits;
  `backtest` seeded zero expected returns for its first solve, so a return
  target with no explicit vector was infeasible before the walk-forward
  started. One `_prepare_inputs` now serves all three, `check` accepts the
  flags `optimize` does, and a config that relies on `expected_returns_method`
  solves everywhere it pre-flights. When `--ingest-currency` has already
  converted the panel, `config.currencies` is no longer applied a second time.
- **`--json` failures carry their reason.** A command that *returned* a
  non-zero code, rather than raising, emitted `"the command exited before
  producing a result"`; the message it printed to stderr now travels in the
  payload's `error` field.
- **MCP `optimizer=` keeps the rest of the mandate.** It replaced the whole
  optimizer block, so a max-Sharpe override solved against a risk-free rate
  of zero on a config that said 4%, and dropped the return target with it.
- **MCP `optimize` reports an infeasible mandate as a `ToolError`.** The
  branch that wrapped the feasibility report was unreachable, because the
  engine's default lets the solver fail instead; the client saw a wrapped,
  message-less exception. Solver failures are wrapped the same way.
- **MCP `backtest` is shaped by its config.** The spec is now built on the
  config's `periods_per_year` (a monthly config was simulated as daily),
  trades only when the process re-solves (`frequency="none"`, matching the
  CLI and `EngineRun.walk_forward_run`), measures Sharpe against the
  config's risk-free rate, and defaults `lookback`/`rebalance_every` from the
  basis rather than from hard-coded daily counts.
- **The ingest cache refuses what it must not keep.** A panel with a failed
  or missing identifier was cached and served for the TTL as "the provider
  returned no series", never retried; a panel whose currency conversion had
  fallen back to native quotes was cached under a fingerprint claiming the
  base currency. Neither is stored now, and the run says `Not cached:` with
  the reason. The cache key also covers the provider options, so the `file`
  provider no longer serves file A's panel for file B.
- **FX alignment no longer back-fills from the future.** A leading gap in
  the rate history — prices starting before the first known rate — was
  filled with a *later* rate without limit. It is now filled for at most
  `MAX_LEADING_FX_GAP` (5) rows, the case of a request that starts on a
  holiday, and refused with an `FXError` beyond that; `fill="bfill"` asks for
  it explicitly. `fill="bfill"` also works on pandas 3, where the
  `fillna(method=)` it used to call no longer exists.
- **A config with a key the loader does not read is refused.** `max_tracking_eror:
  0.03` loaded cleanly and constrained nothing; `EngineConfig.from_dict` now
  raises `ConfigurationError` naming the unknown key, and an unknown optimizer
  key raises the same rather than a bare `TypeError`. The shipped
  `config/indices.yaml` spelled its currency block `asset_currency`, which
  nothing read; it is `currencies` now, so its non-USD indices are converted.

## [0.5.2] — 2026-09-01

Documentation only. No behaviour changed, no signature moved, and the test
suite is untouched — but the reference the library never had now exists, and
CI fails if it stops building.

### Added

- **A generated API reference.** `scripts/build_api_docs.py` renders every
  public module with pdoc and a new `docs` extra installs it. A `Docs`
  workflow builds it on every pull request and publishes it to GitHub Pages
  from `main`. The build runs `--strict`, which refuses to skip a module it
  cannot import: a missing extra now fails the job rather than quietly
  dropping `mcp_server` from the published pages. Modules are discovered by
  walking the package, so a new one is documented the day it lands.
- **`docs/ERRORS.md`** — the refusal contract, written down. All twenty-one
  exception types grouped by what they mean (your inputs are wrong; your
  mandate is impossible; the world got in the way), the full `IngestError`
  hierarchy with what the service does about each subclass, the CLI's exit
  codes and the one case where `--json` changes them, and a closing section on
  the failures that are *reported* rather than raised — degraded cost models,
  skipped identifiers, post-solve constraint breaches, data-quality findings.
  Linked from the README, `AGENTS.md` and `llms.txt`.

### Changed

- **Docstring coverage went from 74.6% to 100%** across the 677 public
  definitions: 166 that had none now do. The parameter documentation moved
  further — 86 functions documented their arguments before, 379 do now — and
  it carries units where a number has any: `cost_bps` per side, `execution_lag`
  in bars, `alpha` as a tail probability rather than a confidence level, VaR
  levels in percent rather than as fractions.
- **Every public function that raises now declares what and when.** There were
  364 `raise` statements behind 67 `Raises:` sections; the gap is closed. The
  same for returns: what a function hands back, in what units, and what it
  does with the values it could not compute.
- `AGENTS.md` states that docstrings are the API reference's only source, and
  therefore part of the public surface rather than a courtesy.


## [0.5.1] — 2026-08-31

A patch: one bug, in the part of `--json` that only a failing run reaches.
Nothing an optimizer, estimator or backtest produces is different.

### Fixed

- `--json` emitted nothing at all when a command raised. The mode promised
  that a failure still produces a parseable document, and delivered it only
  for a command that *returned* a non-zero code — a raised exception, an
  unreadable config being the cheapest way in, propagated instead: traceback
  on stderr, stdout empty. That is precisely the "no output versus output I
  could not parse" ambiguity the flag exists to remove, and it was reachable
  from all four commands. The payload now survives the exception and carries
  the exception's type and message; the traceback still goes to stderr, where
  someone debugging it looks.

### Changed

- The README announces the agent-facing surface instead of leaving it in the
  changelog: the MCP server has its own section with the install, the
  registration for a JSON-config client and for `claude mcp add`, and what
  each of the five tools answers; `--json` is documented in the CLI section
  with real `jq` output rather than a sketch of it. The development install
  gains the `mcp` extra it needed to run the MCP tests.

## [0.5.0] — 2026-08-31

A minor rather than a patch: this adds public API surface — an optional
dependency, a second console script, and a function lifted out of
`run_engine` — on top of a fix to a pre-flight check that was validating a
different mandate from the one it preceded.

### Fixed

- `check` validated a different mandate from the one `optimize` solved. On a
  config with no `expected_returns` block it derived the vector as zeros
  while `run_engine` derived it from the return history, so the pre-flight
  reported a reachable return range of exactly zero to zero and would call a
  target unreachable that the solve then reached. Both now go through
  `resolve_expected_returns`, extracted from `run_engine` so the two cannot
  drift apart again. Affects `optengine check` and the MCP `check_mandate`.

### Added

- CI runs the MCP suite. The `mcp` extra is in no other job's install — it
  cannot go in the 3.9-inclusive matrix — so those tests would have
  skipped everywhere and looked like coverage. The Streamlit job now
  covers both optional-extra surfaces and is named for it, and asserts
  the `optengine-mcp` console script lands on PATH.
- An MCP server, `optengine-mcp`, behind the `mcp` extra (Python 3.10+).
  Five tools — `list_optimizers`, `describe_optimizer`, `check_mandate`,
  `optimize`, `backtest` — returning the same payloads as `--json`, from the
  same module, so the two cannot disagree. Anticipated failures raise the
  SDK's `ToolError`, which is the only class whose message reaches the
  client; anything else is wrapped as "Error executing tool optimize" with
  the reason discarded.

- `--json` on `optimize`, `backtest`, `check` and `describe`. Human
  narration moves to stderr, so stdout is one parseable document and an
  agent or pipeline can act on a result without scraping a formatted table.
  Every payload carries `schema_version`, and a command that fails before
  producing a result still emits JSON — an object with `error` and
  `exit_code` — so a caller never has to distinguish "no output" from
  "output I could not parse".
- `optimization_engine.reporting.payloads`, the JSON contract as an
  importable module rather than formatting buried in the CLI. Keys are
  chosen for the reader instead of inherited from the attributes they came
  from, and a value absent this run is `null` rather than a missing key, so
  a consumer can test a value and never a key.
- `AGENTS.md` and `llms.txt` — an API map for coding agents, including the
  mistakes that cost real debugging time here: `sample_dataset()` returns
  prices rather than returns, the solver is in `result.extras`, the
  backtest hashes are on `result.meta`, and unconstrained `max_sharpe` puts
  98.5% in cash on the sample panel.

## [0.4.1] — 2026-08-31

A correctness and packaging release. Two of the three entries below only
reach you if you had `riskfolio-lib` installed or read the project page on
PyPI; the third is the reason the first two are worth a release at all.

### Fixed

- `covariance_method="shrink"` no longer means different mathematics on
  different machines. It routed through `riskfolio-lib` when that package
  happened to be installed and fell back to scikit-learn's Ledoit-Wolf when
  it did not — a silent fork, since `CovarianceDiagnostics` recorded no such
  thing, so the same config on the same data produced different numbers with
  nothing in the output saying which estimator had run. On the sample panel
  the two differ by 8.3% of the largest element, which is an estimator
  change, not a rounding difference. Worse, the fallback caught bare
  `Exception`, so any riskfolio error or API change swapped the estimator
  silently too.

  `shrink` is now a documented alias for `ledoit_wolf`, so configs and saved
  scenarios written against it keep loading and now reproduce anywhere. The
  `extras` optional dependency, whose only purpose was this route, is gone.

- The README renders on PyPI. Every image and repository link was a relative
  path, which resolves against the repo on GitHub and against nothing on
  PyPI — all nine figures were broken on the project page, along with the
  links to the licence and the research notes. They are absolute now.

### Changed

- The README leads with the install command and a runnable quickstart
  instead of burying `pip install` 600 lines down. The example's printed
  output is checked against what the code in that same block actually
  produces, so it cannot drift into fiction.

## [0.4.0] — 2026-08-31

The release that makes the project installable from PyPI. No optimizer,
estimator or backtest changed behaviour: every number this version produces
matches 0.3.0.

### Changed

- **The distribution is now named `finport-optengine`.** `optimization-engine`
  is taken on PyPI by an unrelated energy-forecasting package. The import
  name (`optimization_engine`) and the console script (`optengine`) are
  unchanged, so only the install command moves:
  `pip install finport-optengine`.
- **The core install is smaller by roughly 190MB.** `plotly`, `openpyxl`,
  `xlsxwriter` and `statsmodels` are no longer mandatory dependencies; they
  now sit behind the `viz`, `excel` and `stats` extras. Anything the core
  can already do — every optimizer, the covariance estimators, the backtest,
  the analytics — still works on a bare `pip install finport-optengine`.

  Reaching a feature whose extra is absent now raises
  `optimization_engine._optional.MissingDependencyError` (a subclass of
  `ImportError`) naming the install command, instead of a
  `ModuleNotFoundError` for an import the caller never wrote.

  To keep the previous all-inclusive install, use
  `pip install "finport-optengine[all]"`.

  `scikit-learn` deliberately stays in the core: it backs the *default*
  covariance estimator, so demoting it would either break the first call in
  the README on a fresh install or silently change the default to the
  sample covariance.

### Added

- A release workflow (`.github/workflows/release.yml`) publishing through
  Trusted Publishing (OIDC), so no API token is stored in the repository.
  A manual run publishes a fresh `.devN` to TestPyPI and then installs it
  back *from* TestPyPI to prove the artifact is reachable; pushing a `v*`
  tag publishes to PyPI behind a required-reviewer gate. The build refuses
  to proceed if `pyproject.toml` and `__init__.py` disagree on the version,
  or if a tag does not match the version it claims.
- `docs/RELEASING.md` — the one-time Trusted Publishing setup each index
  needs, the release procedure, and what to do about a bad release.

- `optimization_engine.ingest` is reachable as an attribute of the top-level
  package, and its four most-used names — `IngestRequest`, `IngestResult`,
  `IngestError`, `PricePanel` — are re-exported at the top level. The
  subpackage's full 44-name API is unchanged and stays where it was; the
  generic field constants (`CLOSE`, `OPEN`, `VOLUME`, …) are deliberately
  *not* raised to the top level.
- `py.typed`, so type checkers actually consume the project's annotations.
  The marker had been declared in `pyproject.toml` since the package was
  laid out, but the file itself was never added — meaning no downstream
  user has ever received the types.
- `[project.urls]`, so the PyPI page links to the repository, the issue
  tracker and this file.
- An `all` extra, covering what the CLI and the README's worked examples
  assume.
- This changelog.

### Removed

- `matplotlib` and `seaborn` as dependencies. Nothing in the project has
  ever imported either one — the plotting is entirely Plotly. Removing them
  changes no behaviour.
- `requirements.txt`. It listed `streamlit`, `yfinance`, `riskfolio-lib` and
  `ipywidgets` as mandatory, contradicting `pyproject.toml`, which has them
  as extras. `pyproject.toml` is now the single source of truth; install the
  development set with `pip install -e ".[all,ui,extras,dev]"`.

### Fixed

- The release workflow could never publish. `pypa/gh-action-pypi-publish`
  was referenced by commit SHA — the usual supply-chain advice, and wrong
  for this action: it is a Docker action, and the runner pulls
  `ghcr.io/pypa/gh-action-pypi-publish` tagged with whatever ref the `uses:`
  line carries. PyPA publishes that image only under release tags, so the
  SHA resolved to no manifest and the step died with `manifest unknown`
  before reaching the index. Now referenced as `@v1.14.2`, a tag confirmed
  to exist in the registry, with the reasoning recorded in
  `docs/RELEASING.md` so nobody "hardens" it back.

- Excluded `scs` 3.3.0. Its wheel ships an incomplete Intel oneMKL bundle,
  so importing it and solving aborts the interpreter — "Cannot load
  libmkl_avx512.so.3 or libmkl_def.so.3" — rather than raising. SCS sits in
  `SOLVER_FALLBACK`, so any solve that got past CLARABEL and ECOS would take
  the process down with it, with no traceback and nothing for the fallback
  chain to catch. Reproduced on GitHub's runners and in a local container;
  3.2.11 is unaffected. `scs` is now named directly in the dependencies,
  since the engine reaches for it by name rather than leaving the choice to
  cvxpy. Remove the exclusion once a fixed release ships.

## [0.3.0] and earlier

Released before this changelog was kept. The repository's commit history is
the record; the headline work was the data-ingestion spine (one panel, many
providers, with per-identifier provenance), the stateless backtest core with
its cost model and trial counting, the walk-forward and final-holdout audit
path, and the constraint-layer editor in the Streamlit app.

[Unreleased]: https://github.com/alanvaa06/optimization_engine/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/alanvaa06/optimization_engine/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/alanvaa06/optimization_engine/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/alanvaa06/optimization_engine/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/alanvaa06/optimization_engine/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/alanvaa06/optimization_engine/releases/tag/v0.4.0
