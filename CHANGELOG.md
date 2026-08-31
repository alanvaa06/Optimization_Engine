# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, the public API may change in a minor
release. Every such change is listed here under **Changed** or **Removed**,
with what to do about it.

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/alanvaa06/optimization_engine/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/alanvaa06/optimization_engine/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/alanvaa06/optimization_engine/releases/tag/v0.4.0
