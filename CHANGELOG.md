# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, the public API may change in a minor
release. Every such change is listed here under **Changed** or **Removed**,
with what to do about it.

## [Unreleased]

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

## [0.4.0] — 2026-08-30

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

## [0.3.0] and earlier

Released before this changelog was kept. The repository's commit history is
the record; the headline work was the data-ingestion spine (one panel, many
providers, with per-identifier provenance), the stateless backtest core with
its cost model and trial counting, the walk-forward and final-holdout audit
path, and the constraint-layer editor in the Streamlit app.

[Unreleased]: https://github.com/alanvaa06/optimization_engine/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/alanvaa06/optimization_engine/releases/tag/v0.4.0
