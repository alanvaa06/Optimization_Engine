# Optimization Engine: UX consistency, speed, and tests — Design

**Status:** Drafted 2026-04-27. Approach B from brainstorming.

**Goal:** Make the Streamlit UI input surface consistent across all 10 optimizers,
fix correctness gaps in the soft-bounds methods, surface the optimizer knobs
that are currently hidden, improve solve speed for the frontier and per-rerun
work, and back the lot with tests.

## Decisions captured from brainstorming

1. Irrelevant inputs are **disabled + tooltip-explained**, not hidden.
2. For post-projection optimizers, fix the correctness issue (iterated
   clip-and-rescale + a constrained reformulation for risk_parity).
3. Speed work targets all of: cov caching, parallel frontier, and the cheap
   per-rerun recomputations.
4. Test scope = smoke gaps + UI-logic helpers (no Streamlit `AppTest`).
5. Approach B = "minimal-surgery refactor + close input gaps".

## 1. Method-requirements registry

New module: `src/optimization_engine/optimizers/requirements.py`.

Single source of truth for *what each optimizer takes and supports*. Both the
factory (validation) and the Streamlit UI (which fields to enable) read it.

```python
@dataclass(frozen=True)
class ExtraInput:
    key: str
    label: str
    kind: Literal["per_asset", "scalar", "choice", "view_table", "market_caps"]
    required: bool
    help: str
    default: Any | None = None
    choices: tuple[str, ...] | None = None

@dataclass(frozen=True)
class MethodRequirements:
    name: str
    requires_mu: bool
    requires_cov: bool
    requires_returns: bool
    supports_target_return: bool
    supports_target_volatility: bool
    supports_risk_aversion: bool
    supports_risk_free_rate: bool
    supports_group_bounds: bool
    bounds_mode: Literal["hard", "soft_iterated", "constrained"]
    supports_frontier: bool
    extras: tuple[ExtraInput, ...]

REQUIREMENTS: dict[str, MethodRequirements] = {
    "mean_variance": MethodRequirements(
        name="mean_variance",
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, extras=(),
    ),
    # ... 9 more entries, one per registered optimizer ...
}

def requirements_for(name: str) -> MethodRequirements:
    try:
        return REQUIREMENTS[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown optimizer '{name}'. "
            f"Available: {sorted(REQUIREMENTS.keys())}"
        ) from e
```

### Per-method matrix

| Method | μ | Σ | r | TR | TV | λ | rf | grp | bounds | frontier | extras |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `mean_variance` | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | hard | ✓ | — |
| `min_variance` | — | ✓ | — | — | — | — | — | ✓ | hard | — | — |
| `max_sharpe` | ✓ | ✓ | — | — | — | — | ✓ | ✓ | hard | — | — |
| `risk_parity` | — | ✓ | — | — | — | — | — | — | constrained | — | risk_budget |
| `hrp` | — | ✓ | — | — | — | — | — | — | soft_iterated | — | linkage_method |
| `black_litterman` | prior | ✓ | — | ✓ | ✓ | ✓ | ✓ | ✓ | hard | ✓ | views, view_confidences, tau, market_caps |
| `cvar` | optional | — | ✓ | ✓ | — | — | ✓ | ✓ | hard | — | alpha |
| `max_diversification` | — | ✓ | — | — | — | — | — | — | soft_iterated | — | — |
| `equal_weight` | — | — | — | — | — | — | — | — | soft_iterated | — | — |
| `inverse_vol` | — | diag | — | — | — | — | — | — | soft_iterated | — | — |

### Validation

`optimizer_factory` calls `requirements_for(spec.name)` and raises
`ConfigurationError` (new exception in `optimizers/__init__.py`) when:

- `requires_mu` and `expected_returns` is empty.
- `requires_cov` and `cov_matrix` is None.
- `requires_returns` and `returns` is None.

If `supports_target_return` is False but a `target_return` is set on the
spec, the factory does **not** raise — it logs a `logging.warning` and
ignores the field. (The same applies to `target_volatility`,
`risk_aversion`, and `risk_budget` mismatches.) Rationale: scenario YAML
written for one method is often loaded into another via the UI, and a hard
raise would force the user to re-enter values they may want to keep.

## 2. UI changes per tab

A pure helper `derive_widget_state(method_name: str) -> dict[str, dict]`
lives in `src/optimization_engine/ui_state.py` (extending the existing
module) and returns
`{widget_key: {"enabled": bool, "tooltip": str | None}}` so the Streamlit
code becomes mostly:

```python
ws = derive_widget_state(optimizer_name)
st.number_input(..., disabled=not ws["risk_free_rate"]["enabled"],
                help=ws["risk_free_rate"]["tooltip"])
```

### Sidebar — "3 · Optimizer"

- Method selectbox: unchanged.
- `risk_free_rate`: enabled iff `req.supports_risk_free_rate`.
- `cov_method`, `ewma_lambda`: enabled iff `req.requires_cov`.
- Mode (target_return / target_vol / utility): visible only when
  `req.supports_target_return` or `req.supports_target_volatility`.
- CVaR α: visible only for cvar.
- Frontier checkbox: enabled iff `req.supports_frontier`.

### Constraints tab

- "Expected Return" column: enabled iff `req.requires_mu`.
- New radio "Expected-returns method": `historical_mean | ema | capm`.
  Visible only when `requires_mu`. Drives a "Reset μ to method default"
  button; the table cell remains the authoritative value when the user
  edits it.
- For `ema`: a span slider (30-504, default 180).
- For `capm`: a `market_return` (optional) and `market_weights` table.
- "Group Constraints" block: visible iff `req.supports_group_bounds`.
- A "Soft bounds (post-projection)" caption above the bounds table when
  `bounds_mode != "hard"`.
- Risk-budget table: visible only for risk_parity (registry-driven).
- BL views table: visible only for BL.
- New BL inputs: `tau` slider [0.01-0.5], `market_caps` per-asset table.
- New HRP input: `linkage_method` selectbox (single | average | complete | ward).

### What-if tab

The "Optimizer extras" block becomes registry-driven: for each `ExtraInput`
in `req.extras`, render the appropriate widget kind (per_asset → data_editor,
scalar → slider/number_input, choice → selectbox).

### Compare tab

Unchanged.

### Optimize tab

The "Efficient Frontier" subsection only renders when `req.supports_frontier`.

## 3. Bounds correctness

### `project_to_bounds_iterated`

New module: `src/optimization_engine/optimizers/_bounds.py`.

```python
class InfeasibleBoundsError(ValueError):
    pass

def project_to_bounds_iterated(
    w: np.ndarray, lb: np.ndarray, ub: np.ndarray,
    max_iter: int = 50, atol: float = 1e-8,
) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    if not (lb <= ub).all():
        raise ValueError("lb must be <= ub element-wise")
    if lb.sum() > 1.0 + atol or ub.sum() < 1.0 - atol:
        raise InfeasibleBoundsError("sum(lb) > 1 or sum(ub) < 1")
    for _ in range(max_iter):
        w = np.clip(w, lb, ub)
        s = w.sum()
        if abs(s - 1.0) < atol:
            return w
        residual = 1.0 - s
        if residual > 0:
            slack = ub - w
            total = float(slack.sum())
            if total <= atol:
                raise InfeasibleBoundsError("no upper slack to absorb residual")
            w = w + slack * (residual / total)
        else:
            slack = w - lb
            total = float(slack.sum())
            if total <= atol:
                raise InfeasibleBoundsError("no lower slack to absorb deficit")
            w = w - slack * (-residual / total)
    raise RuntimeError("project_to_bounds_iterated did not converge")
```

This replaces the one-shot `project_to_bounds` for HRP, max_diversification,
equal_weight, inverse_vol. The original `project_to_bounds` stays available
for callers that need the simple semantics (none in the engine after this
change — it can be removed at the end).

### Constrained risk-parity

`risk_parity._solve` adds bound constraints in `y`-space:

```python
y = cp.Variable(n, pos=True)
sigma_psd = cp.psd_wrap(sigma)
total = cp.sum(y)
cons = [
    y >= cp.multiply(lb_arr, total),
    y <= cp.multiply(ub_arr, total),
]
# group bounds, when present:
for grp, idx in group_indices.items():
    g_lb, g_ub = constraints.group_bounds[grp]
    cons.append(cp.sum(y[idx]) >= g_lb * total)
    cons.append(cp.sum(y[idx]) <= g_ub * total)
problem = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(y, sigma_psd) - b @ cp.log(y)),
    cons,
)
```

`bounds_mode` for risk_parity becomes `"constrained"` and
`supports_group_bounds=True`.

## 4. Speed

### Covariance cache (Streamlit-side)

Helpers in `app/streamlit_app.py`. Caching is a UI concern; the engine module
stays pure.

```python
@st.cache_data(show_spinner=False, max_entries=16)
def _returns_hash(returns: pd.DataFrame) -> str:
    return pd.util.hash_pandas_object(returns, index=True).values.tobytes().hex()

@st.cache_data(show_spinner=False, max_entries=16)
def _covariance_cached(
    returns_hash: str,
    method: str,
    ewma_lambda: float,
    periods_per_year: int,
    annualize: bool,
    _returns: pd.DataFrame,
) -> pd.DataFrame:
    return covariance_matrix(
        _returns, method=method, ewma_lambda=ewma_lambda,
        periods_per_year=periods_per_year, annualize=annualize,
    )
```

The leading underscore on `_returns` tells Streamlit to skip hashing the
DataFrame on every call — `returns_hash` is the cache key, computed once
per unique frame.

### Frontier parallelization

`frontier.efficient_frontier`:

```python
from concurrent.futures import ThreadPoolExecutor

def _solve_one(target, base_config, sweep, cov, mu, returns):
    cfg = copy.deepcopy(base_config)
    if sweep == "return":
        cfg.optimizer.target_return = float(target)
        cfg.optimizer.risk_aversion = 1.0
    else:
        cfg.optimizer.target_return = None
        cfg.optimizer.risk_aversion = float(target)
    try:
        result = optimizer_factory(
            cfg, cov, expected_returns=mu, returns=returns,
        ).optimize()
        return target, result, None
    except Exception as exc:  # infeasible target
        return target, None, str(exc)

def efficient_frontier(..., n_workers: int | None = None):
    ...
    workers = n_workers if n_workers is not None else min(8, len(target_returns))
    if workers <= 1:
        rows = [
            _solve_one(t, base_config, sweep, cov_matrix, expected_returns, returns)
            for t in target_returns
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(
                _solve_one,
                target_returns,
                [base_config] * len(target_returns),
                [sweep] * len(target_returns),
                [cov_matrix] * len(target_returns),
                [expected_returns] * len(target_returns),
                [returns] * len(target_returns),
            ))
    rows.sort(key=lambda r: r[0])  # ex.map preserves order; defensive
```

CVXPY/CLARABEL/ECOS spend most time in C extensions which release the GIL,
so threads (not processes) are appropriate. `n_workers=None` defaults to
`min(8, n_points)` — small sweeps stay sequential; large sweeps cap at 8.

### Memoize `historical_mu`

Constraints tab renders `historical_mu` every Streamlit run. Replace the
inline computation with:

```python
@st.cache_data(show_spinner=False, max_entries=8)
def _historical_mu_cached(
    returns_hash: str,
    periods_per_year: int,
    _returns: pd.DataFrame,
) -> pd.Series:
    return (1 + _returns).prod() ** (periods_per_year / len(_returns)) - 1
```

Keyed on `(returns_hash, periods_per_year)`. Cheap, but avoids repeated
work on every slider drag.

## 5. New inputs surfaced

### `EngineConfig` additions

```python
@dataclass
class EngineConfig:
    ...
    expected_returns_method: Literal["historical_mean", "ema", "capm"] = "historical_mean"
    ema_span: int = 180
    market_return: float | None = None
    market_weights: dict[str, float] | None = None
```

`run_engine` uses these only when `config.expected_returns` is empty —
once the user puts numbers in the table, those are authoritative:

```python
if expected_returns is None and config.expected_returns:
    expected_returns = pd.Series(config.expected_returns)
if expected_returns is None:
    expected_returns = expected_returns_from_history(
        returns,
        method=config.expected_returns_method,
        periods_per_year=config.periods_per_year,
        span=config.ema_span,
        market_return=config.market_return,
        risk_free_rate=config.optimizer.risk_free_rate,
        market_weights=(
            pd.Series(config.market_weights)
            if config.market_weights else None
        ),
        cov_matrix=cov,
    )
```

### `OptimizerSpec` additions

```python
hrp_linkage: Literal["single", "average", "complete", "ward"] = "single"
```

`bl_tau` and `bl_market_caps` already exist; just need UI surfaces.

The factory passes `linkage_method=spec.hrp_linkage` to `HRPOptimizer`.

## 6. Testing

### New test files

`tests/test_requirements.py`:

- Every key in `optimizer_factory._REGISTRY` has a `MethodRequirements`
  entry.
- `requires_*` flags align with the matrix above (table-driven test).
- `requirements_for("nope")` raises `KeyError`.

`tests/test_bounds_projection.py`:

- Already-feasible input is unchanged.
- Single clip+rescale converges within 2 iterations on typical inputs.
- `sum(lb) > 1` raises `InfeasibleBoundsError`.
- `sum(ub) < 1` raises `InfeasibleBoundsError`.
- Output lies inside `[lb, ub]` and sums to 1 within `atol`.

`tests/test_ui_logic.py`:

- For each optimizer, `derive_widget_state` returns the expected
  enabled/disabled flags. (Table-driven.)
- Tooltips populated whenever `enabled=False`.

`tests/test_frontier_parallel.py`:

- `efficient_frontier` with `n_workers=1` and `n_workers=4` produce
  numerically identical summaries.
- Sweep-by-`risk_aversion` is monotone in λ for portfolio variance.
- Infeasible targets surface as `status="failed: ..."` rows, not raises.

### Extended `tests/test_optimizers.py`

- `test_hrp_linkage_methods` — single, average, complete, ward all run and
  return weights summing to 1.
- `test_max_diversification_respects_bounds` — bounded run.
- `test_cvar_with_target_return` — meets target.
- `test_black_litterman_no_views` — equals MV with implied equilibrium μ.
- `test_group_bounds_enforced_for_hard_methods` — mean_variance, min_variance,
  max_sharpe, cvar, BL.
- `test_infeasible_target_raises_clearly` — clear error message.
- `test_constrained_risk_parity_respects_bounds` — bounded ERC.

Roughly 20 new test cases.

## 7. Migration / compatibility

- Existing YAML configs continue to load. New `OptimizerSpec` and
  `EngineConfig` fields default to today's behavior.
- Public Python API surface is purely additive (new module
  `requirements.py`, new helper `derive_widget_state`).
- Saved scenarios (YAML) round-trip — new fields appear when present;
  unknown fields ignored by `from_dict`.
- The original `project_to_bounds` helper is kept (the iterated version is
  added next to it). After all callers migrate, the unused helper is
  removed in the same PR.

## File-level changes summary

**New:**

- `src/optimization_engine/optimizers/requirements.py`
- `src/optimization_engine/optimizers/_bounds.py`
- `src/optimization_engine/ui_state.py` (extend existing module with
  `derive_widget_state`; do not split into a new file)
- `tests/test_requirements.py`
- `tests/test_bounds_projection.py`
- `tests/test_ui_logic.py`
- `tests/test_frontier_parallel.py`

**Modified:**

- `src/optimization_engine/optimizers/factory.py`
- `src/optimization_engine/optimizers/_cvxpy_helpers.py`
- `src/optimization_engine/optimizers/risk_parity.py`
- `src/optimization_engine/optimizers/hrp.py`
- `src/optimization_engine/optimizers/max_diversification.py`
- `src/optimization_engine/optimizers/naive.py`
- `src/optimization_engine/optimizers/__init__.py` (export
  `ConfigurationError`, `requirements_for`, `MethodRequirements`)
- `src/optimization_engine/config.py`
- `src/optimization_engine/engine.py`
- `src/optimization_engine/frontier.py`
- `app/streamlit_app.py`
- `tests/test_optimizers.py`

## Out of scope

- Streamlit `AppTest` end-to-end (deferred to a future spec if needed).
- Process-pool parallelism for the frontier (threads are sufficient given
  CVXPY releases the GIL).
- Custom HRP-with-bounds (the iterated projection is good enough; a true
  constrained HRP is a research project).
- A redesigned tab layout (Approach C).
