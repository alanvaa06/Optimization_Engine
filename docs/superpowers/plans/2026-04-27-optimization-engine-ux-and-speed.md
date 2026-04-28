# Optimization Engine UX, Bounds, Speed & Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Streamlit UI consistent across all 10 optimizers, fix correctness gaps in soft-bounds methods, surface hidden optimizer knobs, parallelize the frontier, cache covariance, and add tests for the previously untested paths.

**Architecture:** A new `MethodRequirements` registry is the single source of truth for "what each optimizer takes / supports." The factory uses it for validation; a new pure helper `derive_widget_state` lets Streamlit consume it without coupling to UI specifics. An iterated bounds-projection routine replaces the silently-incorrect one-shot version for soft-bounds methods; risk-parity is reformulated with native CVXPY constraints. Frontier solves run on a thread pool. Streamlit caches covariance + `historical_mu` per `(returns-hash, params)`.

**Tech Stack:** Python 3.10+, CVXPY (CLARABEL / ECOS / SCS solvers), NumPy, pandas, scipy.cluster, Streamlit, pytest.

**Spec:** [`docs/superpowers/specs/2026-04-27-optimization-engine-ux-and-speed-design.md`](../specs/2026-04-27-optimization-engine-ux-and-speed-design.md)

---

## Setup

The plan assumes you are on `main` with no other in-flight changes (the pre-existing untracked `src/optimization_engine/ui_state.py` and `tests/test_ui_state.py` from prior work are kept; this plan extends them). Confirm pytest runs green before starting:

```bash
pytest -q
```

If anything fails on `main`, stop and report — that's not part of this plan's scope.

---

## Task 1: Iterated bounds projection module

**Files:**
- Create: `src/optimization_engine/optimizers/_bounds.py`
- Create: `tests/test_bounds_projection.py`

- [ ] **Step 1: Write the failing test file**

`tests/test_bounds_projection.py`:

```python
"""Tests for project_to_bounds_iterated."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.optimizers._bounds import (
    InfeasibleBoundsError,
    project_to_bounds_iterated,
)


def test_already_feasible_input_unchanged():
    w = np.array([0.3, 0.3, 0.4])
    out = project_to_bounds_iterated(w, np.zeros(3), np.ones(3))
    np.testing.assert_allclose(out, w, atol=1e-10)


def test_residual_distributed_over_slack():
    w = np.array([0.5, 0.5, 0.5])  # sums to 1.5, must come down to 1.0
    out = project_to_bounds_iterated(w, np.zeros(3), np.full(3, 0.6))
    assert pytest.approx(out.sum(), abs=1e-8) == 1.0
    assert (out >= -1e-9).all() and (out <= 0.6 + 1e-9).all()


def test_clip_then_rescale_breaks_bounds_but_iterated_does_not():
    # A naive clip(0,0.4) + rescale would push the first weight back over 0.4.
    w = np.array([0.9, 0.05, 0.05])
    lb = np.zeros(3)
    ub = np.array([0.4, 1.0, 1.0])
    out = project_to_bounds_iterated(w, lb, ub)
    assert (out <= ub + 1e-9).all()
    assert (out >= lb - 1e-9).all()
    assert pytest.approx(out.sum(), abs=1e-8) == 1.0


def test_infeasible_lb_sum_raises():
    with pytest.raises(InfeasibleBoundsError):
        project_to_bounds_iterated(
            np.array([0.5, 0.5, 0.5]),
            np.full(3, 0.5),
            np.ones(3),
        )


def test_infeasible_ub_sum_raises():
    with pytest.raises(InfeasibleBoundsError):
        project_to_bounds_iterated(
            np.array([0.1, 0.1, 0.1]),
            np.zeros(3),
            np.full(3, 0.2),
        )


def test_lb_greater_than_ub_raises_valueerror():
    with pytest.raises(ValueError, match="lb must be"):
        project_to_bounds_iterated(
            np.array([0.5, 0.5]),
            np.array([0.6, 0.0]),
            np.array([0.4, 1.0]),
        )


def test_negative_residual_distributed_over_slack_above_lb():
    # w sums to 0.5 (deficit), pushed up via lower-side slack
    w = np.array([0.1, 0.1, 0.3])
    lb = np.array([0.05, 0.05, 0.05])
    ub = np.full(3, 1.0)
    out = project_to_bounds_iterated(w, lb, ub)
    assert pytest.approx(out.sum(), abs=1e-8) == 1.0
    assert (out >= lb - 1e-9).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_bounds_projection.py -v
```

Expected: ImportError — `optimization_engine.optimizers._bounds` does not exist.

- [ ] **Step 3: Create the module**

`src/optimization_engine/optimizers/_bounds.py`:

```python
"""Iterated bounds projection.

Replaces the one-shot ``project_to_bounds`` for optimizers that solve
unconstrained then need their weights forced into per-asset bounds while
still summing to one. Iterates clip + redistribute-residual until both
invariants are satisfied or until ``max_iter`` is reached.
"""

from __future__ import annotations

import numpy as np


class InfeasibleBoundsError(ValueError):
    """The bounds and the unit-budget are mutually infeasible."""


def project_to_bounds_iterated(
    w: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_iter: int = 50,
    atol: float = 1e-8,
) -> np.ndarray:
    """Project ``w`` onto ``{w : lb <= w <= ub, sum(w) = 1}``.

    Raises :class:`InfeasibleBoundsError` when ``sum(lb) > 1`` or
    ``sum(ub) < 1``.
    """
    w = np.asarray(w, dtype=float).copy()
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if not (lb <= ub).all():
        raise ValueError("lb must be <= ub element-wise")
    if lb.sum() > 1.0 + atol or ub.sum() < 1.0 - atol:
        raise InfeasibleBoundsError(
            f"sum(lb)={lb.sum():.6f}, sum(ub)={ub.sum():.6f} cannot meet sum=1"
        )
    for _ in range(max_iter):
        w = np.clip(w, lb, ub)
        s = float(w.sum())
        if abs(s - 1.0) < atol:
            return w
        residual = 1.0 - s
        if residual > 0:
            slack = ub - w
            total = float(slack.sum())
            if total <= atol:
                raise InfeasibleBoundsError(
                    "no upper-side slack to absorb residual"
                )
            w = w + slack * (residual / total)
        else:
            slack = w - lb
            total = float(slack.sum())
            if total <= atol:
                raise InfeasibleBoundsError(
                    "no lower-side slack to absorb deficit"
                )
            w = w - slack * (-residual / total)
    raise RuntimeError("project_to_bounds_iterated did not converge")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bounds_projection.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/optimization_engine/optimizers/_bounds.py tests/test_bounds_projection.py
git commit -m "feat: add iterated bounds projection helper"
```

---

## Task 2: Method-requirements registry

**Files:**
- Create: `src/optimization_engine/optimizers/requirements.py`
- Create: `tests/test_requirements.py`

- [ ] **Step 1: Write the failing test**

`tests/test_requirements.py`:

```python
"""Registry shape and per-method matrix tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.optimizers.factory import available_optimizers
from optimization_engine.optimizers.requirements import (
    REQUIREMENTS,
    MethodRequirements,
    requirements_for,
)


EXPECTED_FLAGS = {
    "mean_variance": dict(
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True,
    ),
    "min_variance": dict(
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False,
    ),
    "max_sharpe": dict(
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False,
    ),
    "risk_parity": dict(
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="constrained",
        supports_frontier=False,
    ),
    "hrp": dict(
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False,
    ),
    "black_litterman": dict(
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True,
    ),
    "cvar": dict(
        requires_mu=False, requires_cov=False, requires_returns=True,
        supports_target_return=True, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False,
    ),
    "max_diversification": dict(
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False,
    ),
    "equal_weight": dict(
        requires_mu=False, requires_cov=False, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False,
    ),
    "inverse_vol": dict(
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False,
    ),
}


def test_every_registered_optimizer_has_requirements():
    for name in available_optimizers():
        req = requirements_for(name)
        assert isinstance(req, MethodRequirements)
        assert req.name == name


@pytest.mark.parametrize("method,flags", list(EXPECTED_FLAGS.items()))
def test_requirements_match_matrix(method, flags):
    req = requirements_for(method)
    for k, v in flags.items():
        assert getattr(req, k) == v, f"{method}.{k}: expected {v}, got {getattr(req, k)}"


def test_requirements_for_unknown_raises():
    with pytest.raises(KeyError, match="Unknown optimizer"):
        requirements_for("not_a_method")


def test_extras_for_methods_with_specific_inputs():
    rp = requirements_for("risk_parity")
    keys = {e.key for e in rp.extras}
    assert "risk_budget" in keys

    bl = requirements_for("black_litterman")
    keys = {e.key for e in bl.extras}
    assert {"bl_views", "bl_view_confidences", "bl_tau", "bl_market_caps"} <= keys

    cvar = requirements_for("cvar")
    keys = {e.key for e in cvar.extras}
    assert "cvar_alpha" in keys

    hrp = requirements_for("hrp")
    keys = {e.key for e in hrp.extras}
    assert "hrp_linkage" in keys
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_requirements.py -v
```

Expected: ImportError — `optimization_engine.optimizers.requirements` does not exist.

- [ ] **Step 3: Create the module**

`src/optimization_engine/optimizers/requirements.py`:

```python
"""Per-method input/support metadata for the optimizer registry.

Single source of truth shared by the engine (validation) and the Streamlit
UI (which fields to enable). Adding a new optimizer means: register it in
``factory._REGISTRY`` AND add a ``MethodRequirements`` entry here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ExtraKind = Literal["per_asset", "scalar", "choice", "view_table", "market_caps"]
BoundsMode = Literal["hard", "soft_iterated", "constrained"]


@dataclass(frozen=True)
class ExtraInput:
    key: str
    label: str
    kind: ExtraKind
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
    bounds_mode: BoundsMode
    supports_frontier: bool
    extras: tuple[ExtraInput, ...] = field(default_factory=tuple)


_RISK_BUDGET = ExtraInput(
    key="risk_budget", label="Risk budget",
    kind="per_asset", required=False,
    help="Each asset's target share of total variance (sums to 1).",
)
_BL_VIEWS = ExtraInput(
    key="bl_views", label="Black-Litterman views",
    kind="view_table", required=False,
    help="Asset → annualized expected return.",
)
_BL_VIEW_CONFIDENCES = ExtraInput(
    key="bl_view_confidences", label="View confidences (Ω diagonal)",
    kind="view_table", required=False,
    help="Variance of each view's error term. Defaults to tau · σ_i².",
)
_BL_TAU = ExtraInput(
    key="bl_tau", label="Tau (prior uncertainty scale)",
    kind="scalar", required=False, default=0.05,
    help="Scales the prior covariance in the BL posterior.",
)
_BL_MARKET_CAPS = ExtraInput(
    key="bl_market_caps", label="Market caps / weights",
    kind="market_caps", required=False,
    help="Equilibrium market portfolio. Empty → equal weights.",
)
_CVAR_ALPHA = ExtraInput(
    key="cvar_alpha", label="CVaR tail probability α",
    kind="scalar", required=False, default=0.05,
    help="0.05 ⇒ 95% CVaR.",
)
_HRP_LINKAGE = ExtraInput(
    key="hrp_linkage", label="HRP linkage method",
    kind="choice", required=False, default="single",
    choices=("single", "average", "complete", "ward"),
    help="Hierarchical clustering linkage rule.",
)


REQUIREMENTS: dict[str, MethodRequirements] = {
    "mean_variance": MethodRequirements(
        name="mean_variance",
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True, extras=(),
    ),
    "min_variance": MethodRequirements(
        name="min_variance",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, extras=(),
    ),
    "max_sharpe": MethodRequirements(
        name="max_sharpe",
        requires_mu=True, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, extras=(),
    ),
    "risk_parity": MethodRequirements(
        name="risk_parity",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="constrained",
        supports_frontier=False, extras=(_RISK_BUDGET,),
    ),
    "hrp": MethodRequirements(
        name="hrp",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, extras=(_HRP_LINKAGE,),
    ),
    "black_litterman": MethodRequirements(
        name="black_litterman",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=True, supports_target_volatility=True,
        supports_risk_aversion=True, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=True,
        extras=(_BL_VIEWS, _BL_VIEW_CONFIDENCES, _BL_TAU, _BL_MARKET_CAPS),
    ),
    "cvar": MethodRequirements(
        name="cvar",
        requires_mu=False, requires_cov=False, requires_returns=True,
        supports_target_return=True, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=True,
        supports_group_bounds=True, bounds_mode="hard",
        supports_frontier=False, extras=(_CVAR_ALPHA,),
    ),
    "max_diversification": MethodRequirements(
        name="max_diversification",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, extras=(),
    ),
    "equal_weight": MethodRequirements(
        name="equal_weight",
        requires_mu=False, requires_cov=False, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, extras=(),
    ),
    "inverse_vol": MethodRequirements(
        name="inverse_vol",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=False, bounds_mode="soft_iterated",
        supports_frontier=False, extras=(),
    ),
}


def requirements_for(name: str) -> MethodRequirements:
    """Return the :class:`MethodRequirements` for an optimizer name.

    Raises ``KeyError`` with the list of known names when ``name`` is unknown.
    """
    try:
        return REQUIREMENTS[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown optimizer '{name}'. Available: {sorted(REQUIREMENTS.keys())}"
        ) from e
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_requirements.py -v
```

Expected: 13 PASS (1 + 10 parametrized + 1 + 1).

- [ ] **Step 5: Commit**

```bash
git add src/optimization_engine/optimizers/requirements.py tests/test_requirements.py
git commit -m "feat: add MethodRequirements registry for optimizers"
```

---

## Task 3: ConfigurationError + factory validation

**Files:**
- Modify: `src/optimization_engine/optimizers/__init__.py`
- Modify: `src/optimization_engine/optimizers/factory.py`
- Modify: `tests/test_optimizers.py` (add validation tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_optimizers.py`:

```python
from optimization_engine.optimizers import ConfigurationError


def test_factory_raises_when_required_mu_missing(returns):
    cfg = EngineConfig(
        expected_returns={},  # empty -> required for mean_variance
        bounds={a: [0.0, 1.0] for a in returns.columns},
        optimizer=OptimizerSpec(name="mean_variance"),
    )
    with pytest.raises(ConfigurationError, match="expected_returns"):
        run_engine(returns, cfg)


def test_factory_raises_when_returns_missing_for_cvar(returns):
    # CVaR needs the returns DataFrame; we exercise the factory directly
    # because the engine always supplies returns.
    from optimization_engine.data.covariance import covariance_matrix
    from optimization_engine.optimizers.factory import optimizer_factory

    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 1.0] for a in returns.columns},
        optimizer=OptimizerSpec(name="cvar"),
    )
    cov = covariance_matrix(returns, method="ledoit_wolf")
    with pytest.raises(ConfigurationError, match="returns"):
        optimizer_factory(cfg, cov, expected_returns=None, returns=None)


def test_factory_warns_on_incompatible_target_return(returns, baseline_config, caplog):
    import logging
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        optimizer=OptimizerSpec(name="hrp", target_return=0.05),
    )
    with caplog.at_level(logging.WARNING):
        run_engine(returns, cfg)
    assert any("target_return" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_optimizers.py -k "factory_raises or factory_warns" -v
```

Expected: ImportError or AttributeError on `ConfigurationError`.

- [ ] **Step 3: Add `ConfigurationError`**

Modify `src/optimization_engine/optimizers/__init__.py` — add `ConfigurationError` and the new public re-exports near the top:

```python
"""Portfolio optimizers."""


class ConfigurationError(ValueError):
    """The supplied EngineConfig is missing inputs the chosen optimizer requires."""


from optimization_engine.optimizers.base import (  # noqa: E402
    BaseOptimizer,
    OptimizationResult,
    PortfolioConstraints,
)
from optimization_engine.optimizers.black_litterman import BlackLittermanOptimizer  # noqa: E402
from optimization_engine.optimizers.cvar import CVaROptimizer  # noqa: E402
from optimization_engine.optimizers.factory import optimizer_factory  # noqa: E402
from optimization_engine.optimizers.hrp import HRPOptimizer  # noqa: E402
from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer  # noqa: E402
from optimization_engine.optimizers.mean_variance import (  # noqa: E402
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
)
from optimization_engine.optimizers.naive import (  # noqa: E402
    EqualWeightOptimizer,
    InverseVolatilityOptimizer,
)
from optimization_engine.optimizers.requirements import (  # noqa: E402
    MethodRequirements,
    requirements_for,
)
from optimization_engine.optimizers.risk_parity import RiskParityOptimizer  # noqa: E402

__all__ = [
    "BaseOptimizer",
    "BlackLittermanOptimizer",
    "ConfigurationError",
    "CVaROptimizer",
    "EqualWeightOptimizer",
    "HRPOptimizer",
    "InverseVolatilityOptimizer",
    "MaxDiversificationOptimizer",
    "MaxSharpeOptimizer",
    "MeanVarianceOptimizer",
    "MethodRequirements",
    "MinVarianceOptimizer",
    "OptimizationResult",
    "PortfolioConstraints",
    "RiskParityOptimizer",
    "optimizer_factory",
    "requirements_for",
]
```

- [ ] **Step 4: Wire validation into the factory**

Modify `src/optimization_engine/optimizers/factory.py`. Replace the `optimizer_factory` body so it validates via the registry first:

```python
"""Factory: build the right optimizer from a config object."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.optimizers import ConfigurationError
from optimization_engine.optimizers.base import BaseOptimizer, PortfolioConstraints
from optimization_engine.optimizers.black_litterman import BlackLittermanOptimizer
from optimization_engine.optimizers.cvar import CVaROptimizer
from optimization_engine.optimizers.hrp import HRPOptimizer
from optimization_engine.optimizers.max_diversification import MaxDiversificationOptimizer
from optimization_engine.optimizers.mean_variance import (
    MaxSharpeOptimizer,
    MeanVarianceOptimizer,
    MinVarianceOptimizer,
)
from optimization_engine.optimizers.naive import (
    EqualWeightOptimizer,
    InverseVolatilityOptimizer,
)
from optimization_engine.optimizers.requirements import requirements_for
from optimization_engine.optimizers.risk_parity import RiskParityOptimizer

_LOG = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseOptimizer]] = {
    "mean_variance": MeanVarianceOptimizer,
    "min_variance": MinVarianceOptimizer,
    "max_sharpe": MaxSharpeOptimizer,
    "risk_parity": RiskParityOptimizer,
    "hrp": HRPOptimizer,
    "black_litterman": BlackLittermanOptimizer,
    "cvar": CVaROptimizer,
    "max_diversification": MaxDiversificationOptimizer,
    "equal_weight": EqualWeightOptimizer,
    "inverse_vol": InverseVolatilityOptimizer,
}


def available_optimizers() -> list[str]:
    return sorted(_REGISTRY.keys())


def _constraints_from_config(config: EngineConfig) -> PortfolioConstraints:
    bounds = {k: tuple(v) for k, v in config.bounds.items()}
    group_bounds = {k: tuple(v) for k, v in config.group_bounds.items()}
    return PortfolioConstraints(
        bounds=bounds,
        groups=dict(config.groups),
        group_bounds=group_bounds,
        target_return=config.optimizer.target_return,
        target_volatility=config.optimizer.target_volatility,
    )


def _validate(spec: OptimizerSpec, expected_returns, cov_matrix, returns) -> None:
    req = requirements_for(spec.name)
    if req.requires_mu and (expected_returns is None or len(expected_returns) == 0):
        raise ConfigurationError(
            f"Optimizer '{spec.name}' requires expected_returns; got empty."
        )
    if req.requires_cov and cov_matrix is None:
        raise ConfigurationError(
            f"Optimizer '{spec.name}' requires a covariance matrix; got None."
        )
    if req.requires_returns and returns is None:
        raise ConfigurationError(
            f"Optimizer '{spec.name}' requires a returns DataFrame; got None."
        )
    if not req.supports_target_return and spec.target_return is not None:
        _LOG.warning(
            "Optimizer '%s' does not support target_return; ignoring value %s.",
            spec.name, spec.target_return,
        )
    if not req.supports_target_volatility and spec.target_volatility is not None:
        _LOG.warning(
            "Optimizer '%s' does not support target_volatility; ignoring value %s.",
            spec.name, spec.target_volatility,
        )


def optimizer_factory(
    config: EngineConfig,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    returns: pd.DataFrame | None = None,
    **overrides: Any,
) -> BaseOptimizer:
    """Build an optimizer instance from an :class:`EngineConfig`."""
    spec: OptimizerSpec = config.optimizer
    name = spec.name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {available_optimizers()}"
        )
    cls = _REGISTRY[name]

    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns, name="expected_return")

    _validate(spec, expected_returns, cov_matrix, returns)

    constraints = _constraints_from_config(config)

    common = dict(
        cov_matrix=cov_matrix,
        constraints=constraints,
        risk_free_rate=spec.risk_free_rate,
    )
    if cls is not CVaROptimizer:
        common["expected_returns"] = expected_returns

    if cls is MeanVarianceOptimizer:
        return cls(risk_aversion=spec.risk_aversion, **common, **overrides)
    if cls is RiskParityOptimizer:
        return cls(risk_budget=spec.risk_budget, **common, **overrides)
    if cls is HRPOptimizer:
        return cls(linkage_method=spec.hrp_linkage, **common, **overrides)
    if cls is BlackLittermanOptimizer:
        return cls(
            market_weights=spec.bl_market_caps,
            views=spec.bl_views,
            view_confidences=spec.bl_view_confidences,
            tau=spec.bl_tau,
            risk_aversion=spec.risk_aversion,
            **common,
            **overrides,
        )
    if cls is CVaROptimizer:
        return cls(
            returns=returns,
            alpha=spec.cvar_alpha,
            target_return=spec.target_return,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            constraints=constraints,
            risk_free_rate=spec.risk_free_rate,
            **overrides,
        )
    return cls(**common, **overrides)
```

Note this introduces a reference to `spec.hrp_linkage` that doesn't exist yet on `OptimizerSpec`. We'll add it in Task 4. To keep this commit green, **temporarily** read it via `getattr`:

Replace the HRP branch with:

```python
    if cls is HRPOptimizer:
        return cls(linkage_method=getattr(spec, "hrp_linkage", "single"), **common, **overrides)
```

- [ ] **Step 5: Run tests to verify validation works and existing tests still pass**

```bash
pytest tests/test_optimizers.py tests/test_requirements.py -v
```

Expected: all pass, including the three new validation tests.

- [ ] **Step 6: Commit**

```bash
git add src/optimization_engine/optimizers/__init__.py src/optimization_engine/optimizers/factory.py tests/test_optimizers.py
git commit -m "feat: validate optimizer config via MethodRequirements"
```

---

## Task 4: HRP iterated bounds + linkage_method spec field

**Files:**
- Modify: `src/optimization_engine/config.py` (`OptimizerSpec.hrp_linkage`)
- Modify: `src/optimization_engine/optimizers/hrp.py`
- Modify: `src/optimization_engine/optimizers/factory.py` (drop the `getattr` shim)
- Modify: `tests/test_optimizers.py` (parametrized linkage test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_optimizers.py`:

```python
@pytest.mark.parametrize("linkage", ["single", "average", "complete", "ward"])
def test_hrp_linkage_methods(returns, baseline_config, linkage):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        optimizer=OptimizerSpec(name="hrp", hrp_linkage=linkage),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert pytest.approx(w.sum(), abs=1e-3) == 1.0
    assert (w >= -1e-6).all()
    assert (w <= 0.5 + 1e-6).all()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_optimizers.py::test_hrp_linkage_methods -v
```

Expected: TypeError on `OptimizerSpec(..., hrp_linkage=...)`.

- [ ] **Step 3: Add the `hrp_linkage` field to `OptimizerSpec`**

In `src/optimization_engine/config.py`, modify `OptimizerSpec`:

```python
from typing import Any, Literal

@dataclass
class OptimizerSpec:
    """Specification of which optimizer to run and its hyperparameters."""

    name: str = "mean_variance"
    target_return: float | None = None
    target_volatility: float | None = None
    risk_free_rate: float = 0.0
    risk_aversion: float = 1.0
    cvar_alpha: float = 0.05
    risk_budget: dict[str, float] | None = None
    bl_views: dict[str, float] | None = None
    bl_view_confidences: dict[str, float] | None = None
    bl_tau: float = 0.05
    bl_market_caps: dict[str, float] | None = None
    hrp_linkage: Literal["single", "average", "complete", "ward"] = "single"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
```

- [ ] **Step 4: Switch HRP to iterated bounds**

In `src/optimization_engine/optimizers/hrp.py`, change the import and the projection call:

```python
# Replace this import:
#   from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays, project_to_bounds
# with:
from optimization_engine.optimizers._bounds import project_to_bounds_iterated
from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays
```

And in `_solve`, replace the final two lines:

```python
        lb, ub = bounds_arrays(self.assets, self.constraints)
        return project_to_bounds_iterated(weights, lb, ub)
```

- [ ] **Step 5: Drop the `getattr` shim in factory**

In `src/optimization_engine/optimizers/factory.py`, replace:

```python
    if cls is HRPOptimizer:
        return cls(linkage_method=getattr(spec, "hrp_linkage", "single"), **common, **overrides)
```

with:

```python
    if cls is HRPOptimizer:
        return cls(linkage_method=spec.hrp_linkage, **common, **overrides)
```

- [ ] **Step 6: Run all optimizer tests**

```bash
pytest tests/test_optimizers.py -v
```

Expected: all PASS (including 4 new HRP linkage cases).

- [ ] **Step 7: Commit**

```bash
git add src/optimization_engine/config.py src/optimization_engine/optimizers/hrp.py src/optimization_engine/optimizers/factory.py tests/test_optimizers.py
git commit -m "feat: HRP linkage method config + iterated bounds"
```

---

## Task 5: Max-diversification iterated bounds

**Files:**
- Modify: `src/optimization_engine/optimizers/max_diversification.py`
- Modify: `tests/test_optimizers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_optimizers.py`:

```python
def test_max_diversification_respects_tight_bounds(returns):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 0.3] for a in returns.columns},
        optimizer=OptimizerSpec(name="max_diversification"),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w <= 0.3 + 1e-6).all(), w[w > 0.3].to_dict()
    assert (w >= -1e-6).all()
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0
```

- [ ] **Step 2: Run to verify it may fail (depending on data, the one-shot version may exceed 0.3 after rescale)**

```bash
pytest tests/test_optimizers.py::test_max_diversification_respects_tight_bounds -v
```

Expected: either FAIL (bounds violated by one-shot rescale) or PASS by luck. Either way, switch to iterated.

- [ ] **Step 3: Switch the import + call**

In `src/optimization_engine/optimizers/max_diversification.py`:

```python
# Replace:
#   from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays, project_to_bounds
# with:
from optimization_engine.optimizers._bounds import project_to_bounds_iterated
from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays
```

And replace the last two lines of `_solve`:

```python
        lb, ub = bounds_arrays(self.assets, self.constraints)
        return project_to_bounds_iterated(w, lb, ub)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_optimizers.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/optimization_engine/optimizers/max_diversification.py tests/test_optimizers.py
git commit -m "feat: max-diversification iterated bounds projection"
```

---

## Task 6: Naive optimizers iterated bounds (equal_weight, inverse_vol)

**Files:**
- Modify: `src/optimization_engine/optimizers/naive.py`
- Modify: `tests/test_optimizers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_optimizers.py`:

```python
@pytest.mark.parametrize("method", ["equal_weight", "inverse_vol"])
def test_naive_methods_respect_tight_bounds(returns, method):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.0, 0.2] for a in returns.columns},
        optimizer=OptimizerSpec(name=method),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w <= 0.2 + 1e-6).all()
    assert (w >= -1e-6).all()
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0
```

- [ ] **Step 2: Run to verify failure on equal_weight (10 assets, bound 0.2 forces non-uniform allocation, one-shot rescale violates)**

```bash
pytest tests/test_optimizers.py::test_naive_methods_respect_tight_bounds -v
```

Expected: at least one FAIL.

- [ ] **Step 3: Switch the import + calls**

In `src/optimization_engine/optimizers/naive.py`:

```python
# Replace:
#   from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays, project_to_bounds
# with:
from optimization_engine.optimizers._bounds import project_to_bounds_iterated
from optimization_engine.optimizers._cvxpy_helpers import bounds_arrays
```

In both `EqualWeightOptimizer._solve` and `InverseVolatilityOptimizer._solve`, replace the final return:

```python
        lb, ub = bounds_arrays(self.assets, self.constraints)
        return project_to_bounds_iterated(w, lb, ub)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_optimizers.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/optimization_engine/optimizers/naive.py tests/test_optimizers.py
git commit -m "feat: naive optimizers iterated bounds projection"
```

---

## Task 7: Constrained risk-parity reformulation

**Files:**
- Modify: `src/optimization_engine/optimizers/risk_parity.py`
- Modify: `src/optimization_engine/optimizers/requirements.py`
- Modify: `tests/test_optimizers.py`
- Modify: `tests/test_requirements.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_optimizers.py`:

```python
def test_constrained_risk_parity_respects_bounds(returns):
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in returns.columns},
        bounds={a: [0.05, 0.25] for a in returns.columns},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    run = run_engine(returns, cfg)
    w = run.result.weights
    assert (w >= 0.05 - 1e-5).all(), w[w < 0.05].to_dict()
    assert (w <= 0.25 + 1e-5).all(), w[w > 0.25].to_dict()
    assert pytest.approx(w.sum(), abs=1e-4) == 1.0


def test_risk_parity_with_group_bounds(returns):
    cols = list(returns.columns)
    half = len(cols) // 2
    groups = {a: ("A" if i < half else "B") for i, a in enumerate(cols)}
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in cols},
        bounds={a: [0.0, 1.0] for a in cols},
        groups=groups,
        group_bounds={"A": [0.45, 0.55], "B": [0.45, 0.55]},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    run = run_engine(returns, cfg)
    g = run.result.weights.groupby(groups).sum()
    assert 0.45 - 1e-3 <= g["A"] <= 0.55 + 1e-3
    assert 0.45 - 1e-3 <= g["B"] <= 0.55 + 1e-3
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_optimizers.py -k "risk_parity_respects or risk_parity_with_group" -v
```

Expected: FAIL — current ERC ignores group bounds and the post-projection drifts.

- [ ] **Step 3: Reformulate `risk_parity._solve`**

Replace `src/optimization_engine/optimizers/risk_parity.py` `_solve` method body:

```python
    def _solve(self) -> np.ndarray:
        sigma = self._sigma_matrix()
        if sigma is None:
            raise ValueError("Covariance matrix required")
        n = len(self.assets)

        if self.risk_budget:
            b = np.array(
                [float(self.risk_budget.get(a, 1.0 / n)) for a in self.assets]
            )
        else:
            b = np.ones(n) / n
        if b.sum() <= 0:
            raise ValueError("Risk budget must sum to a positive number")
        b = b / b.sum()

        lb_arr = np.array(
            [float(self.constraints.get_bounds(a)[0]) for a in self.assets]
        )
        ub_arr = np.array(
            [float(self.constraints.get_bounds(a)[1]) for a in self.assets]
        )
        # Strict positivity for the log-barrier; tighten lb a hair if zero.
        lb_pos = np.maximum(lb_arr, 1e-8)

        y = cp.Variable(n, pos=True)
        sigma_psd = cp.psd_wrap(sigma)
        total = cp.sum(y)
        cons: list = [
            y >= cp.multiply(lb_pos, total),
            y <= cp.multiply(ub_arr, total),
        ]

        if self.constraints.groups and self.constraints.group_bounds:
            grouped: dict[str, list[int]] = {}
            for i, a in enumerate(self.assets):
                g = self.constraints.groups.get(a)
                if g is not None:
                    grouped.setdefault(g, []).append(i)
            for g, idx in grouped.items():
                if g in self.constraints.group_bounds:
                    g_lb, g_ub = self.constraints.group_bounds[g]
                    cons.append(cp.sum(y[idx]) >= float(g_lb) * total)
                    cons.append(cp.sum(y[idx]) <= float(g_ub) * total)

        objective = cp.Minimize(0.5 * cp.quad_form(y, sigma_psd) - b @ cp.log(y))
        problem = cp.Problem(objective, cons)
        try:
            problem.solve()
        except cp.SolverError:
            problem.solve(solver=cp.SCS)
        if y.value is None:
            raise RuntimeError(f"Solver failed: status={problem.status}")

        w = np.array(y.value) / float(np.sum(y.value))
        # Clamp tiny floating drift back into the box.
        return np.clip(w, lb_arr, ub_arr)
```

The `bounds_arrays` / `project_to_bounds` imports become unused — remove them from the imports block:

```python
import cvxpy as cp
import numpy as np
import pandas as pd

from optimization_engine.optimizers.base import BaseOptimizer
```

- [ ] **Step 4: Update the registry to reflect the new capabilities**

In `src/optimization_engine/optimizers/requirements.py`, find the `risk_parity` entry and flip the two flags that the constrained reformulation now actually honors:

```python
    "risk_parity": MethodRequirements(
        name="risk_parity",
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="constrained",
        supports_frontier=False, extras=(_RISK_BUDGET,),
    ),
```

In `tests/test_requirements.py`, update `EXPECTED_FLAGS["risk_parity"]` to match:

```python
    "risk_parity": dict(
        requires_mu=False, requires_cov=True, requires_returns=False,
        supports_target_return=False, supports_target_volatility=False,
        supports_risk_aversion=False, supports_risk_free_rate=False,
        supports_group_bounds=True, bounds_mode="constrained",
        supports_frontier=False,
    ),
```

(Task 2 had set both to the conservative-but-current values `False` /
`"soft_iterated"` because the old `_solve` only post-projected per-asset
bounds. Now that the CVXPY reformulation enforces both per-asset and group
bounds natively, the registry can advertise them.)

- [ ] **Step 5: Run tests**

```bash
.venv/Scripts/pytest.exe tests/test_optimizers.py tests/test_requirements.py -v
```

Expected: all PASS, including the two new constrained-RP tests, the existing `test_risk_parity_equal_contributions`, and the registry parametrized case for `risk_parity`.

- [ ] **Step 6: Commit**

```bash
git add src/optimization_engine/optimizers/risk_parity.py src/optimization_engine/optimizers/requirements.py tests/test_optimizers.py tests/test_requirements.py
git commit -m "feat: constrained risk-parity respects asset and group bounds"
```

---

## Task 8: Remove unused `project_to_bounds`

**Files:**
- Modify: `src/optimization_engine/optimizers/_cvxpy_helpers.py`

- [ ] **Step 1: Verify nothing imports `project_to_bounds`**

```bash
grep -rn "project_to_bounds[^_]" src/ tests/ app/
```

Expected: no matches (only `project_to_bounds_iterated` should appear). If anything still imports the one-shot version, fix that file first.

- [ ] **Step 2: Remove the helper**

Open `src/optimization_engine/optimizers/_cvxpy_helpers.py` and delete the `project_to_bounds` function and its export line. Keep `bounds_arrays`, `build_constraints`, etc.

- [ ] **Step 3: Run the full suite**

```bash
pytest -q
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/optimization_engine/optimizers/_cvxpy_helpers.py
git commit -m "refactor: drop unused one-shot project_to_bounds"
```

---

## Task 9: Expected-returns method on EngineConfig + engine wiring

**Files:**
- Modify: `src/optimization_engine/config.py`
- Modify: `src/optimization_engine/engine.py`
- Modify: `tests/test_optimizers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_optimizers.py`:

```python
def test_engine_uses_ema_expected_returns_when_specified(returns):
    cfg = EngineConfig(
        expected_returns={},  # empty -> engine seeds from history
        bounds={a: [0.0, 1.0] for a in returns.columns},
        expected_returns_method="ema",
        ema_span=120,
        optimizer=OptimizerSpec(name="min_variance"),  # min_variance ignores mu but engine still computes it
    )
    run = run_engine(returns, cfg)
    # Sanity: μ vector populated with finite values.
    assert run.expected_returns.notna().all()
    assert run.expected_returns.shape[0] == returns.shape[1]


def test_engine_uses_capm_expected_returns_when_specified(returns):
    cols = list(returns.columns)
    cfg = EngineConfig(
        expected_returns={},
        bounds={a: [0.0, 1.0] for a in cols},
        expected_returns_method="capm",
        market_weights={a: 1.0 / len(cols) for a in cols},
        market_return=0.08,
        optimizer=OptimizerSpec(name="min_variance", risk_free_rate=0.03),
    )
    run = run_engine(returns, cfg)
    assert run.expected_returns.notna().all()


def test_engine_default_method_unchanged(returns, baseline_config):
    # Existing test_optimizer_runs already covers this; just ensure default
    # historical_mean still works when method left at default.
    cfg = EngineConfig(
        expected_returns={},
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        optimizer=OptimizerSpec(name="min_variance"),
    )
    run = run_engine(returns, cfg)
    assert run.expected_returns.notna().all()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_optimizers.py -k "ema_expected or capm_expected or default_method_unchanged" -v
```

Expected: TypeError on `EngineConfig(..., expected_returns_method=...)`.

- [ ] **Step 3: Add the new fields to `EngineConfig`**

Modify `src/optimization_engine/config.py`. Add to `EngineConfig`:

```python
from typing import Any, Literal


@dataclass
class EngineConfig:
    expected_returns: dict[str, float] = field(default_factory=dict)
    bounds: dict[str, list[float]] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    group_bounds: dict[str, list[float]] = field(default_factory=dict)
    currencies: dict[str, str] = field(default_factory=dict)
    base_currency: str = "USD"
    periods_per_year: int = 252
    covariance_method: str = "ledoit_wolf"
    ewma_lambda: float = 0.94
    expected_returns_method: Literal["historical_mean", "ema", "capm"] = "historical_mean"
    ema_span: int = 180
    market_return: float | None = None
    market_weights: dict[str, float] | None = None
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    benchmark_weights: dict[str, float] | None = None
```

Update `to_dict`:

```python
    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_returns": dict(self.expected_returns),
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "groups": dict(self.groups),
            "group_bounds": {k: list(v) for k, v in self.group_bounds.items()},
            "currencies": dict(self.currencies),
            "base_currency": self.base_currency,
            "periods_per_year": self.periods_per_year,
            "covariance_method": self.covariance_method,
            "ewma_lambda": self.ewma_lambda,
            "expected_returns_method": self.expected_returns_method,
            "ema_span": self.ema_span,
            "market_return": self.market_return,
            "market_weights": (dict(self.market_weights) if self.market_weights else None),
            "optimizer": self.optimizer.to_dict(),
            "benchmark_weights": self.benchmark_weights,
        }
```

Update `from_dict`:

```python
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        opt_raw = data.get("optimizer") or {}
        if isinstance(opt_raw, str):
            opt_raw = {"name": opt_raw}
        return cls(
            expected_returns=dict(data.get("expected_returns") or {}),
            bounds={k: list(v) for k, v in (data.get("bounds") or {}).items()},
            groups=dict(data.get("groups") or {}),
            group_bounds={k: list(v) for k, v in (data.get("group_bounds") or {}).items()},
            currencies=dict(data.get("currencies") or {}),
            base_currency=str(data.get("base_currency", "USD")).upper(),
            periods_per_year=int(data.get("periods_per_year", 252)),
            covariance_method=str(data.get("covariance_method", "ledoit_wolf")),
            ewma_lambda=float(data.get("ewma_lambda", 0.94)),
            expected_returns_method=str(
                data.get("expected_returns_method", "historical_mean")
            ),
            ema_span=int(data.get("ema_span", 180)),
            market_return=(
                float(data["market_return"])
                if data.get("market_return") is not None else None
            ),
            market_weights=(
                dict(data["market_weights"])
                if data.get("market_weights") else None
            ),
            optimizer=OptimizerSpec(**opt_raw),
            benchmark_weights=data.get("benchmark_weights"),
        )
```

- [ ] **Step 4: Wire `run_engine` to use the new fields**

In `src/optimization_engine/engine.py`, replace the expected-returns-seeding block in `run_engine`:

```python
    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns)
    if expected_returns is None:
        from optimization_engine.data.covariance import expected_returns_from_history

        market_w = (
            pd.Series(config.market_weights) if config.market_weights else None
        )
        expected_returns = expected_returns_from_history(
            returns,
            method=config.expected_returns_method,  # "historical_mean" | "ema" | "capm"
            periods_per_year=config.periods_per_year,
            span=config.ema_span,
            market_return=config.market_return,
            risk_free_rate=config.optimizer.risk_free_rate,
            market_weights=market_w,
            cov_matrix=cov,
        )
    expected_returns = expected_returns.reindex(returns.columns).fillna(0.0)
```

(Remove the inline `(1 + returns).prod() ** ...` fallback — `expected_returns_from_history` covers it via `method="historical_mean"`.)

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_optimizers.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/optimization_engine/config.py src/optimization_engine/engine.py tests/test_optimizers.py
git commit -m "feat: configurable expected-returns method (historical/ema/capm)"
```

---

## Task 10: Frontier parallelization

**Files:**
- Modify: `src/optimization_engine/frontier.py`
- Create: `tests/test_frontier_parallel.py`

- [ ] **Step 1: Write the failing test**

`tests/test_frontier_parallel.py`:

```python
"""Frontier parallelization correctness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.config import EngineConfig, OptimizerSpec
from optimization_engine.data.covariance import covariance_matrix
from optimization_engine.data.loader import prices_to_returns, sample_dataset
from optimization_engine.frontier import efficient_frontier


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return prices_to_returns(sample_dataset(n_periods=252 * 4, seed=11))


@pytest.fixture(scope="module")
def baseline_config(returns: pd.DataFrame) -> EngineConfig:
    expected = (1 + returns).prod() ** (252 / len(returns)) - 1
    return EngineConfig(
        expected_returns=expected.to_dict(),
        bounds={a: [0.0, 0.5] for a in returns.columns},
        groups={a: "All" for a in returns.columns},
        group_bounds={"All": [1.0, 1.0]},
        optimizer=OptimizerSpec(name="mean_variance"),
    )


def test_parallel_matches_serial(returns, baseline_config):
    cov = covariance_matrix(returns, method="ledoit_wolf")
    mu = pd.Series(baseline_config.expected_returns)
    f1 = efficient_frontier(baseline_config, cov, mu, n_points=12, n_workers=1)
    f4 = efficient_frontier(baseline_config, cov, mu, n_points=12, n_workers=4)
    pd.testing.assert_frame_equal(
        f1.summary.drop(columns="status"),
        f4.summary.drop(columns="status"),
        check_exact=False, atol=1e-6,
    )


def test_risk_aversion_sweep_monotone(returns, baseline_config):
    cov = covariance_matrix(returns, method="ledoit_wolf")
    mu = pd.Series(baseline_config.expected_returns)
    fr = efficient_frontier(
        baseline_config, cov, mu,
        n_points=10, sweep="risk_aversion",
    )
    vols = fr.summary["expected_volatility"].dropna().values
    assert (np.diff(vols) <= 1e-6).all(), vols.tolist()


def test_infeasible_targets_surface_as_failed_rows(returns, baseline_config):
    cov = covariance_matrix(returns, method="ledoit_wolf")
    mu = pd.Series(baseline_config.expected_returns)
    fr = efficient_frontier(
        baseline_config, cov, mu,
        n_points=5, return_range=(10.0, 20.0),
    )
    assert fr.summary["status"].str.startswith("failed").any()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_frontier_parallel.py -v
```

Expected: TypeError on `efficient_frontier(..., n_workers=...)`.

- [ ] **Step 3: Update `efficient_frontier`**

In `src/optimization_engine/frontier.py`, replace the body of `efficient_frontier` (keep the dataclass and `_group_weights` helper):

```python
"""Efficient frontier construction."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from optimization_engine.config import EngineConfig
from optimization_engine.optimizers.factory import optimizer_factory


@dataclass
class FrontierResult:
    summary: pd.DataFrame
    weights: pd.DataFrame
    group_weights: pd.DataFrame | None = None

    @property
    def max_sharpe_index(self) -> int:
        return int(np.argmax(self.summary["sharpe_ratio"].values))


def _group_weights(weights: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame()
    g = pd.Series(groups)
    expanded = weights.copy()
    expanded["__group__"] = expanded.index.map(g)
    grouped = expanded.groupby("__group__").sum(numeric_only=True)
    return grouped


def _solve_one(
    target: float,
    base_config: EngineConfig,
    sweep: str,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None,
    returns: pd.DataFrame | None,
):
    cfg = copy.deepcopy(base_config)
    if sweep == "return":
        cfg.optimizer.target_return = float(target)
        cfg.optimizer.risk_aversion = 1.0
    else:
        cfg.optimizer.target_return = None
        cfg.optimizer.risk_aversion = float(target)
    try:
        result = optimizer_factory(
            cfg, cov_matrix,
            expected_returns=expected_returns,
            returns=returns,
        ).optimize()
        return float(target), result, None
    except Exception as exc:
        return float(target), None, str(exc)


def efficient_frontier(
    config: EngineConfig,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    returns: pd.DataFrame | None = None,
    target_returns: Iterable[float] | None = None,
    n_points: int = 25,
    sweep: Literal["return", "risk_aversion"] = "return",
    return_range: tuple[float, float] | None = None,
    n_workers: int | None = None,
) -> FrontierResult:
    if expected_returns is None and config.expected_returns:
        expected_returns = pd.Series(config.expected_returns)

    if sweep == "return":
        if target_returns is None:
            if expected_returns is None:
                raise ValueError("Cannot sweep returns without expected_returns")
            mu = expected_returns.reindex(cov_matrix.columns).fillna(0.0).values
            lo, hi = (return_range
                      if return_range is not None
                      else (float(mu.min()), float(mu.max())))
            target_returns = list(np.linspace(lo, hi, n_points))
        else:
            target_returns = list(target_returns)
    elif sweep == "risk_aversion":
        target_returns = list(np.geomspace(0.5, 50.0, n_points))
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    base_config = copy.deepcopy(config)
    if base_config.optimizer.name not in {"mean_variance", "cvar"}:
        base_config.optimizer.name = "mean_variance"

    workers = n_workers if n_workers is not None else min(8, len(target_returns))
    if workers <= 1:
        rows = [
            _solve_one(t, base_config, sweep, cov_matrix, expected_returns, returns)
            for t in target_returns
        ]
    else:
        n = len(target_returns)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(
                _solve_one,
                target_returns,
                [base_config] * n,
                [sweep] * n,
                [cov_matrix] * n,
                [expected_returns] * n,
                [returns] * n,
            ))

    summary_rows: list[dict[str, float]] = []
    weights_rows: list[pd.Series] = []
    for target, result, err in rows:
        if result is None:
            summary_rows.append({
                "target": target,
                "expected_return": np.nan,
                "expected_volatility": np.nan,
                "sharpe_ratio": np.nan,
                "status": f"failed: {err}",
            })
            weights_rows.append(
                pd.Series(np.nan, index=cov_matrix.columns, name=target)
            )
        else:
            summary_rows.append({
                "target": target,
                "expected_return": result.expected_return,
                "expected_volatility": result.expected_volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "status": "ok",
            })
            weights_rows.append(result.weights.rename(target))

    summary = pd.DataFrame(summary_rows)
    weights_df = pd.concat(weights_rows, axis=1)
    weights_df.columns = summary["target"].values
    return FrontierResult(
        summary=summary,
        weights=weights_df,
        group_weights=_group_weights(weights_df, base_config.groups),
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_frontier_parallel.py tests/test_optimizers.py -v
```

Expected: all PASS, including the existing `test_efficient_frontier`.

- [ ] **Step 5: Commit**

```bash
git add src/optimization_engine/frontier.py tests/test_frontier_parallel.py
git commit -m "feat: parallelize frontier solves on a thread pool"
```

---

## Task 11: `derive_widget_state` helper

**Files:**
- Modify: `src/optimization_engine/ui_state.py`
- Create: `tests/test_ui_logic.py`

- [ ] **Step 1: Write the failing test**

`tests/test_ui_logic.py`:

```python
"""Tests for the registry-driven widget-state helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.optimizers.factory import available_optimizers
from optimization_engine.ui_state import derive_widget_state


def test_returns_dict_for_known_method():
    ws = derive_widget_state("mean_variance")
    assert isinstance(ws, dict)
    assert ws["risk_free_rate"]["enabled"] is True
    assert ws["frontier"]["enabled"] is True
    assert ws["target_return"]["enabled"] is True


def test_disabled_widgets_have_tooltip():
    for name in available_optimizers():
        ws = derive_widget_state(name)
        for key, state in ws.items():
            if not state["enabled"]:
                assert state["tooltip"], f"{name}.{key} disabled with no tooltip"


@pytest.mark.parametrize("method,enabled,disabled", [
    ("mean_variance",
     {"risk_free_rate", "cov_method", "frontier", "expected_returns_column",
      "group_bounds", "target_return", "target_volatility", "risk_aversion"},
     set()),
    ("min_variance",
     {"cov_method", "group_bounds"},
     {"risk_free_rate", "frontier", "expected_returns_column",
      "target_return", "target_volatility", "risk_aversion"}),
    ("max_sharpe",
     {"risk_free_rate", "cov_method", "expected_returns_column",
      "group_bounds"},
     {"frontier", "target_return", "target_volatility", "risk_aversion"}),
    ("hrp",
     {"cov_method", "hrp_linkage"},
     {"risk_free_rate", "frontier", "expected_returns_column",
      "group_bounds", "target_return", "target_volatility", "risk_aversion"}),
    ("equal_weight",
     set(),
     {"risk_free_rate", "cov_method", "frontier", "expected_returns_column",
      "group_bounds", "target_return", "target_volatility", "risk_aversion"}),
    ("risk_parity",
     {"cov_method", "group_bounds", "risk_budget"},
     {"risk_free_rate", "frontier", "expected_returns_column",
      "target_return", "target_volatility", "risk_aversion"}),
    ("cvar",
     {"risk_free_rate", "target_return", "cvar_alpha", "group_bounds"},
     {"cov_method", "frontier", "target_volatility", "risk_aversion"}),
    ("black_litterman",
     {"risk_free_rate", "cov_method", "frontier", "group_bounds",
      "bl_views", "bl_tau", "bl_market_caps", "risk_aversion"},
     set()),
    ("max_diversification",
     {"cov_method"},
     {"risk_free_rate", "frontier", "expected_returns_column",
      "group_bounds", "target_return", "target_volatility", "risk_aversion"}),
    ("inverse_vol",
     {"cov_method"},
     {"risk_free_rate", "frontier", "expected_returns_column",
      "group_bounds", "target_return", "target_volatility", "risk_aversion"}),
])
def test_widget_state_per_method(method, enabled, disabled):
    ws = derive_widget_state(method)
    for key in enabled:
        assert ws[key]["enabled"], f"{method}.{key} should be enabled"
    for key in disabled:
        assert not ws[key]["enabled"], f"{method}.{key} should be disabled"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_ui_logic.py -v
```

Expected: ImportError on `derive_widget_state`.

- [ ] **Step 3: Add the helper**

Append to `src/optimization_engine/ui_state.py`:

```python
from optimization_engine.optimizers.requirements import (
    MethodRequirements,
    requirements_for,
)


_NOT_USED = "Not used by this optimizer."


def _state(enabled: bool, tooltip: str | None = None) -> dict[str, object]:
    return {"enabled": enabled, "tooltip": tooltip if not enabled else None}


def derive_widget_state(method_name: str) -> dict[str, dict[str, object]]:
    """Map widget keys to enabled/tooltip state for the given optimizer.

    Pure function — used by the Streamlit app and easy to unit-test.
    """
    req: MethodRequirements = requirements_for(method_name)
    extra_keys = {e.key for e in req.extras}

    state: dict[str, dict[str, object]] = {
        "risk_free_rate": _state(
            req.supports_risk_free_rate,
            f"{_NOT_USED} (risk-free rate)",
        ),
        "cov_method": _state(
            req.requires_cov,
            f"{_NOT_USED} (no covariance estimate needed)",
        ),
        "ewma_lambda": _state(
            req.requires_cov,
            f"{_NOT_USED} (no covariance estimate needed)",
        ),
        "expected_returns_column": _state(
            req.requires_mu,
            f"{method_name} doesn't use expected returns.",
        ),
        "expected_returns_method": _state(
            req.requires_mu,
            f"{method_name} doesn't use expected returns.",
        ),
        "group_bounds": _state(
            req.supports_group_bounds,
            f"{method_name} does not enforce group bounds.",
        ),
        "frontier": _state(
            req.supports_frontier,
            f"Frontier sweep is only meaningful for mean-variance / Black-Litterman.",
        ),
        "target_return": _state(
            req.supports_target_return,
            f"{method_name} does not accept a return target.",
        ),
        "target_volatility": _state(
            req.supports_target_volatility,
            f"{method_name} does not accept a volatility target.",
        ),
        "risk_aversion": _state(
            req.supports_risk_aversion,
            f"{method_name} does not use a risk-aversion utility.",
        ),
        "soft_bounds_caption": _state(
            req.bounds_mode != "hard",
            None,
        ),
    }

    # Optimizer-specific extras: enabled iff present in this method's extras.
    for extra_key in (
        "risk_budget", "bl_views", "bl_view_confidences",
        "bl_tau", "bl_market_caps", "cvar_alpha", "hrp_linkage",
    ):
        state[extra_key] = _state(
            extra_key in extra_keys,
            f"Used only by methods that expose '{extra_key}'.",
        )
    return state
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ui_logic.py -v
```

Expected: all PASS (1 + 1 over-all-methods + 10 parametrized).

- [ ] **Step 5: Commit**

```bash
git add src/optimization_engine/ui_state.py tests/test_ui_logic.py
git commit -m "feat: derive_widget_state for registry-driven Streamlit UI"
```

---

## Task 12: Streamlit sidebar refactor

**Files:**
- Modify: `app/streamlit_app.py`

This task touches the sidebar's "3 · Optimizer" block and the "4 · Frontier" block. The constraints tab and what-if tab are handled in later tasks.

- [ ] **Step 1: Extend the existing `ui_state` import**

In `app/streamlit_app.py` find the existing block:

```python
from optimization_engine.ui_state import (  # noqa: E402
    yahoo_cache_key,
    yahoo_prices_for_rerun,
)
```

and replace it with:

```python
from optimization_engine.ui_state import (  # noqa: E402
    derive_widget_state,
    yahoo_cache_key,
    yahoo_prices_for_rerun,
)
```

(Both names live in the same `ui_state.py` module.)

- [ ] **Step 2: Compute widget state once after the optimizer is chosen**

In the sidebar block, after `optimizer_name = st.selectbox(...)`, add:

```python
    ws = derive_widget_state(optimizer_name)
```

- [ ] **Step 3: Drive risk-free rate, cov method, EWMA λ via the registry**

Replace the existing inputs with:

```python
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0, max_value=0.30,
        value=0.04, step=0.005, format="%.4f",
        key="risk_free_rate",
        disabled=not ws["risk_free_rate"]["enabled"],
        help=ws["risk_free_rate"]["tooltip"],
    )
    periods_per_year = st.number_input(
        "Periods per year", min_value=1, max_value=365, value=252,
        key="periods_per_year",
    )
    cov_method = st.selectbox(
        "Covariance estimator",
        options=["ledoit_wolf", "sample", "oas", "ewma", "semi", "shrink"],
        index=0,
        key="cov_method",
        disabled=not ws["cov_method"]["enabled"],
        help=ws["cov_method"]["tooltip"],
    )
    ewma_lambda = (
        st.slider(
            "EWMA λ", 0.80, 0.999, 0.94, 0.005,
            key="ewma_lambda",
            disabled=not ws["ewma_lambda"]["enabled"],
            help=ws["ewma_lambda"]["tooltip"],
        )
        if cov_method == "ewma" and ws["cov_method"]["enabled"]
        else 0.94
    )
```

- [ ] **Step 4: Show MV mode block only when supported**

Replace the `if optimizer_name == "mean_variance":` block with:

```python
    target_return: float | None = None
    target_volatility: float | None = None
    risk_aversion = 1.0
    cvar_alpha = 0.05
    risk_budget: dict[str, float] | None = None

    # The Mode radio is for methods that genuinely offer >1 mode
    # (mean_variance and Black-Litterman). CVaR has only target_return and
    # is handled in its own block below.
    show_mode_radio = (
        optimizer_name != "cvar"
        and (ws["target_return"]["enabled"] or ws["target_volatility"]["enabled"])
        and ws["risk_aversion"]["enabled"]
    )
    if show_mode_radio:
        modes = []
        if ws["target_return"]["enabled"]:
            modes.append("Target return")
        if ws["target_volatility"]["enabled"]:
            modes.append("Target volatility")
        if ws["risk_aversion"]["enabled"]:
            modes.append("Utility")
        mode = st.radio(
            "Mode", modes, horizontal=True, key="mv_mode",
        )
        if mode == "Target return":
            target_return = st.number_input(
                "Target return (annual)",
                value=0.07, step=0.005, format="%.4f",
                key="target_return",
            )
        elif mode == "Target volatility":
            target_volatility = st.number_input(
                "Target volatility (annual)",
                value=0.10, step=0.005, format="%.4f",
                key="target_volatility",
            )
        else:
            risk_aversion = st.slider(
                "Risk aversion λ", 0.1, 20.0, 2.5, key="risk_aversion",
            )
    if optimizer_name == "cvar":
        cvar_alpha = st.slider(
            "CVaR tail prob α", 0.01, 0.20, 0.05, 0.01,
            key="cvar_alpha",
            help="0.05 ⇒ 95% CVaR.",
        )
        target_return = st.number_input(
            "Target return (optional)",
            value=0.0, step=0.005, format="%.4f",
            key="target_return_cvar",
        )
        target_return = None if target_return == 0.0 else target_return
```

- [ ] **Step 5: Frontier checkbox driven by registry**

Replace the "4 · Frontier" block with:

```python
    st.divider()
    st.header("4 · Frontier")
    build_frontier = st.checkbox(
        "Build efficient frontier",
        value=True,
        disabled=not ws["frontier"]["enabled"],
        help=ws["frontier"]["tooltip"],
    )
    n_frontier_points = st.slider(
        "Frontier points", 5, 100, 25,
        disabled=not ws["frontier"]["enabled"],
    )
    if not ws["frontier"]["enabled"]:
        build_frontier = False
```

- [ ] **Step 6: Smoke-run the app**

```bash
streamlit run app/streamlit_app.py
```

Confirm visually that switching between optimizers (mean_variance → hrp → equal_weight) disables the appropriate sidebar fields with hover tooltips. Stop the server with Ctrl-C.

- [ ] **Step 7: Run pytest (no UI tests touched, but ensure nothing else regressed)**

```bash
pytest -q
```

- [ ] **Step 8: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: registry-driven sidebar widget visibility"
```

---

## Task 13: Streamlit constraints tab refactor

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Disable the "Expected Return" column when not required**

In the constraints tab, replace the `column_config` block on the `st.data_editor(st.session_state.config_table, ...)` call:

```python
    edited = st.data_editor(
        st.session_state.config_table,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Expected Return": st.column_config.NumberColumn(
                format="%.4f",
                disabled=not ws["expected_returns_column"]["enabled"],
                help=ws["expected_returns_column"]["tooltip"],
            ),
            "Min Weight": st.column_config.NumberColumn(min_value=-1.0, max_value=1.0, step=0.01, format="%.2f"),
            "Max Weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.5, step=0.01, format="%.2f"),
            "Group": st.column_config.TextColumn(),
            "Currency": st.column_config.SelectboxColumn(
                "Currency",
                options=supported_currencies(),
                help="ISO code of the currency the price series is quoted in.",
            ),
        },
    )
```

- [ ] **Step 2: Add expected-returns-method radio + EMA + CAPM inputs**

Above the data_editor (still inside the constraints tab), add:

```python
    if ws["expected_returns_method"]["enabled"]:
        er_method = st.radio(
            "Expected-returns method",
            options=["historical_mean", "ema", "capm"],
            index=["historical_mean", "ema", "capm"].index(
                st.session_state.get("expected_returns_method", "historical_mean")
            ),
            horizontal=True,
            key="expected_returns_method",
        )
        if er_method == "ema":
            st.session_state.ema_span = st.slider(
                "EMA span (periods)",
                min_value=30, max_value=504,
                value=int(st.session_state.get("ema_span", 180)),
                step=10,
                key="ema_span_slider",
            )
        elif er_method == "capm":
            st.session_state.market_return = st.number_input(
                "Market return (annual, optional)",
                value=float(st.session_state.get("market_return") or 0.08),
                step=0.005, format="%.4f",
                key="market_return_input",
            )
            mw_idx = list(returns.columns)
            mw_default = pd.DataFrame(
                {"Market weight": [1.0 / len(mw_idx)] * len(mw_idx)},
                index=mw_idx,
            )
            if "market_weights_table" not in st.session_state:
                st.session_state.market_weights_table = mw_default
            st.session_state.market_weights_table = st.data_editor(
                st.session_state.market_weights_table,
                num_rows="fixed",
                column_config={
                    "Market weight": st.column_config.NumberColumn(
                        min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                    ),
                },
            )

        if st.button("Reset μ to method default", key="reset_mu_btn"):
            from optimization_engine.data.covariance import expected_returns_from_history, covariance_matrix as _cov

            mw_for_capm = (
                pd.Series(st.session_state.market_weights_table["Market weight"])
                if er_method == "capm"
                and "market_weights_table" in st.session_state
                else None
            )
            seeded = expected_returns_from_history(
                returns,
                method=er_method,
                periods_per_year=int(periods_per_year),
                span=int(st.session_state.get("ema_span", 180)),
                market_return=float(st.session_state.get("market_return") or 0.0) or None,
                risk_free_rate=float(risk_free_rate),
                market_weights=mw_for_capm,
                cov_matrix=_cov(returns, method=cov_method, ewma_lambda=ewma_lambda,
                                periods_per_year=int(periods_per_year)),
            )
            st.session_state.config_table["Expected Return"] = seeded.round(4)
            st.rerun()
```

- [ ] **Step 3: Hide group-bounds editor when not supported**

Wrap the existing "Group constraints" block:

```python
    if ws["group_bounds"]["enabled"]:
        st.markdown("**Group constraints**")
        # ... existing code unchanged ...
    else:
        st.caption(
            f"_{optimizer_name} does not enforce group bounds — group editor hidden._"
        )
```

- [ ] **Step 4: Soft-bounds caption**

Just above the `st.data_editor` for `config_table`, add:

```python
    if ws["soft_bounds_caption"]["enabled"]:
        st.caption(
            "_Bounds are enforced via projection on solved weights "
            "(soft bounds; small drift may occur for marginal cases)._"
        )
```

- [ ] **Step 5: BL extras (tau slider + market_caps table)**

Replace the existing BL views block:

```python
    if optimizer_name == "black_litterman":
        st.markdown("**Black-Litterman views** (asset → annual expected return)")
        if "bl_views" not in st.session_state or set(st.session_state.bl_views.index) != set(returns.columns):
            st.session_state.bl_views = pd.DataFrame(
                {"View": np.nan, "Confidence (variance)": np.nan},
                index=returns.columns,
            )
        st.session_state.bl_views = st.data_editor(
            st.session_state.bl_views,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "View": st.column_config.NumberColumn(format="%.4f"),
                "Confidence (variance)": st.column_config.NumberColumn(format="%.6f"),
            },
        )

        st.markdown("**Black-Litterman extras**")
        st.session_state["bl_tau"] = st.slider(
            "τ (prior uncertainty)",
            min_value=0.01, max_value=0.5,
            value=float(st.session_state.get("bl_tau", 0.05)),
            step=0.01,
            key="bl_tau_slider",
        )
        if (
            "bl_market_caps_table" not in st.session_state
            or set(st.session_state.bl_market_caps_table.index) != set(returns.columns)
        ):
            st.session_state.bl_market_caps_table = pd.DataFrame(
                {"Market cap weight": [1.0 / len(returns.columns)] * len(returns.columns)},
                index=returns.columns,
            )
        st.session_state.bl_market_caps_table = st.data_editor(
            st.session_state.bl_market_caps_table,
            num_rows="fixed",
            column_config={
                "Market cap weight": st.column_config.NumberColumn(
                    min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                    help="Equal weights → equilibrium under no views. Set to your view of the market portfolio.",
                ),
            },
        )
```

- [ ] **Step 6: HRP linkage selectbox**

Add inside the constraints tab, after the BL block:

```python
    if optimizer_name == "hrp":
        st.session_state["hrp_linkage"] = st.selectbox(
            "HRP linkage method",
            options=["single", "average", "complete", "ward"],
            index=["single", "average", "complete", "ward"].index(
                st.session_state.get("hrp_linkage", "single")
            ),
            key="hrp_linkage_select",
            help="Hierarchical clustering linkage rule.",
        )
```

- [ ] **Step 7: Update `_build_config` to include the new fields**

Modify the `_build_config()` function:

```python
def _build_config() -> EngineConfig:
    table = st.session_state.config_table
    bounds = {
        a: [float(table.loc[a, "Min Weight"]), float(table.loc[a, "Max Weight"])]
        for a in returns.columns
    }
    groups = {a: str(table.loc[a, "Group"]) for a in returns.columns}
    group_bounds: dict[str, list[float]] = {}
    if "group_bounds" in st.session_state:
        for g, row in st.session_state.group_bounds.iterrows():
            group_bounds[str(g)] = [float(row["Min Weight"]), float(row["Max Weight"])]

    expected_returns = {a: float(table.loc[a, "Expected Return"]) for a in returns.columns}

    spec = OptimizerSpec(
        name=optimizer_name,
        target_return=target_return,
        target_volatility=target_volatility,
        risk_free_rate=float(risk_free_rate),
        risk_aversion=float(risk_aversion),
        cvar_alpha=float(cvar_alpha),
        bl_tau=float(st.session_state.get("bl_tau", 0.05)),
        hrp_linkage=str(st.session_state.get("hrp_linkage", "single")),
    )

    if optimizer_name == "risk_parity" and "risk_budget" in st.session_state:
        spec.risk_budget = st.session_state.risk_budget["Risk Budget"].to_dict()

    if optimizer_name == "black_litterman":
        if "bl_views" in st.session_state:
            v = st.session_state.bl_views
            spec.bl_views = {
                a: float(v.loc[a, "View"])
                for a in v.index
                if pd.notna(v.loc[a, "View"])
            }
            spec.bl_view_confidences = {
                a: float(v.loc[a, "Confidence (variance)"])
                for a in v.index
                if pd.notna(v.loc[a, "Confidence (variance)"])
            }
        if "bl_market_caps_table" in st.session_state:
            spec.bl_market_caps = (
                st.session_state.bl_market_caps_table["Market cap weight"].to_dict()
            )

    market_weights = None
    if (
        st.session_state.get("expected_returns_method") == "capm"
        and "market_weights_table" in st.session_state
    ):
        market_weights = (
            st.session_state.market_weights_table["Market weight"].to_dict()
        )

    return EngineConfig(
        expected_returns=expected_returns,
        bounds=bounds,
        groups=groups,
        group_bounds=group_bounds,
        currencies=dict(st.session_state.asset_currency),
        base_currency=base_currency,
        periods_per_year=int(periods_per_year),
        covariance_method=cov_method,
        ewma_lambda=float(ewma_lambda),
        expected_returns_method=str(
            st.session_state.get("expected_returns_method", "historical_mean")
        ),
        ema_span=int(st.session_state.get("ema_span", 180)),
        market_return=(
            float(st.session_state.get("market_return"))
            if st.session_state.get("expected_returns_method") == "capm"
            and st.session_state.get("market_return")
            else None
        ),
        market_weights=market_weights,
        optimizer=spec,
    )
```

- [ ] **Step 8: Smoke-run the app**

```bash
streamlit run app/streamlit_app.py
```

Walk through: switch to HRP → linkage selectbox appears, μ column disabled. Switch to BL → views table + tau slider + market caps table show. Switch to mean_variance → CAPM radio works, "Reset μ" button reseeds the table.

- [ ] **Step 9: Run pytest**

```bash
pytest -q
```

- [ ] **Step 10: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: registry-driven constraints tab + new method-specific inputs"
```

---

## Task 14: What-if tab — registry-driven extras

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Replace the hardcoded mode/α block in the What-if tab**

In the What-if tab body, replace the `**Optimizer extras**` section (`if opt_name == "mean_variance":` ... `elif opt_name == "cvar":` ...) with:

```python
        st.markdown("**Optimizer extras**")
        from optimization_engine.optimizers.requirements import requirements_for as _req_for

        req = _req_for(opt_name)
        extras: dict[str, object] = dict(st.session_state.whatif_extra)

        if req.supports_target_return or req.supports_target_volatility:
            modes = []
            if req.supports_target_return:
                modes.append("Target return")
            if req.supports_target_volatility:
                modes.append("Target volatility")
            if req.supports_risk_aversion:
                modes.append("Utility")
            if anchor_cfg.optimizer.target_return is not None:
                default_mode = "Target return"
            elif anchor_cfg.optimizer.target_volatility is not None:
                default_mode = "Target volatility"
            else:
                default_mode = modes[0]
            wf_mode = st.radio(
                "Mode", modes,
                index=modes.index(default_mode) if default_mode in modes else 0,
                horizontal=True,
                key="whatif_mv_mode",
            )
            if wf_mode == "Target return":
                tr = st.number_input(
                    "Target return (annual)",
                    value=float(anchor_cfg.optimizer.target_return or 0.07),
                    step=0.005, format="%.4f",
                    key="whatif_target_return",
                )
                extras = {"target_return": float(tr), "target_volatility": None}
            elif wf_mode == "Target volatility":
                tv = st.number_input(
                    "Target volatility (annual)",
                    value=float(anchor_cfg.optimizer.target_volatility or 0.10),
                    step=0.005, format="%.4f",
                    key="whatif_target_vol",
                )
                extras = {"target_return": None, "target_volatility": float(tv)}
            else:
                ra = st.slider(
                    "Risk aversion λ", 0.1, 20.0,
                    float(anchor_cfg.optimizer.risk_aversion or 2.5),
                    key="whatif_risk_aversion",
                )
                extras = {
                    "target_return": None,
                    "target_volatility": None,
                    "risk_aversion": float(ra),
                }

        for extra in req.extras:
            if extra.kind == "scalar" and extra.key == "cvar_alpha":
                ca = st.slider(
                    "CVaR tail prob α", 0.01, 0.20,
                    float(anchor_cfg.optimizer.cvar_alpha or 0.05), 0.01,
                    key="whatif_cvar_alpha",
                )
                extras["cvar_alpha"] = float(ca)
            elif extra.kind == "scalar" and extra.key == "bl_tau":
                tau = st.slider(
                    "τ (prior uncertainty)", 0.01, 0.5,
                    float(anchor_cfg.optimizer.bl_tau or 0.05), 0.01,
                    key="whatif_bl_tau",
                )
                extras["bl_tau"] = float(tau)
            elif extra.kind == "choice" and extra.key == "hrp_linkage":
                lk = st.selectbox(
                    "HRP linkage", list(extra.choices or ()),
                    index=(extra.choices or ("single",)).index(
                        anchor_cfg.optimizer.hrp_linkage or "single"
                    ),
                    key="whatif_hrp_linkage",
                )
                extras["hrp_linkage"] = str(lk)
            elif extra.kind == "per_asset" and extra.key == "risk_budget":
                # Editable risk budget as a small frame.
                rb_idx = list(anchor_cfg.expected_returns.keys())
                default_rb = anchor_cfg.optimizer.risk_budget or {
                    a: 1.0 / len(rb_idx) for a in rb_idx
                }
                rb_df = pd.DataFrame(
                    {"Risk Budget": [default_rb.get(a, 0.0) for a in rb_idx]},
                    index=rb_idx,
                )
                rb_df = st.data_editor(
                    rb_df, num_rows="fixed",
                    column_config={
                        "Risk Budget": st.column_config.NumberColumn(
                            min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
                        ),
                    },
                    key="whatif_rb_editor",
                )
                extras["risk_budget"] = rb_df["Risk Budget"].to_dict()
            # view tables and market caps are skipped in What-if to keep it light;
            # users can edit those in the constraints tab.

        st.session_state.whatif_extra = extras
```

- [ ] **Step 2: Smoke-run**

```bash
streamlit run app/streamlit_app.py
```

Save a risk_parity scenario, switch to What-if, edit the risk-budget table, confirm the live solve recomputes with the new budget.

- [ ] **Step 3: Run pytest**

```bash
pytest -q
```

- [ ] **Step 4: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: registry-driven What-if optimizer extras (RP/BL/HRP)"
```

---

## Task 15: Streamlit covariance + historical_mu caching

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Add the helpers near the top of the file**

After the existing `_load_yahoo_cached` helper:

```python
def _frame_hash(df: pd.DataFrame) -> str:
    return pd.util.hash_pandas_object(df, index=True).values.tobytes().hex()


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


@st.cache_data(show_spinner=False, max_entries=8)
def _historical_mu_cached(
    returns_hash: str,
    periods_per_year: int,
    _returns: pd.DataFrame,
) -> pd.Series:
    return (1 + _returns).prod() ** (periods_per_year / len(_returns)) - 1
```

- [ ] **Step 2: Replace inline `historical_mu` in the constraints tab**

Find:

```python
    historical_mu = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
```

Replace with:

```python
    historical_mu = _historical_mu_cached(
        _frame_hash(returns), int(periods_per_year), returns,
    )
```

- [ ] **Step 3: Replace covariance computation in the "Reset μ to method default" CAPM branch**

Find:

```python
                cov_matrix=_cov(returns, method=cov_method, ewma_lambda=ewma_lambda,
                                periods_per_year=int(periods_per_year)),
```

Replace with:

```python
                cov_matrix=_covariance_cached(
                    _frame_hash(returns),
                    cov_method, float(ewma_lambda),
                    int(periods_per_year), True,
                    returns,
                ),
```

(And drop the now-unused `from optimization_engine.data.covariance import ... covariance_matrix as _cov` import inside that button handler.)

- [ ] **Step 4: Smoke-run**

```bash
streamlit run app/streamlit_app.py
```

Toggle a slider on the constraints tab; the page should feel notably snappier (no covariance recompute on each rerun for unrelated UI changes).

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py
git commit -m "perf: cache covariance + historical_mu per (returns,params)"
```

---

## Task 16: Optimize tab — frontier section visibility

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Wrap the frontier rendering**

In the Optimize tab, find:

```python
    if run.frontier is not None:
        st.markdown("### Efficient Frontier")
```

Replace the entire `if run.frontier is not None:` block with:

```python
    if run.frontier is not None and ws["frontier"]["enabled"]:
        st.markdown("### Efficient Frontier")
        st.plotly_chart(
            plot_efficient_frontier(run.frontier.summary, run.frontier.max_sharpe_index),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plot_portfolio_composition(run.frontier.weights, "Weights along frontier"),
                use_container_width=True,
            )
        with c2:
            if run.frontier.group_weights is not None and not run.frontier.group_weights.empty:
                st.plotly_chart(
                    plot_portfolio_composition(
                        run.frontier.group_weights, "Group weights along frontier"
                    ),
                    use_container_width=True,
                )
```

- [ ] **Step 2: Smoke-run**

```bash
streamlit run app/streamlit_app.py
```

Switch to HRP → frontier checkbox is disabled, frontier panel doesn't render even if a previous mean_variance run had populated `run.frontier`.

- [ ] **Step 3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: hide frontier section when optimizer doesn't support it"
```

---

## Task 17: Extended optimizer tests — group bounds, infeasible, BL, CVaR target

**Files:**
- Modify: `tests/test_optimizers.py`

- [ ] **Step 1: Append all new tests**

Append to `tests/test_optimizers.py`:

```python
def test_cvar_with_target_return(returns, baseline_config):
    target = 0.05
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        optimizer=OptimizerSpec(
            name="cvar", cvar_alpha=0.05, target_return=target,
        ),
    )
    run = run_engine(returns, cfg)
    assert run.result.expected_return >= target - 1e-3


def test_black_litterman_no_views_runs(returns, baseline_config):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        groups=baseline_config.groups,
        group_bounds=baseline_config.group_bounds,
        optimizer=OptimizerSpec(name="black_litterman", risk_aversion=2.5),
    )
    run = run_engine(returns, cfg)
    assert pytest.approx(run.result.weights.sum(), abs=1e-3) == 1.0
    assert (run.result.weights >= -1e-6).all()


@pytest.mark.parametrize("method", [
    "mean_variance", "min_variance", "max_sharpe", "cvar", "black_litterman",
])
def test_group_bounds_enforced_for_hard_methods(returns, method):
    cols = list(returns.columns)
    half = len(cols) // 2
    groups = {a: ("A" if i < half else "B") for i, a in enumerate(cols)}
    cfg = EngineConfig(
        expected_returns={a: 0.05 for a in cols},
        bounds={a: [0.0, 1.0] for a in cols},
        groups=groups,
        group_bounds={"A": [0.4, 0.6], "B": [0.4, 0.6]},
        optimizer=OptimizerSpec(name=method, risk_free_rate=0.0),
    )
    run = run_engine(returns, cfg)
    g = run.result.weights.groupby(groups).sum()
    assert 0.4 - 2e-3 <= g["A"] <= 0.6 + 2e-3, g.to_dict()
    assert 0.4 - 2e-3 <= g["B"] <= 0.6 + 2e-3, g.to_dict()


def test_infeasible_target_raises_clearly(returns, baseline_config):
    cfg = EngineConfig(
        expected_returns=baseline_config.expected_returns,
        bounds=baseline_config.bounds,
        optimizer=OptimizerSpec(name="mean_variance", target_return=10.0),
    )
    with pytest.raises(RuntimeError, match=r"infeasible|status|Solver"):
        run_engine(returns, cfg)
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_optimizers.py -v
```

Expected: all PASS (5 new test rows from the parametrize, plus 3 individual tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizers.py
git commit -m "test: cover CVaR target, BL no-views, hard-method group bounds, infeasible"
```

---

## Task 18: Final pytest sweep and PR readiness

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
pytest -q --tb=short
```

Expected: all PASS, including the existing tests on `main` plus the new ones in:

- `tests/test_bounds_projection.py` (7)
- `tests/test_requirements.py` (13)
- `tests/test_ui_logic.py` (12)
- `tests/test_frontier_parallel.py` (3)
- `tests/test_optimizers.py` (existing + ~20 new)

If any test fails, stop and investigate the root cause — do **not** mark the plan complete with red tests.

- [ ] **Step 2: Final smoke run of the Streamlit app**

```bash
streamlit run app/streamlit_app.py
```

Walk through every optimizer in the dropdown and confirm:
- Disabled fields show tooltips when hovered.
- HRP shows linkage selectbox; BL shows tau + market caps + views.
- Frontier toggle disables for non-frontier methods.
- Save a scenario per method, then Compare; What-if recomputes live.

Stop with Ctrl-C.

- [ ] **Step 3: Confirm `git status` is clean**

```bash
git status
```

Expected: working tree clean (apart from any pre-existing modifications to `app/streamlit_app.py`, `src/optimization_engine/ui_state.py`, and `tests/test_ui_state.py` that were present when this plan started — those were from earlier work and aren't this plan's responsibility).

- [ ] **Step 4: View commit log**

```bash
git log --oneline -20
```

Expected: 17 new commits (one per Task 1–17), each with a clear `feat:`/`refactor:`/`perf:`/`test:` prefix.

---

## What's intentionally out of scope

- Streamlit `AppTest` end-to-end driving (deferred to a future spec if it ever feels worth the maintenance cost).
- A constrained-HRP variant — iterated projection is good enough.
- Process-pool parallelism on the frontier — threads suffice for the scale we ship at.
- The redesigned tab layout (Approach C from brainstorming).
