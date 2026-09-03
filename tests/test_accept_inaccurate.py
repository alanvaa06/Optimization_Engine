"""How ``accept_inaccurate`` travels from a config down to a CVXPY solve.

The refusal itself is tested in ``test_analytical_rigor.py``; what is tested
here is the wiring. There is no argument threaded through the optimizers --
each one calls ``solve_problem(problem)`` with no keyword from inside its own
``_solve`` -- so the only proof that a config's setting reaches the solver is
to force every solver to answer ``optimal_inaccurate`` and see which end of
the pipe notices.
"""

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

from optimization_engine.config import EngineConfig, OptimizerSpec  # noqa: E402
from optimization_engine.optimizers._cvxpy_helpers import (  # noqa: E402
    SolverFailure,
    accepting_inaccurate,
)
from optimization_engine.optimizers.factory import optimizer_factory  # noqa: E402

ASSETS = ["A", "B", "C", "D"]


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        rng.normal(0.0004, 0.011, size=(400, len(ASSETS))),
        columns=ASSETS,
        index=pd.bdate_range("2021-01-01", periods=400),
    )


@pytest.fixture
def cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * 252


@pytest.fixture
def mu(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * 252


def _force_inaccurate(monkeypatch) -> None:
    """Make every solver answer ``optimal_inaccurate`` on an easy problem.

    Patched at ``cp.Problem.solve`` rather than at ``solve_problem`` so that
    every call site in the package is covered at once, including the ones
    reached through a sub-optimizer.
    """
    import cvxpy as cp

    real_solve = cp.Problem.solve

    def always_inaccurate(self, *args, **kwargs):
        real_solve(self, *args, **kwargs)
        self._status = "optimal_inaccurate"

    monkeypatch.setattr(cp.Problem, "solve", always_inaccurate)


def _config(name: str, mu: pd.Series, **spec_kwargs) -> EngineConfig:
    return EngineConfig(
        expected_returns=mu.to_dict(),
        bounds={a: [0.0, 0.6] for a in ASSETS},
        optimizer=OptimizerSpec(name=name, **spec_kwargs),
    )


# ---------------------------------------------------------------------------
# The config field itself
# ---------------------------------------------------------------------------


def test_accept_inaccurate_round_trips_through_the_config():
    """``to_dict`` drops ``None``, not ``False``, so the default survives."""
    config = _config("min_variance", pd.Series(0.05, index=ASSETS))
    payload = config.to_dict()
    assert payload["optimizer"]["accept_inaccurate"] is False
    assert EngineConfig.from_dict(payload).optimizer.accept_inaccurate is False

    config.optimizer.accept_inaccurate = True
    again = EngineConfig.from_dict(config.to_dict())
    assert again.optimizer.accept_inaccurate is True
    assert again.to_dict() == config.to_dict()


def test_accept_inaccurate_is_an_accepted_optimizer_key():
    # ``_OPTIMIZER_KEYS`` derives from the dataclass fields, so a config file
    # that sets it must load rather than raise ConfigurationError.
    config = EngineConfig.from_dict(
        {"optimizer": {"name": "min_variance", "accept_inaccurate": True}}
    )
    assert config.optimizer.accept_inaccurate is True


# ---------------------------------------------------------------------------
# From the config to an actual solve
# ---------------------------------------------------------------------------


def test_config_default_refuses_at_the_solve(monkeypatch, cov, mu):
    """The default reaches the solver, and the solver's answer is refused."""
    import optimization_engine.optimizers.mean_variance as mean_variance

    seen: list[object] = []
    real = mean_variance.solve_problem

    def spy(problem, *args, **kwargs):
        seen.append(kwargs.get("accept_inaccurate", "not-passed"))
        return real(problem, *args, **kwargs)

    monkeypatch.setattr(mean_variance, "solve_problem", spy)

    optimizer = optimizer_factory(
        _config("min_variance", mu), cov_matrix=cov, expected_returns=mu
    )
    assert optimizer.accept_inaccurate is False

    _force_inaccurate(monkeypatch)
    with pytest.raises(SolverFailure) as excinfo:
        optimizer.optimize()

    assert excinfo.value.status == "optimal_inaccurate"
    # The optimizer really did reach ``solve_problem``, and really did call it
    # without a keyword -- the scope is what carried the decision.
    assert seen == ["not-passed"]


def test_config_opt_in_reaches_the_solve(monkeypatch, cov, mu):
    import optimization_engine.optimizers.mean_variance as mean_variance

    seen: list[object] = []
    real = mean_variance.solve_problem

    def spy(problem, *args, **kwargs):
        seen.append(kwargs.get("accept_inaccurate", "not-passed"))
        return real(problem, *args, **kwargs)

    monkeypatch.setattr(mean_variance, "solve_problem", spy)

    optimizer = optimizer_factory(
        _config("min_variance", mu, accept_inaccurate=True),
        cov_matrix=cov,
        expected_returns=mu,
    )
    assert optimizer.accept_inaccurate is True

    _force_inaccurate(monkeypatch)
    result = optimizer.optimize()

    assert result.solver_status == "optimal_inaccurate"
    assert seen == ["not-passed"]
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", ["cvar", "cdar"])
def test_path_dependent_methods_get_the_flag(monkeypatch, returns, cov, mu, name):
    """CVaR and CDaR are built without the factory's ``common`` dict.

    They re-list every base argument by hand, so a base-class parameter added
    to ``common`` alone silently stops at them.
    """
    config = _config(name, mu, accept_inaccurate=True)
    optimizer = optimizer_factory(
        config, cov_matrix=cov, expected_returns=mu, returns=returns
    )
    assert optimizer.accept_inaccurate is True

    refusing = optimizer_factory(
        _config(name, mu), cov_matrix=cov, expected_returns=mu, returns=returns
    )
    assert refusing.accept_inaccurate is False

    _force_inaccurate(monkeypatch)
    assert optimizer.optimize().solver_status == "optimal_inaccurate"
    with pytest.raises(SolverFailure) as excinfo:
        refusing.optimize()
    assert excinfo.value.status == "optimal_inaccurate"


def test_nco_sub_solves_inherit_the_setting(monkeypatch, cov, mu):
    """NCO is not a no-op method: both its layers are solved by optimizers.

    Those sub-optimizers are built inside ``_solve`` and are never told what
    the outer solve decided, so only the ambient scope can reach them.
    """
    optimizer = optimizer_factory(
        _config("nco", mu, accept_inaccurate=True), cov_matrix=cov, expected_returns=mu
    )
    refusing = optimizer_factory(_config("nco", mu), cov_matrix=cov, expected_returns=mu)

    _force_inaccurate(monkeypatch)
    assert optimizer.optimize().weights.sum() == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(SolverFailure):
        refusing.optimize()


@pytest.mark.parametrize("name", ["hrp", "herc", "equal_weight", "inverse_vol"])
def test_solverless_methods_are_unaffected(monkeypatch, returns, cov, mu, name):
    """The flag is a no-op for the methods that never call a solver."""
    _force_inaccurate(monkeypatch)
    optimizer = optimizer_factory(
        _config(name, mu), cov_matrix=cov, expected_returns=mu, returns=returns
    )
    assert optimizer.optimize().weights.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The one call site that keeps accepting
# ---------------------------------------------------------------------------


def test_dust_cleanup_projection_still_accepts_inaccurate(monkeypatch, cov):
    """``project_to_constraints`` is a correction, not the portfolio.

    Refusing an inaccurate answer there would fail every soft-bounds method
    whose real solve had already succeeded, so that call site passes
    ``accept_inaccurate=True`` explicitly and ignores the surrounding scope.
    """
    from optimization_engine.optimizers._bounds import project_to_constraints
    from optimization_engine.optimizers.base import PortfolioConstraints

    constraints = PortfolioConstraints(
        groups={"A": "x", "B": "x", "C": "y", "D": "y"},
        group_bounds={"x": (0.0, 0.5), "y": (0.0, 0.5)},
    )
    _force_inaccurate(monkeypatch)
    with accepting_inaccurate(False):
        projected, distance = project_to_constraints(
            np.array([0.7, 0.1, 0.1, 0.1]), ASSETS, constraints
        )
    assert projected.sum() == pytest.approx(1.0, abs=1e-4)
    assert distance >= 0.0


def test_risk_parity_keeps_its_own_solver_chain(monkeypatch, cov, mu):
    """The narrowed chain at ``risk_parity.py`` survives the threading.

    Risk parity's objective is exponential-cone, so it passes
    ``solvers=("CLARABEL", "SCS", "ECOS")`` of its own and must keep doing so.
    Carrying ``accept_inaccurate`` ambiently rather than as a keyword is what
    lets it reach that call without the call being rewritten.
    """
    optimizer = optimizer_factory(
        _config("risk_parity", mu, accept_inaccurate=True),
        cov_matrix=cov,
        expected_returns=mu,
    )
    refusing = optimizer_factory(
        _config("risk_parity", mu), cov_matrix=cov, expected_returns=mu
    )

    _force_inaccurate(monkeypatch)
    result = optimizer.optimize()
    assert result.solver_status == "optimal_inaccurate"
    # Its own chain, not SOLVER_FALLBACK -- which would also have tried OSQP.
    assert set(result.extras["solvers_attempted"]) <= {"CLARABEL", "SCS", "ECOS"}

    with pytest.raises(SolverFailure) as excinfo:
        refusing.optimize()
    assert excinfo.value.attempts and set(excinfo.value.attempts) <= {
        "CLARABEL",
        "SCS",
        "ECOS",
    }
