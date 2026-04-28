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
