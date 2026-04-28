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
