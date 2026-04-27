"""Tests for the named-scenarios module."""

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
from optimization_engine.data.loader import prices_to_returns, sample_dataset  # noqa: E402
from optimization_engine.engine import run_engine  # noqa: E402
from optimization_engine.scenarios import (  # noqa: E402
    NOTES_MAX_LEN,
    SCHEMA_VERSION,
    Scenario,
    config_signature,
    delete_scenario,
    dump_scenarios_yaml,
    load_scenarios,
    load_scenarios_yaml,
    rename_scenario,
    save_scenarios,
    scenario_signature,
)


# ---------------------------------------------------------------------------
# Fixtures (mirrored from tests/test_optimizers.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    prices = sample_dataset(n_periods=252 * 4, seed=7)
    return prices_to_returns(prices)


@pytest.fixture(scope="module")
def baseline_config(returns: pd.DataFrame) -> EngineConfig:
    expected = (1 + returns).prod() ** (252 / len(returns)) - 1
    return EngineConfig(
        expected_returns=expected.to_dict(),
        bounds={a: [0.0, 0.5] for a in returns.columns},
        groups={a: "All" for a in returns.columns},
        group_bounds={"All": [1.0, 1.0]},
        optimizer=OptimizerSpec(name="mean_variance", risk_free_rate=0.03),
    )


@pytest.fixture
def two_scenarios(baseline_config: EngineConfig) -> dict[str, Scenario]:
    rp_cfg = EngineConfig.from_dict(baseline_config.to_dict())
    rp_cfg.optimizer = OptimizerSpec(name="risk_parity", risk_free_rate=0.03)
    return {
        "Baseline": Scenario(
            name="Baseline",
            config=baseline_config,
            notes="MV target=auto",
            created_at="2026-04-27T10:00:00",
            updated_at="2026-04-27T10:00:00",
        ),
        "RiskParity": Scenario(
            name="RiskParity",
            config=rp_cfg,
            notes="ERC fallback",
            created_at="2026-04-27T10:01:00",
            updated_at="2026-04-27T10:01:00",
        ),
    }


# ---------------------------------------------------------------------------
# 1. YAML round-trip (in-memory)
# ---------------------------------------------------------------------------


def test_scenario_yaml_round_trip(two_scenarios: dict[str, Scenario]):
    text = dump_scenarios_yaml(two_scenarios)
    back = load_scenarios_yaml(text)
    assert list(back.keys()) == list(two_scenarios.keys())
    for k in two_scenarios:
        assert back[k].name == two_scenarios[k].name
        assert back[k].notes == two_scenarios[k].notes
        assert back[k].created_at == two_scenarios[k].created_at
        assert back[k].updated_at == two_scenarios[k].updated_at
        assert back[k].config.to_dict() == two_scenarios[k].config.to_dict()


# ---------------------------------------------------------------------------
# 2. Filesystem round-trip (yaml + json)
# ---------------------------------------------------------------------------


def test_save_and_load_file_round_trip(
    tmp_path: Path, two_scenarios: dict[str, Scenario]
):
    yaml_path = tmp_path / "scenarios.yaml"
    save_scenarios(two_scenarios, yaml_path)
    yaml_back = load_scenarios(yaml_path)
    assert {k: v.config.to_dict() for k, v in yaml_back.items()} == {
        k: v.config.to_dict() for k, v in two_scenarios.items()
    }

    json_path = tmp_path / "scenarios.json"
    save_scenarios(two_scenarios, json_path)
    json_back = load_scenarios(json_path)
    assert list(json_back.keys()) == list(two_scenarios.keys())


# ---------------------------------------------------------------------------
# 3. Schema version guard
# ---------------------------------------------------------------------------


def test_load_rejects_unknown_schema_version():
    payload = (
        "schema_version: 999\n"
        "scenarios: []\n"
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_scenarios_yaml(payload)


# ---------------------------------------------------------------------------
# 4. Duplicate name guard
# ---------------------------------------------------------------------------


def test_load_rejects_duplicate_names():
    payload = """
schema_version: 1
scenarios:
  - name: Foo
    config: {expected_returns: {A: 0.05}, bounds: {A: [0.0, 1.0]}, optimizer: {name: equal_weight}}
  - name: Foo
    config: {expected_returns: {A: 0.06}, bounds: {A: [0.0, 1.0]}, optimizer: {name: equal_weight}}
"""
    with pytest.raises(ValueError, match="Duplicate"):
        load_scenarios_yaml(payload)


# ---------------------------------------------------------------------------
# 5. End-to-end: round-trip preserves solver result exactly
# ---------------------------------------------------------------------------


def test_loaded_scenario_solves_identically(
    returns: pd.DataFrame, baseline_config: EngineConfig
):
    scn = Scenario(name="Baseline", config=baseline_config)
    text = dump_scenarios_yaml({"Baseline": scn})
    back = load_scenarios_yaml(text)["Baseline"]

    direct = run_engine(returns, baseline_config)
    via_yaml = run_engine(returns, back.config)
    np.testing.assert_allclose(
        direct.result.weights.values,
        via_yaml.result.weights.values,
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# 6. Rename helper
# ---------------------------------------------------------------------------


def test_rename_preserves_config_and_metadata(two_scenarios: dict[str, Scenario]):
    orig_created = two_scenarios["Baseline"].created_at
    out = rename_scenario(two_scenarios, "Baseline", "Baseline V2", touch=True)
    assert list(out.keys()) == ["Baseline V2", "RiskParity"]
    moved = out["Baseline V2"]
    assert moved.name == "Baseline V2"
    assert moved.config.to_dict() == two_scenarios["Baseline"].config.to_dict()
    assert moved.created_at == orig_created
    assert moved.updated_at != orig_created  # bumped


def test_rename_rejects_collision(two_scenarios: dict[str, Scenario]):
    with pytest.raises(ValueError, match="already exists"):
        rename_scenario(two_scenarios, "Baseline", "RiskParity")


# ---------------------------------------------------------------------------
# 7. Delete helper
# ---------------------------------------------------------------------------


def test_delete_removes_only_target(two_scenarios: dict[str, Scenario]):
    out = delete_scenario(two_scenarios, "Baseline")
    assert list(out.keys()) == ["RiskParity"]
    # original dict not mutated
    assert "Baseline" in two_scenarios


# ---------------------------------------------------------------------------
# 8. Infeasible bounds raise (covers the What-if error path)
# ---------------------------------------------------------------------------


def test_infeasible_bounds_raise(returns: pd.DataFrame, baseline_config: EngineConfig):
    bad_dict = baseline_config.to_dict()
    # Cap every asset at 5%; with N>20 assets the sum of upper bounds < 1
    bad_dict["bounds"] = {a: [0.0, 0.05] for a in bad_dict["expected_returns"]}
    bad_cfg = EngineConfig.from_dict(bad_dict)
    with pytest.raises(Exception):
        run_engine(returns, bad_cfg)


# ---------------------------------------------------------------------------
# 9. Cache-key signature stable across dict insertion order
# ---------------------------------------------------------------------------


def test_config_signature_stable_across_dict_order():
    cfg_a = EngineConfig(
        expected_returns={"A": 0.05, "B": 0.03, "C": 0.04},
        bounds={"A": [0.0, 1.0], "B": [0.0, 1.0], "C": [0.0, 1.0]},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    cfg_b = EngineConfig(
        expected_returns={"C": 0.04, "A": 0.05, "B": 0.03},
        bounds={"B": [0.0, 1.0], "A": [0.0, 1.0], "C": [0.0, 1.0]},
        optimizer=OptimizerSpec(name="risk_parity"),
    )
    assert config_signature(cfg_a) == config_signature(cfg_b)
    assert scenario_signature(Scenario("A", cfg_a)) == scenario_signature(Scenario("B", cfg_b))


# ---------------------------------------------------------------------------
# 10. Notes truncation respected on save
# ---------------------------------------------------------------------------


def test_notes_truncation(baseline_config: EngineConfig):
    big = "x" * (NOTES_MAX_LEN + 50)
    scn = Scenario(name="Big", config=baseline_config, notes=big)
    text = dump_scenarios_yaml({"Big": scn})
    back = load_scenarios_yaml(text)["Big"]
    assert len(back.notes) == NOTES_MAX_LEN


def test_schema_version_constant():
    # Pin: bumping SCHEMA_VERSION is a deliberate breaking change.
    assert SCHEMA_VERSION == 1
