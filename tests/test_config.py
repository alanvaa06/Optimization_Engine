"""The config loader refuses what it cannot honour (review fix E8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine.config import EngineConfig, load_config  # noqa: E402
from optimization_engine.optimizers import ConfigurationError  # noqa: E402


def test_a_misspelt_key_is_refused_rather_than_ignored():
    # `max_tracking_eror` loaded cleanly and simply did not constrain anything.
    with pytest.raises(ConfigurationError, match="max_tracking_eror"):
        EngineConfig.from_dict({"max_tracking_eror": 0.03, "optimizer": "min_variance"})


def test_an_unknown_optimizer_key_is_a_configuration_error_not_a_type_error():
    with pytest.raises(ConfigurationError, match="Unknown optimizer key"):
        EngineConfig.from_dict({"optimizer": {"name": "cvar", "cvar_alph": 0.05}})


def test_a_config_round_trips_through_its_own_dict():
    config = load_config(ROOT / "config" / "example_multi_asset.yaml")
    again = EngineConfig.from_dict(config.to_dict())
    assert again.to_dict() == config.to_dict()


def test_the_shipped_indices_config_declares_its_currencies():
    # It used to spell the block `asset_currency`, which nothing read, so the
    # conversion its own comment promised never happened.
    config = load_config(ROOT / "config" / "indices.yaml")
    assert config.currencies["DAX"] == "EUR"
    assert config.currencies["NIKKEI225"] == "JPY"
