"""The ``scenarios`` → ``presets`` deprecation shim.

The behaviour these names implement is covered by ``tests/test_presets.py``.
What is covered here is the promise the shim makes: that the old import path
still warns *and* still works — including a full save/load round-trip through
it, because a deprecation that quietly stops persisting anything is worse than
no deprecation at all.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimization_engine import presets  # noqa: E402
from optimization_engine.config import EngineConfig, OptimizerSpec  # noqa: E402

#: Every pre-0.7.0 name, and what it is now. The shim is only as good as this
#: table, so the table is the test.
RENAMES = {
    "Scenario": "Preset",
    "scenario_to_dict": "preset_to_dict",
    "scenario_from_dict": "preset_from_dict",
    "dump_scenarios_yaml": "dump_presets_yaml",
    "load_scenarios_yaml": "load_presets_yaml",
    "save_scenarios": "save_presets",
    "load_scenarios": "load_presets",
    "rename_scenario": "rename_preset",
    "delete_scenario": "delete_preset",
    "scenario_signature": "preset_signature",
}

#: The names that did not move.
UNCHANGED = ("SCHEMA_VERSION", "NOTES_MAX_LEN", "config_signature", "now_iso")


def _reimport_scenarios():
    """Import ``optimization_engine.scenarios`` with its module body re-run.

    A module-level warning fires once per interpreter, when the module body
    executes. Dropping the entry from ``sys.modules`` is what makes the
    warning observable in a suite that has already imported it. Safe to do
    repeatedly: every name the shim binds comes from ``presets``, which is not
    reloaded, so the re-executed module binds the identical objects.
    """
    sys.modules.pop("optimization_engine.scenarios", None)
    return importlib.import_module("optimization_engine.scenarios")


@pytest.fixture
def config() -> EngineConfig:
    return EngineConfig(
        expected_returns={"A": 0.05, "B": 0.03},
        bounds={"A": [0.0, 1.0], "B": [0.0, 1.0]},
        optimizer=OptimizerSpec(name="equal_weight"),
    )


# ---------------------------------------------------------------------------
# 1. The warning
# ---------------------------------------------------------------------------


def test_importing_scenarios_warns_and_names_its_replacement():
    with pytest.warns(DeprecationWarning, match="optimization_engine.presets"):
        module = _reimport_scenarios()
    assert module.Preset is presets.Preset


def test_importing_the_package_itself_does_not_warn():
    """``__init__`` must reach ``presets`` directly.

    Routing the package's own eager imports through the shim would fire a
    deprecation warning at ``import optimization_engine`` — a warning about a
    module the caller never named, which they cannot act on.
    """
    sys.modules.pop("optimization_engine.scenarios", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(importlib.import_module("optimization_engine"))
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not [m for m in messages if "optimization_engine.scenarios" in m]


# ---------------------------------------------------------------------------
# 2. Every old name still resolves, to the same object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("old", "new"), sorted(RENAMES.items()))
def test_old_name_is_the_new_object(old: str, new: str):
    with pytest.warns(DeprecationWarning):
        module = _reimport_scenarios()
    assert getattr(module, old) is getattr(presets, new)


@pytest.mark.parametrize("name", UNCHANGED)
def test_unchanged_names_are_still_there(name: str):
    with pytest.warns(DeprecationWarning):
        module = _reimport_scenarios()
    assert getattr(module, name) == getattr(presets, name)


def test_scenario_is_preset_not_a_subclass_of_it(config: EngineConfig):
    """``isinstance`` against the old name must still hold for new objects."""
    with pytest.warns(DeprecationWarning):
        module = _reimport_scenarios()
    assert module.Scenario is presets.Preset
    assert isinstance(presets.Preset("X", config), module.Scenario)


def test_the_shim_exports_everything_it_documents():
    with pytest.warns(DeprecationWarning):
        module = _reimport_scenarios()
    for name in [*RENAMES, *RENAMES.values(), *UNCHANGED]:
        assert name in module.__all__, f"{name} is missing from the shim's __all__"
        assert hasattr(module, name)


# ---------------------------------------------------------------------------
# 3. The old path still round-trips a saved config
# ---------------------------------------------------------------------------


def test_the_old_import_path_still_round_trips_a_saved_config(
    tmp_path: Path, config: EngineConfig
):
    """Save through the deprecated names, load through them, get it back.

    The end-to-end promise of "keep it working for one release": a caller who
    has not migrated yet writes a file and reads it back with no change in
    behaviour, and the file is the same file the new names produce.
    """
    with pytest.warns(DeprecationWarning):
        module = _reimport_scenarios()

    saved = {
        "Baseline": module.Scenario(
            name="Baseline",
            config=config,
            notes="via the old path",
            created_at="2026-09-02T10:00:00+00:00",
            updated_at="2026-09-02T10:00:00+00:00",
        )
    }

    path = tmp_path / "scenarios.yaml"
    module.save_scenarios(saved, path)
    back = module.load_scenarios(path)

    assert list(back) == ["Baseline"]
    assert back["Baseline"].name == "Baseline"
    assert back["Baseline"].notes == "via the old path"
    assert back["Baseline"].created_at == "2026-09-02T10:00:00+00:00"
    assert back["Baseline"].config.to_dict() == config.to_dict()
    assert module.scenario_signature(back["Baseline"]) == presets.preset_signature(
        saved["Baseline"]
    )

    # The bytes are what the new names write, so a file is portable in both
    # directions across the rename.
    assert path.read_text(encoding="utf-8") == presets.dump_presets_yaml(saved)

    # And the new loader reads the file the old writer produced.
    assert list(presets.load_presets(path)) == ["Baseline"]


def test_old_mutation_helpers_still_work(config: EngineConfig):
    with pytest.warns(DeprecationWarning):
        module = _reimport_scenarios()
    presets_map = {
        "A": module.Scenario("A", config),
        "B": module.Scenario("B", config),
    }
    renamed = module.rename_scenario(presets_map, "A", "A2")
    assert list(renamed) == ["A2", "B"]
    assert list(module.delete_scenario(renamed, "B")) == ["A2"]
