"""Named scenarios for the optimization engine.

A `Scenario` wraps a fully-specified `EngineConfig` plus a small amount of
human-facing metadata (name, notes, timestamps). Collections of scenarios
round-trip through YAML so they can be downloaded from the Streamlit app,
diffed in version control, and shared between users.

The serialization deliberately delegates to `EngineConfig.to_dict()` /
`EngineConfig.from_dict()` so the engine's existing schema stays the single
source of truth — adding a new field to `EngineConfig` automatically flows
through here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from optimization_engine.config import EngineConfig

SCHEMA_VERSION = 1
NOTES_MAX_LEN = 2000


@dataclass
class Scenario:
    """A named optimizer configuration."""

    name: str
    config: EngineConfig
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Per-scenario serialization
# ---------------------------------------------------------------------------


def scenario_to_dict(s: Scenario) -> dict[str, Any]:
    return {
        "name": str(s.name),
        "notes": _truncate_notes(s.notes),
        "created_at": s.created_at or "",
        "updated_at": s.updated_at or "",
        "config": s.config.to_dict(),
    }


def scenario_from_dict(d: dict[str, Any]) -> Scenario:
    if "name" not in d:
        raise ValueError("Scenario entry is missing required key 'name'.")
    if "config" not in d:
        raise ValueError(f"Scenario {d['name']!r} is missing required key 'config'.")
    return Scenario(
        name=str(d["name"]),
        config=EngineConfig.from_dict(d["config"]),
        notes=_truncate_notes(d.get("notes") or ""),
        created_at=str(d.get("created_at") or ""),
        updated_at=str(d.get("updated_at") or ""),
    )


# ---------------------------------------------------------------------------
# Collection round-trip
# ---------------------------------------------------------------------------


def dump_scenarios_yaml(scenarios: dict[str, Scenario]) -> str:
    """Serialize a name→Scenario mapping into YAML text."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scenarios": [scenario_to_dict(scenarios[k]) for k in scenarios],
    }
    return yaml.safe_dump(payload, sort_keys=False)


def load_scenarios_yaml(text: str) -> dict[str, Scenario]:
    """Parse a scenarios YAML string into a name→Scenario mapping."""
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML payload must be a mapping.")
    version = data.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported scenarios schema_version={version!r}; expected {SCHEMA_VERSION}."
        )
    raw_list = data.get("scenarios") or []
    if not isinstance(raw_list, list):
        raise ValueError("'scenarios' must be a list.")

    out: dict[str, Scenario] = {}
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError("Each scenario entry must be a mapping.")
        scn = scenario_from_dict(entry)
        if scn.name in out:
            raise ValueError(f"Duplicate scenario name in payload: {scn.name!r}")
        out[scn.name] = scn
    return out


def save_scenarios(scenarios: dict[str, Scenario], path: str | Path) -> None:
    """Persist scenarios to a YAML or JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".yaml", ".yml"}:
        p.write_text(dump_scenarios_yaml(scenarios), encoding="utf-8")
    elif p.suffix.lower() == ".json":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scenarios": [scenario_to_dict(scenarios[k]) for k in scenarios],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported scenarios extension: {p.suffix}")


def load_scenarios(path: str | Path) -> dict[str, Scenario]:
    """Load scenarios from a YAML or JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        return load_scenarios_yaml(text)
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON payload must be an object.")
        version = data.get("schema_version", 1)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported scenarios schema_version={version!r}; expected {SCHEMA_VERSION}."
            )
        out: dict[str, Scenario] = {}
        for entry in data.get("scenarios") or []:
            scn = scenario_from_dict(entry)
            if scn.name in out:
                raise ValueError(f"Duplicate scenario name in payload: {scn.name!r}")
            out[scn.name] = scn
        return out
    raise ValueError(f"Unsupported scenarios extension: {p.suffix}")


# ---------------------------------------------------------------------------
# Mutation helpers (used by the Streamlit handlers and tests)
# ---------------------------------------------------------------------------


def rename_scenario(
    scenarios: dict[str, Scenario], old: str, new: str, *, touch: bool = True
) -> dict[str, Scenario]:
    """Return a new ordered dict with ``old`` renamed to ``new``.

    Preserves insertion order, refuses collisions, and (by default) bumps
    ``updated_at``.
    """
    if old not in scenarios:
        raise KeyError(f"No scenario named {old!r}")
    if new == old:
        return dict(scenarios)
    if not new:
        raise ValueError("New scenario name cannot be empty.")
    if new in scenarios:
        raise ValueError(f"Scenario {new!r} already exists.")
    out: dict[str, Scenario] = {}
    for key, scn in scenarios.items():
        if key == old:
            renamed = _replace_scenario(scn, name=new)
            if touch:
                renamed.updated_at = now_iso()
            out[new] = renamed
        else:
            out[key] = scn
    return out


def delete_scenario(scenarios: dict[str, Scenario], name: str) -> dict[str, Scenario]:
    """Return a new dict with ``name`` removed."""
    if name not in scenarios:
        raise KeyError(f"No scenario named {name!r}")
    return {k: v for k, v in scenarios.items() if k != name}


# ---------------------------------------------------------------------------
# Cache-key signatures
# ---------------------------------------------------------------------------


def config_signature(cfg: EngineConfig) -> str:
    """JSON signature of an EngineConfig, stable across dict insertion order."""
    return json.dumps(cfg.to_dict(), sort_keys=True, default=str)


def scenario_signature(scn: Scenario) -> str:
    return config_signature(scn.config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (seconds resolution)."""
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _truncate_notes(notes: str | None) -> str:
    if not notes:
        return ""
    s = str(notes)
    return s if len(s) <= NOTES_MAX_LEN else s[:NOTES_MAX_LEN]


def _replace_scenario(s: Scenario, **kwargs) -> Scenario:
    """Backport of dataclasses.replace for older Python."""
    fields = {
        "name": s.name,
        "config": s.config,
        "notes": s.notes,
        "created_at": s.created_at,
        "updated_at": s.updated_at,
    }
    fields.update(kwargs)
    return Scenario(**fields)
