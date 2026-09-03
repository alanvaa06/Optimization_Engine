"""Named configurations — save, load, rename, delete.

A `Preset` wraps a fully-specified `EngineConfig` plus a small amount of
human-facing metadata (name, notes, timestamps). Collections of presets
round-trip through YAML so they can be downloaded from the Streamlit app,
diffed in version control, and shared between users.

The serialization deliberately delegates to `EngineConfig.to_dict()` /
`EngineConfig.from_dict()` so the engine's existing schema stays the single
source of truth — adding a new field to `EngineConfig` automatically flows
through here.

**On the name.** This module was called ``scenarios`` until 0.7.0. It never
had anything to do with a *stress* scenario — a shock applied to a book, which
now lives in :mod:`optimization_engine.stress` — and the collision cost more
than the rename does. :mod:`optimization_engine.scenarios` still works, and
warns.

**On the file format.** The rename is an API rename, not a format change. The
YAML and JSON payloads still carry their entries under a ``scenarios`` key at
``schema_version`` 1, byte-for-byte what the previous release wrote, so a file
saved before the rename loads after it and a file saved after it loads in the
release before. A format that changed its key in the same breath as the module
would have made every stored preset unreadable by exactly one version in each
direction, silently in one of them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from optimization_engine.config import EngineConfig

#: Version of the on-disk payload. Bumping it is a deliberate breaking change:
#: :func:`load_presets_yaml` refuses anything else rather than guessing.
SCHEMA_VERSION = 1

#: Longest note kept, in characters. Longer notes are truncated on the way in
#: and on the way out, so a stored preset can never carry more than this.
NOTES_MAX_LEN = 2000

#: The key the entries live under in the payload. Unchanged by the module
#: rename on purpose — see the module docstring.
_PAYLOAD_KEY = "scenarios"

__all__ = [
    "NOTES_MAX_LEN",
    "Preset",
    "SCHEMA_VERSION",
    "config_signature",
    "delete_preset",
    "dump_presets_yaml",
    "load_presets",
    "load_presets_yaml",
    "now_iso",
    "preset_from_dict",
    "preset_signature",
    "preset_to_dict",
    "rename_preset",
    "save_presets",
]


@dataclass
class Preset:
    """A named optimizer configuration.

    Attributes:
        name: The label the preset is stored and looked up under. Unique
            within a collection.
        config: The full engine configuration this preset restores.
        notes: Free text, truncated to :data:`NOTES_MAX_LEN` characters on
            serialization.
        created_at: ISO-8601 UTC timestamp, seconds resolution, or ``""``
            when the preset was built in memory and never stamped.
        updated_at: ISO-8601 UTC timestamp of the last edit, same form.
    """

    name: str
    config: EngineConfig
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# Per-preset serialization
# ---------------------------------------------------------------------------


def preset_to_dict(p: Preset) -> dict[str, Any]:
    """One preset as a plain, YAML-serializable dict.

    Args:
        p: The preset to serialize.

    Returns:
        Its name, notes truncated to :data:`NOTES_MAX_LEN` characters,
        timestamps, and the full config dict. Round-trips through
        :func:`preset_from_dict`.
    """
    return {
        "name": str(p.name),
        "notes": _truncate_notes(p.notes),
        "created_at": p.created_at or "",
        "updated_at": p.updated_at or "",
        "config": p.config.to_dict(),
    }


def preset_from_dict(d: dict[str, Any]) -> Preset:
    """Rebuild a preset from its serialized form.

    Args:
        d: A mapping as produced by :func:`preset_to_dict`.

    Returns:
        The :class:`Preset`.

    Raises:
        ValueError: If ``name`` or ``config`` is missing.
        ConfigurationError: If the nested config carries an unknown key.
        LayerConfigurationError: If the nested config's layers are malformed.
    """
    if "name" not in d:
        raise ValueError("Preset entry is missing required key 'name'.")
    if "config" not in d:
        raise ValueError(f"Preset {d['name']!r} is missing required key 'config'.")
    return Preset(
        name=str(d["name"]),
        config=EngineConfig.from_dict(d["config"]),
        notes=_truncate_notes(d.get("notes") or ""),
        created_at=str(d.get("created_at") or ""),
        updated_at=str(d.get("updated_at") or ""),
    )


# ---------------------------------------------------------------------------
# Collection round-trip
# ---------------------------------------------------------------------------


def dump_presets_yaml(presets: dict[str, Preset]) -> str:
    """Serialize a name→Preset mapping into YAML text.

    Args:
        presets: The presets to write, keyed by name.

    Returns:
        YAML carrying a ``schema_version`` and a ``scenarios`` list, in the
        mapping's own order. The key is ``scenarios`` rather than ``presets``
        so the payload stays readable by the release before the rename; see
        the module docstring.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        _PAYLOAD_KEY: [preset_to_dict(presets[k]) for k in presets],
    }
    return yaml.safe_dump(payload, sort_keys=False)


def load_presets_yaml(text: str) -> dict[str, Preset]:
    """Parse a presets YAML string into a name→Preset mapping.

    Args:
        text: The YAML document.

    Returns:
        The presets, keyed by name, in the document's order.

    Raises:
        ValueError: If the payload is not a mapping, carries an unsupported
            ``schema_version``, has a ``scenarios`` key that is not a list or
            whose entries are not mappings, or names the same preset twice.
        ConfigurationError: If a nested config carries an unknown key.
        LayerConfigurationError: If a nested config's layers are malformed.
    """
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML payload must be a mapping.")
    _check_schema_version(data)
    raw_list = data.get(_PAYLOAD_KEY) or []
    if not isinstance(raw_list, list):
        raise ValueError(f"'{_PAYLOAD_KEY}' must be a list.")
    return _presets_from_entries(raw_list)


def save_presets(presets: dict[str, Preset], path: str | Path) -> None:
    """Persist presets to a YAML or JSON file.

    Args:
        presets: The presets to write, keyed by name.
        path: Destination. The format follows the extension — ``.yaml``,
            ``.yml`` or ``.json``. Parent directories are created.

    Raises:
        ValueError: If the extension is neither YAML nor JSON.
        OSError: If the file cannot be written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".yaml", ".yml"}:
        p.write_text(dump_presets_yaml(presets), encoding="utf-8")
    elif p.suffix.lower() == ".json":
        payload = {
            "schema_version": SCHEMA_VERSION,
            _PAYLOAD_KEY: [preset_to_dict(presets[k]) for k in presets],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported presets extension: {p.suffix}")


def load_presets(path: str | Path) -> dict[str, Preset]:
    """Load presets from a YAML or JSON file.

    Args:
        path: The file to read. The parser follows the extension.

    Returns:
        The presets, keyed by name, in the file's order.

    Raises:
        ValueError: If the extension is neither YAML nor JSON, the JSON
            payload is not an object, the ``schema_version`` is unsupported,
            or the same preset is named twice.
        FileNotFoundError: If the path does not exist.
        ConfigurationError: If a nested config carries an unknown key.
        LayerConfigurationError: If a nested config's layers are malformed.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        return load_presets_yaml(text)
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON payload must be an object.")
        _check_schema_version(data)
        return _presets_from_entries(data.get(_PAYLOAD_KEY) or [])
    raise ValueError(f"Unsupported presets extension: {p.suffix}")


# ---------------------------------------------------------------------------
# Mutation helpers (used by the Streamlit handlers and tests)
# ---------------------------------------------------------------------------


def rename_preset(
    presets: dict[str, Preset], old: str, new: str, *, touch: bool = True
) -> dict[str, Preset]:
    """Return a new ordered dict with ``old`` renamed to ``new``.

    Preserves insertion order, refuses collisions, and (by default) bumps
    ``updated_at``.

    Args:
        presets: The mapping to rename within. It is not mutated.
        old: The existing name.
        new: The new name.
        touch: Bump the renamed preset's ``updated_at`` timestamp to now.

    Returns:
        A new mapping with the rename applied, in the original order.

    Raises:
        KeyError: If ``old`` names no preset.
        ValueError: If ``new`` is empty, or already taken.
    """
    if old not in presets:
        raise KeyError(f"No preset named {old!r}")
    if new == old:
        return dict(presets)
    if not new:
        raise ValueError("New preset name cannot be empty.")
    if new in presets:
        raise ValueError(f"Preset {new!r} already exists.")
    out: dict[str, Preset] = {}
    for key, preset in presets.items():
        if key == old:
            renamed = _replace_preset(preset, name=new)
            if touch:
                renamed.updated_at = now_iso()
            out[new] = renamed
        else:
            out[key] = preset
    return out


def delete_preset(presets: dict[str, Preset], name: str) -> dict[str, Preset]:
    """Return a new dict with ``name`` removed.

    Args:
        presets: The mapping to delete from. It is not mutated.
        name: The preset to remove.

    Returns:
        A new mapping without it, in the original order.

    Raises:
        KeyError: If ``name`` names no preset.
    """
    if name not in presets:
        raise KeyError(f"No preset named {name!r}")
    return {k: v for k, v in presets.items() if k != name}


# ---------------------------------------------------------------------------
# Cache-key signatures
# ---------------------------------------------------------------------------


def config_signature(cfg: EngineConfig) -> str:
    """JSON signature of an EngineConfig, stable across dict insertion order.

    Args:
        cfg: The configuration to sign.

    Returns:
        Key-sorted JSON. Two configurations describing the same mandate
        produce the same string, whatever order their dicts were built in.
    """
    return json.dumps(cfg.to_dict(), sort_keys=True, default=str)


def preset_signature(preset: Preset) -> str:
    """A stable signature of the preset's configuration.

    Args:
        preset: The preset to sign.

    Returns:
        The JSON signature of its config, insensitive to dict insertion order,
        so two presets describing the same mandate compare equal. Notes, name
        and timestamps are deliberately outside it: renaming a preset does not
        make it a different one.
    """
    return config_signature(preset.config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (seconds resolution).

    Returns:
        e.g. ``"2026-09-02T10:00:00+00:00"``. Sub-second precision is dropped
        deliberately: these timestamps are read by people, not diffed.
    """
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _check_schema_version(data: dict[str, Any]) -> None:
    """Refuse a payload written by a schema this release does not understand.

    Args:
        data: The parsed top-level mapping.

    Raises:
        ValueError: If ``schema_version`` is anything but
            :data:`SCHEMA_VERSION`. Absent means 1, which is what every file
            written before the key existed carries.
    """
    version = data.get("schema_version", 1)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported presets schema_version={version!r}; expected {SCHEMA_VERSION}."
        )


def _presets_from_entries(raw_list: Any) -> dict[str, Preset]:
    """Turn the payload's entry list into a name→Preset mapping.

    Args:
        raw_list: The list of serialized presets.

    Returns:
        The presets, keyed by name, in the list's order.

    Raises:
        ValueError: If an entry is not a mapping, or two entries share a name —
            silently keeping the last would drop a stored configuration.
    """
    out: dict[str, Preset] = {}
    for entry in raw_list:
        if not isinstance(entry, dict):
            raise ValueError("Each preset entry must be a mapping.")
        preset = preset_from_dict(entry)
        if preset.name in out:
            raise ValueError(f"Duplicate preset name in payload: {preset.name!r}")
        out[preset.name] = preset
    return out


def _truncate_notes(notes: str | None) -> str:
    """Clip a note to :data:`NOTES_MAX_LEN` characters.

    Args:
        notes: The note, or ``None``.

    Returns:
        The note, truncated, or ``""``.
    """
    if not notes:
        return ""
    s = str(notes)
    return s if len(s) <= NOTES_MAX_LEN else s[:NOTES_MAX_LEN]


def _replace_preset(preset: Preset, **kwargs: Any) -> Preset:
    """Backport of :func:`dataclasses.replace` for older Python.

    Args:
        preset: The preset to copy.
        **kwargs: Fields to override.

    Returns:
        A new :class:`Preset`; the original is untouched.
    """
    fields: dict[str, Any] = {
        "name": preset.name,
        "config": preset.config,
        "notes": preset.notes,
        "created_at": preset.created_at,
        "updated_at": preset.updated_at,
    }
    fields.update(kwargs)
    return Preset(**fields)


# ---------------------------------------------------------------------------
# Deprecated aliases
#
# The pre-0.7.0 names, kept reachable for one release. They are deliberately
# absent from ``__all__``: ``from optimization_engine.presets import *`` gives
# the new vocabulary only, while every old call site keeps resolving. The
# deprecation *warning* is attached to the old module path
# (:mod:`optimization_engine.scenarios`), not to these bindings, so importing
# the package does not warn.
# ---------------------------------------------------------------------------

#: Deprecated alias of :class:`Preset`. Use :class:`Preset`.
Scenario = Preset
#: Deprecated alias of :func:`preset_to_dict`.
scenario_to_dict = preset_to_dict
#: Deprecated alias of :func:`preset_from_dict`.
scenario_from_dict = preset_from_dict
#: Deprecated alias of :func:`dump_presets_yaml`.
dump_scenarios_yaml = dump_presets_yaml
#: Deprecated alias of :func:`load_presets_yaml`.
load_scenarios_yaml = load_presets_yaml
#: Deprecated alias of :func:`save_presets`.
save_scenarios = save_presets
#: Deprecated alias of :func:`load_presets`.
load_scenarios = load_presets
#: Deprecated alias of :func:`rename_preset`.
rename_scenario = rename_preset
#: Deprecated alias of :func:`delete_preset`.
delete_scenario = delete_preset
#: Deprecated alias of :func:`preset_signature`.
scenario_signature = preset_signature
