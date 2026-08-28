"""An on-disk cache, so iterating on a strategy does not re-download the world.

A walk-forward sweep re-reads the same panel dozens of times. Without a cache
that is dozens of downloads, a rate limit, and a wait long enough that people
stop iterating. With one it is a single fetch and a Parquet read.

Three properties the design is built around:

* **Opt-in.** No directory, no cache. The library never writes to a user's
  filesystem because they called a function.
* **Keyed by what affects the data.** The key comes from
  :meth:`~optimization_engine.ingest.spec.IngestRequest.fingerprint`, which
  covers the universe, window, interval, fields, provider and currency — and
  deliberately excludes worker count and cache settings, which change how the
  fetch runs rather than what it returns.
* **Never poisonous.** A corrupt, unreadable or half-written entry is a cache
  miss, not an exception. Writes go to a temporary file and are moved into
  place atomically, so an interrupted run cannot leave a truncated entry that
  a later run trusts.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.panel import PricePanel, SeriesMeta

_LOG = logging.getLogger(__name__)

_MANIFEST = "manifest.json"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CacheEntry:
    """What a cache lookup found, for reporting."""

    key: str
    path: Path
    age_seconds: float
    fields: tuple[str, ...]

    @property
    def age_label(self) -> str:
        minutes = self.age_seconds / 60.0
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes:.0f} min ago"
        hours = minutes / 60.0
        if hours < 24:
            return f"{hours:.1f} h ago"
        return f"{hours / 24.0:.1f} days ago"


class PanelCache:
    """Reads and writes :class:`PricePanel` objects under a directory.

    Each entry is a folder named by the request fingerprint: one Parquet file
    per field plus a JSON manifest holding the provenance and the timestamp.
    Parquet keeps the dtypes and the DatetimeIndex intact, which CSV does not.
    """

    def __init__(self, directory: str | Path, ttl_seconds: int = 24 * 60 * 60) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = int(ttl_seconds)

    def path_for(self, key: str) -> Path:
        return self.directory / key

    def load(self, key: str) -> tuple[PricePanel, CacheEntry] | None:
        """Return a cached panel, or ``None`` on any miss.

        A miss includes: no entry, an entry older than the TTL, a manifest
        written by a different format version, and any read failure at all.
        Nothing here raises — a broken cache must degrade to a fetch.
        """
        folder = self.path_for(key)
        manifest_path = folder / _MANIFEST
        if not manifest_path.is_file():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("format_version", 0)) != _FORMAT_VERSION:
                return None

            age = time.time() - float(manifest["written_at"])
            if self.ttl_seconds and age > self.ttl_seconds:
                return None

            frames: dict[str, pd.DataFrame] = {}
            for name in manifest["fields"]:
                frame_path = folder / f"{name}.parquet"
                if not frame_path.is_file():
                    return None
                frames[name] = pd.read_parquet(frame_path)

            meta = {
                identifier: SeriesMeta(
                    identifier=identifier,
                    provider_symbol=record["provider_symbol"],
                    provider=record["provider"],
                    kind=F.InstrumentKind(record["kind"]),
                    currency=record.get("currency"),
                    name=record.get("name"),
                    exchange=record.get("exchange"),
                )
                for identifier, record in manifest.get("meta", {}).items()
            }
            panel = PricePanel.from_frames(frames, meta)
        except Exception as exc:  # corrupt entry, schema drift, partial write
            _LOG.debug("Ignoring unreadable cache entry %s: %s", key, exc)
            return None

        return panel, CacheEntry(
            key=key,
            path=folder,
            age_seconds=age,
            fields=tuple(manifest["fields"]),
        )

    def store(self, key: str, panel: PricePanel) -> bool:
        """Write a panel to the cache. Returns whether the write succeeded.

        Failures are logged and swallowed: a read-only directory or a full
        disk should slow the next run down, not fail this one.
        """
        folder = self.path_for(key)
        staging = folder.with_name(f".{folder.name}.tmp-{os.getpid()}")
        try:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)

            for name, frame in panel.frames.items():
                frame.to_parquet(staging / f"{name}.parquet")

            manifest = {
                "format_version": _FORMAT_VERSION,
                "written_at": time.time(),
                "fields": list(panel.frames),
                "identifiers": list(panel.identifiers),
                "meta": {
                    identifier: {
                        "provider_symbol": record.provider_symbol,
                        "provider": record.provider,
                        "kind": record.kind.value,
                        "currency": record.currency,
                        "name": record.name,
                        "exchange": record.exchange,
                    }
                    for identifier, record in panel.meta.items()
                },
            }
            (staging / _MANIFEST).write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

            # Replace atomically: a reader either sees the old entry or the
            # new one, never a directory being written into.
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
            staging.replace(folder)
            return True
        except Exception as exc:
            _LOG.warning("Could not cache panel %s: %s", key, exc)
            shutil.rmtree(staging, ignore_errors=True)
            return False

    def clear(self) -> int:
        """Delete every entry. Returns how many were removed."""
        if not self.directory.is_dir():
            return 0
        removed = 0
        for child in self.directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        return removed


__all__ = ["CacheEntry", "PanelCache"]
