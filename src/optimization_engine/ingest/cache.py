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
  miss, not an exception. An interrupted run cannot leave a truncated entry
  that a later run trusts.

One entry is one file, and that is load-bearing rather than incidental. The
obvious layout — a directory per entry holding a Parquet per field — cannot be
published atomically: ``rename`` onto a directory that already exists fails,
so publishing means removing the old one first, and two writers racing through
that gap leave one of them with ``Directory not empty`` and readers with a
window where the entry does not exist at all. ``os.replace`` on a *file* has
neither problem. So an entry is a Zip holding exactly what the directory would
have held, written to a temporary name beside it and moved into place in one
step: a reader sees the old entry or the new one, a second writer simply wins,
and a reader already mid-read keeps the file it opened.

The Zip is stored uncompressed — Parquet has already done that work, and
deflating it again costs time to save almost nothing.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from optimization_engine.ingest import fields as F
from optimization_engine.ingest.panel import PricePanel, SeriesMeta

_LOG = logging.getLogger(__name__)

_MANIFEST = "manifest.json"

#: Bumped when the on-disk layout changes. An entry written by a different
#: version is a miss, never a parse attempt — version 1 was a directory per
#: entry, which is why version 2 exists.
_FORMAT_VERSION = 2

#: Suffix that marks a complete entry. A file only carries it at the instant
#: it is whole, so a half-written one can never be mistaken for a hit.
_SUFFIX = ".panel"


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

    Each entry is one Zip file named by the request fingerprint, holding a
    Parquet per field plus a JSON manifest with the provenance and the write
    time. Parquet keeps the dtypes and the ``DatetimeIndex`` intact, which CSV
    does not; the single file is what makes publishing atomic.
    """

    def __init__(self, directory: str | Path, ttl_seconds: int = 24 * 60 * 60) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = int(ttl_seconds)

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}{_SUFFIX}"

    def load(self, key: str) -> tuple[PricePanel, CacheEntry] | None:
        """Return a cached panel, or ``None`` on any miss.

        A miss includes: no entry, an entry older than the TTL, one written by
        a different format version, and any read failure at all. Nothing here
        raises — a broken cache must degrade to a fetch, never to a stack
        trace, because the fetch is always available and always correct.
        """
        path = self.path_for(key)
        if not path.is_file():
            return None

        try:
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read(_MANIFEST).decode("utf-8"))
                if int(manifest.get("format_version", 0)) != _FORMAT_VERSION:
                    return None

                age = time.time() - float(manifest["written_at"])
                if self.ttl_seconds and age > self.ttl_seconds:
                    return None

                frames = {
                    name: pd.read_parquet(io.BytesIO(archive.read(f"{name}.parquet")))
                    for name in manifest["fields"]
                }

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
            path=path,
            age_seconds=age,
            fields=tuple(manifest["fields"]),
        )

    def store(self, key: str, panel: PricePanel) -> bool:
        """Write a panel to the cache. Returns whether the write succeeded.

        The entry is built under a temporary name in the same directory — so
        the move below stays on one filesystem — and then moved into place
        with a single :func:`os.replace`. That call is atomic, which is what
        lets two runs fetch the same request concurrently without either of
        them failing and without a reader ever seeing a partial entry.

        Failures are logged and swallowed: a read-only directory or a full
        disk should slow the next run down, not fail this one.
        """
        target = self.path_for(key)
        handle, staging = -1, ""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            handle, staging = tempfile.mkstemp(
                dir=self.directory, prefix=f".{key}.", suffix=".tmp"
            )
            with os.fdopen(handle, "wb") as raw:
                handle = -1  # now owned by the file object
                self._write_archive(raw, panel)

            # One atomic step. A concurrent writer of the same key simply
            # wins; a reader holding the old file keeps reading it.
            os.replace(staging, target)
            return True
        except Exception as exc:
            _LOG.warning("Could not cache panel %s: %s", key, exc)
            if handle != -1:
                os.close(handle)
            if staging:
                # The temporary name never carried the entry suffix, so even
                # if this cleanup fails the leftover cannot be read as a hit.
                with contextlib.suppress(OSError):
                    os.unlink(staging)
            return False

    @staticmethod
    def _write_archive(stream, panel: PricePanel) -> None:
        """Serialize a panel into an open binary stream as a Zip."""
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
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(_MANIFEST, json.dumps(manifest, indent=2))
            for name, frame in panel.frames.items():
                buffer = io.BytesIO()
                frame.to_parquet(buffer)
                archive.writestr(f"{name}.parquet", buffer.getvalue())

    def clear(self) -> int:
        """Delete every entry. Returns how many were removed.

        Sweeps up three things: current entries, the directories version 1
        wrote, and any temporary file an interrupted run left behind.
        """
        if not self.directory.is_dir():
            return 0
        removed = 0
        for child in self.directory.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
            elif child.suffix == _SUFFIX or child.name.endswith(".tmp"):
                with contextlib.suppress(OSError):
                    child.unlink()
                    removed += 1
        return removed


__all__ = ["CacheEntry", "PanelCache"]
