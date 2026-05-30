"""Filesystem scanning layer for Zarr stores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.types import TimeResolution


@dataclass
class DirectoryState:
    """Snapshot of zarr file names and their mtimes in a store directory."""

    files: dict[str, float]  # filename -> mtime
    count: int
    last_modified: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirectoryState):
            return NotImplemented
        return self.files == other.files

    def __hash__(self) -> int:
        file_str = ",".join(f"{k}:{v}" for k, v in sorted(self.files.items()))
        return int(hashlib.md5(file_str.encode()).hexdigest()[:16], 16)


if TYPE_CHECKING:
    from h2mare.models import KeyVarConfigEntry


class ZarrDirectoryScanner:
    """
    Filesystem-only scanning layer for a single Zarr store directory.

    Responsible for:
    - Detecting directory changes via mtime snapshots (``has_changes``)
    - Scanning Zarr files and extracting metadata records (``scan``)
    - Listing variable names without opening full datasets (``scan_variables``)

    ``ZarrCatalog`` owns the persistent catalog and query interface;
    this class handles only the I/O needed to build it.
    """

    def __init__(
        self,
        store_root: Path,
        time_resolution: TimeResolution,
        var_config: "KeyVarConfigEntry",
        verbose: bool = False,
    ) -> None:
        self.store_root = store_root
        self.time_resolution = time_resolution
        self.var_config = var_config
        self.verbose = verbose
        self._cached_state: Optional[DirectoryState] = None

    def _log(self, level: str, msg: str) -> None:
        """Log at *level* when verbose, silent otherwise."""
        if self.verbose:
            getattr(logger, level)(msg)

    # ------------------------------------------------------------------
    # Directory state
    # ------------------------------------------------------------------

    def _get_state(self) -> DirectoryState:
        """Return current mtime snapshot of all *.zarr entries in store_root."""
        if not self.store_root.exists():
            raise FileNotFoundError(f"Store directory not found: {self.store_root}")

        files: dict[str, float] = {}
        try:
            for entry in self.store_root.iterdir():
                if entry.is_dir() and entry.name.endswith(".zarr"):
                    files[entry.name] = entry.stat().st_mtime
        except PermissionError as e:
            self._log("warning", f"Permission denied scanning {self.store_root}: {e}")
            return DirectoryState(files={}, count=0, last_modified=0.0)

        if not files:
            return DirectoryState(files={}, count=0, last_modified=0.0)

        return DirectoryState(
            files=files,
            count=len(files),
            last_modified=max(files.values()),
        )

    def has_changes(self) -> bool:
        """Return True if the directory has changed since the last call."""
        try:
            current = self._get_state()
        except FileNotFoundError:
            self._log("debug", f"Store directory not found: {self.store_root}")
            return False

        if self._cached_state is None:
            self._cached_state = current
            return False

        changed = current != self._cached_state
        if changed:
            self._log(
                "info",
                f"Changes detected in {self.store_root}: "
                f"{self._cached_state.count} → {current.count} files",
            )
            self._cached_state = current
        return changed

    def get_change_summary(self) -> dict:
        """Return a dict with added / removed / modified file names."""
        try:
            current = self._get_state()
        except FileNotFoundError:
            return {"error": "Store directory not found"}

        if self._cached_state is None:
            return {
                "added": list(current.files.keys()),
                "removed": [],
                "modified": [],
                "total": current.count,
            }

        old = set(self._cached_state.files.keys())
        new = set(current.files.keys())
        modified = {
            f for f in old & new if self._cached_state.files[f] != current.files[f]
        }
        return {
            "added": sorted(new - old),
            "removed": sorted(old - new),
            "modified": sorted(modified),
            "total": current.count,
        }

    def reset(self) -> None:
        """Clear the cached directory state (forces a full re-check next call)."""
        self._cached_state = None

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def scan(self) -> list[dict]:
        """
        Open every *.zarr in store_root and return a list of metadata records.

        Returns one dict per source dataset inside each file (two when a
        provenance sidecar distinguishes rep from nrt data).
        """
        if not self.store_root.exists():
            self._log("warning", f"Store directory not found: {self.store_root}")
            return []

        zarr_files = sorted(self.store_root.glob("*.zarr"))
        records: list[dict] = []

        for zarr_path in zarr_files:
            try:
                record_list = self._extract_zarr_metadata(zarr_path)
                if record_list:
                    records.extend(record_list)
            except (OSError, RuntimeError) as e:
                self._log("warning", f"Failed to read {zarr_path.name}: {e}")

        self._log("debug", f"Scan complete: {len(zarr_files)} zarr files read")
        return records

    def scan_variables(self) -> set[str]:
        """Return all variable names found across every *.zarr in store_root."""
        if not self.store_root.exists():
            self._log("warning", f"Store directory not found: {self.store_root}")
            return set()

        zarr_files = sorted(self.store_root.glob("*.zarr"))
        if not zarr_files:
            return set()

        all_vars: set[str] = set()
        for zarr_path in zarr_files:
            try:
                ds = xr.open_zarr(zarr_path, decode_cf=False)
                all_vars.update(ds.data_vars.keys())
                ds.close()
            except Exception as e:
                self._log("warning", f"Could not read variables from {zarr_path.name}: {e}")
        return all_vars

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_zarr_metadata(self, zarr_path: Path) -> list[dict]:
        """
        Extract metadata from a single zarr file.

        Returns one dict per source dataset. When a provenance sidecar
        ``{stem}_prov.json`` exists (or ``source_datasets`` is in zarr attrs),
        the result contains one entry per recorded dataset_id. Falls back to a
        single row using ``dataset_id_rep`` when no provenance is present.
        """
        try:
            ds = xr.open_zarr(zarr_path, consolidated=False)

            time_min = pd.to_datetime(ds.time.min().compute().item()).normalize()
            time_max = pd.to_datetime(ds.time.max().compute().item()).normalize()

            xmin = ds.lon.min().compute().item()
            ymin = ds.lat.min().compute().item()
            xmax = ds.lon.max().compute().item()
            ymax = ds.lat.max().compute().item()

            period_value = (
                time_min.year
                if self.time_resolution == TimeResolution.YEAR
                else f"{time_min.year}-{time_min.month:02d}"
            )

            base = {
                "path": str(zarr_path),
                "filename": zarr_path.name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "variables": list(ds.data_vars.keys()),
                "file_mtime": zarr_path.stat().st_mtime,
                "scanned_at": datetime.now(),
                "period": period_value,
                "start_date": time_min,
                "end_date": time_max,
                "num_timesteps": len(ds.time),
            }

            raw = ds.attrs.get("source_datasets")
            if raw is None:
                prov_file = zarr_path.parent / (zarr_path.stem + "_prov.json")
                if prov_file.exists():
                    raw = prov_file.read_text()

            if raw is not None:
                sources = json.loads(raw)
                records = []
                for i, src in enumerate(sources):
                    p_start = pd.to_datetime(src["start_date"]).normalize()
                    p_end = pd.to_datetime(src["end_date"]).normalize()
                    rec_start = time_min if i == 0 else max(p_start, time_min)
                    rec_end = (
                        time_max if i == len(sources) - 1 else min(p_end, time_max)
                    )
                    if rec_start > rec_end:
                        continue
                    n_ts = len(ds.sel(time=slice(rec_start, rec_end)).time)
                    records.append(
                        {
                            **base,
                            "dataset": src["dataset_id"],
                            "start_date": rec_start,
                            "end_date": rec_end,
                            "num_timesteps": n_ts,
                        }
                    )
                if records:
                    return records

            return [{**base, "dataset": self.var_config.dataset_id_rep}]

        except (OSError, KeyError, ValueError, RuntimeError) as e:
            logger.error(f"Error extracting metadata from {zarr_path}: {e}")
            return []
