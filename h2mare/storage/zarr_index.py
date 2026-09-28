"""
Catalog index for a Zarr store: the Parquet sidecar that tracks which files
exist, what they cover, and which variables they hold.

Owns the catalog DataFrame and its refresh lifecycle. Reading the Zarr data
itself lives in :class:`~h2mare.storage.zarr_catalog.ZarrCatalog`, which holds
one of these and delegates every index query to it.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from h2mare.storage.zarr_scanner import ZarrDirectoryScanner
from h2mare.types import BBox, DateLike, DateRange, DatesLike, FilePeriod
from h2mare.utils.datetime_utils import normalize_dates


def _variables_list(vs: Any) -> list[str]:
    """
    Normalise a catalog ``variables`` cell to a plain list.

    The column round-trips through Parquet, so a freshly scanned catalog holds
    Python ``list`` objects while one loaded from disk holds ``numpy.ndarray``.
    Membership checks must handle both (and missing/NaN cells).
    """
    if isinstance(vs, np.ndarray):
        return vs.tolist()
    if isinstance(vs, (list, tuple)):
        return list(vs)
    return []


class ZarrIndex:
    """
    The catalog DataFrame for one variable's Zarr store, plus its lifecycle.

    Holds the only mutable state involved in cataloguing — ``_df_cache`` — and
    is the sole writer of the catalog Parquet sidecar. Every method here either
    maintains that cache or answers a question from it; none of them open Zarr
    data.

    Note the cost of ``auto_refresh=True``: each ``.df`` access re-stats the
    store directory. Callers that read the catalog in a loop should snapshot
    ``.df`` once rather than accessing it per iteration.
    """

    def __init__(
        self,
        var_key: str,
        var_config: Any,
        *,
        store_root: Path,
        metadata_root: Path,
        file_period: FilePeriod,
        auto_refresh: bool = True,
        verbose: bool = False,
    ) -> None:
        self.var_key = var_key
        self.var_config = var_config
        self.store_root = store_root
        self.metadata_root = metadata_root
        self.file_period = file_period
        self.auto_refresh = auto_refresh
        self.verbose = verbose

        self._scanner = ZarrDirectoryScanner(
            self.store_root, self.file_period, self.var_config, verbose=verbose
        )
        self._df_cache: Optional[pd.DataFrame] = None

        if auto_refresh:
            self.refresh()

    def _log(self, level: str, msg: str) -> None:
        """Log at *level* when verbose, silent otherwise."""
        if self.verbose:
            getattr(logger, level)(msg)

    # ===============   IO Internal helpers  ================================

    @property
    def catalog_path(self) -> Path:
        """Path to the catalog parquet file."""
        filename = f"{self.var_key}_zarr_catalog.parquet"
        return self.metadata_root / filename

    def exists(self) -> bool:
        """Check if catalog file exists."""
        return self.catalog_path.exists()

    def _load_from_disk(self) -> pd.DataFrame:
        """
        Load catalog from parquet file.

        Returns:
            Catalog DataFrame, empty if file doesn't exist
        """
        if not self.catalog_path.exists():
            self._log("debug", f"Catalog file not found: {self.catalog_path}")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(self.catalog_path)
            if "dataset" not in df.columns:
                df["dataset"] = self.var_config.dataset_id_rep
            self._log(
                "debug",
                f"Loaded {self.var_key} catalog with {len(df)} entries from {self.catalog_path}",
            )
            return df
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            return pd.DataFrame()

    def _catalogs_another_root(self, df: pd.DataFrame) -> bool:
        """
        True when *df* lists files that do not sit directly under this store root.

        The sidecar is named for the ``var_key`` alone and lives under the
        project's ``METADATA_DIR``, so nothing ties it to the root it was built
        from. Repoint a variable — add a per-variable ``store_root`` to
        config.yaml, or pass ``--store-path`` — and the catalog left behind
        still holds the *previous* root's absolute paths.

        The ordinary staleness check cannot catch that. ``has_changes()``
        swallows the scanner's ``FileNotFoundError`` and reports False, and the
        disk-vs-catalog comparison below is gated on ``store_root.exists()``, so
        a root that is unmounted or not yet created would quietly serve the old
        location's files. Comparing the recorded parent against the current root
        is what makes the move visible.
        """
        if df.empty or "path" not in df.columns:
            return False

        root = self.store_root.resolve()
        # The scanner only ever records files it globbed directly in the root,
        # so parent equality is the exact invariant — not mere containment.
        return any(Path(p).resolve().parent != root for p in df["path"].unique())

    def _scan_and_build(self) -> pd.DataFrame:
        """Scan zarr files via the scanner and build a fresh catalog DataFrame."""
        self._log("info", f"Scanning {self.store_root}")
        records = self._scanner.scan()

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("start_date")
        self._save_catalog(df)
        return df

    def _save_catalog(self, df: pd.DataFrame) -> None:
        """
        Save catalog DataFrame to parquet.

        Args:
            df: Catalog DataFrame to save
        """
        # Ensure directory exists
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to parquet
        df.to_parquet(self.catalog_path, index=False)
        self._log(
            "info", f"Saved catalog with {len(df)} entries to {self.catalog_path}"
        )

    def has_changes(self) -> bool:
        """Check if the store directory has changed since the last scan."""
        return self._scanner.has_changes()

    def refresh(self, force: bool = False) -> pd.DataFrame:
        """
        Refresh catalog if needed.

        Args:
            force: Force rescan even if no changes detected

        Returns:
            Updated catalog DataFrame

        Logic:
            1. If force=True: rescan
            2. If changes detected: rescan
            3. If catalog file exists: load from disk
            4. Otherwise: rescan
        """
        # Check if rescan needed
        should_rescan = force or self.has_changes()

        if should_rescan:
            self._df_cache = self._scan_and_build()
        elif self._df_cache is None:
            self._df_cache = self._load_from_disk()
            if self._catalogs_another_root(self._df_cache):
                self._log(
                    "debug",
                    f"[{self.var_key}] Catalog was built for a different store root "
                    f"— rescanning {self.store_root}",
                )
                self._df_cache = self._scan_and_build()
            elif self.store_root.exists():
                if self._df_cache.empty:
                    self._log("debug", "No catalog file found, performing initial scan")
                    self._df_cache = self._scan_and_build()
                else:
                    # Detect files added to disk since the catalog was last written
                    disk_files = {p.name for p in self.store_root.glob("*.zarr")}
                    catalog_files = {
                        Path(p).name for p in self._df_cache["path"].unique()
                    }
                    if disk_files != catalog_files:
                        self._log(
                            "debug",
                            f"[{self.var_key}] Catalog is stale "
                            f"(disk={len(disk_files)}, catalog={len(catalog_files)}) — rescanning",
                        )
                        self._df_cache = self._scan_and_build()
        # else: use existing cache
        return self._df_cache

    @property
    def df(self) -> pd.DataFrame:
        """
        Get catalog DataFrame.

        Behavior:
            - If auto_refresh=True: Check for changes on each access
            - If auto_refresh=False: Load once and cache forever

        Returns:
            Catalog DataFrame
        """
        if self.auto_refresh:
            # Always check for changes
            return self.refresh()
        else:
            # Load once, cache forever
            if self._df_cache is None:
                self._df_cache = self.refresh()
            return self._df_cache

    def reload(self) -> pd.DataFrame:
        """Force reload from disk or rescan."""
        self._df_cache = None
        self._scanner.reset()
        return self.refresh(force=True)

    def get_change_summary(self) -> dict[str, Any]:
        """Return a summary of added / removed / modified zarr files."""
        return self._scanner.get_change_summary()

    # ==================== Query Methods ====================
    def _find_overlapping_files(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Return catalog rows whose date range overlaps [start, end], sorted by start_date."""
        df = self.df
        if df.empty:
            self._log("warning", "Catalog is empty, no paths available")
            return pd.DataFrame()

        return df[(df["start_date"] <= end) & (df["end_date"] >= start)].sort_values(
            "start_date"
        )

    def map_dates_to_paths(self, dates: DatesLike) -> dict[str, list[pd.Timestamp]]:
        """
        Map zarr file paths to their corresponding dates.

        Note: Finds files containing each date, not a date range.
        For ranges, use get_paths_for_range() or pass a list of all dates.

        Args:
            dates: Single date or sequence of dates

        Returns:
            Dictionary mapping file paths to lists of matched dates

        Example:
            >>> catalog.get_paths(['1998-01-15', '2020-06-20'])
            {'/path/1998.zarr': [Timestamp('1998-01-15')],
             '/path/2020.zarr': [Timestamp('2020-06-20')]}
        """
        date_list = normalize_dates(dates)
        if not date_list:
            return {}

        result: dict[str, list[pd.Timestamp]] = defaultdict(list)

        # Snapshot the catalog once: with auto_refresh=True every .df access
        # re-stats the store directory, which per-date adds up to one directory
        # sweep per extraction sample.
        df = self.df
        if df.empty:
            self._log("warning", "Catalog is empty, no paths available")
            return {}
        df = df.sort_values("start_date")

        for ts in date_list:
            matches = df[(df["start_date"] <= ts) & (df["end_date"] >= ts)]
            if matches.empty:
                self._log("debug", f"No zarr file contains date: {ts}")
                continue
            result[str(matches.iloc[0]["path"])].append(ts)

        return dict(result)  # Convert defaultdict to regular dict

    def get_paths_in_range(
        self,
        start_date: DateLike,
        end_date: DateLike,
    ) -> list[str]:
        """
        Get all zarr file paths that overlap with a date range.

        Args:
            start_date: Range start
            end_date: Range end

        Returns:
            List of file paths, sorted by start date

        Example:
            >>> catalog.get_paths_for_range('2020-01-01', '2020-12-31')
            ['/path/2020.zarr']
        """
        start = pd.to_datetime(start_date).normalize()
        end = pd.to_datetime(end_date).normalize()

        # Find overlapping files
        matches = self._find_overlapping_files(start, end)
        if matches.empty:
            self._log(
                "warning",
                f"[{self.var_key}] No zarr files overlap range {start.date()} to {end.date()}",
            )
            return []

        # Sort by start date and return unique paths (a file with rep+nrt rows
        # would otherwise appear twice and break xr.open_mfdataset)
        matches = matches.sort_values("start_date")
        return list(dict.fromkeys(matches["path"]))

    # ==================== Metadata Queries ====================
    def get_var_time_coverage(self, var_name: str) -> DateRange | None:
        """
        Time coverage for zarr files that contain *var_name* as a data variable.

        Returns ``None`` if no file in the catalog contains that variable.
        """
        df = self.df
        if df.empty or "variables" not in df.columns:
            return None
        mask = df["variables"].apply(lambda vs: var_name in _variables_list(vs))
        sub = df[mask]
        if sub.empty:
            return None
        return DateRange(sub["start_date"].min(), sub["end_date"].max())

    def get_variables(self) -> set[str]:
        """
        Get all unique variables across all zarr files.

        Returns:
            Set of variable names, empty if none found

        Example:
            >>> catalog.get_variables()
            {'temperature', 'salinity', 'velocity_u', 'velocity_v'}
        """
        # Use cached catalog if available
        if not self.df.empty and "variables" in self.df.columns:
            # Combine variables from catalog metadata
            all_vars = set()
            for var_list in self.df["variables"].dropna():
                all_vars.update(var_list)
            return all_vars

        # Fallback: scan files directly
        return self._scanner.scan_variables()

    def get_time_coverage(self) -> DateRange | None:
        """
        Get overall time coverage across all files.

        Returns:
            DateRange(earliest_start, latest_end) or None if no data
        """
        df = self.df

        if df.empty:
            return None

        return DateRange(
            start=df["start_date"].min(),
            end=df["end_date"].max(),
        )

    def get_store_bbox(self) -> BBox | None:
        """
        Union of the geographic extents recorded for the catalogued files.

        This is what the store actually holds, as scanned from each file's
        lon/lat bounds — distinct from ``ZarrCatalog.get_bbox()``, which
        reports the bbox *requested* in config.yaml.

        Returns:
            BBox spanning every file, or None if the catalog is empty, lacks
            the extent columns, or holds a degenerate extent (a single-cell
            store, which BBox rejects).
        """
        df = self.df

        if df.empty or not {"xmin", "ymin", "xmax", "ymax"} <= set(df.columns):
            return None

        bounds = (
            df["xmin"].min(),
            df["ymin"].min(),
            df["xmax"].max(),
            df["ymax"].max(),
        )
        if any(pd.isna(b) for b in bounds):
            return None

        try:
            return BBox(*(float(b) for b in bounds))
        except ValueError:
            return None
