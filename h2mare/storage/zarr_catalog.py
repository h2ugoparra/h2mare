"""
Zarr catalog management for tracking processed zarr datasets.

This module provides functionality to create and maintain a Parquet catalog
of processed zarr files, enabling efficient dataset discovery and metadata queries.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.storage.zarr_scanner import ZarrDirectoryScanner
from h2mare.types import BBox, DateLike, DateRange, TimeResolution
from h2mare.utils.datetime_utils import normalize_date
from h2mare.utils.labels import create_label_from_dataset
from h2mare.utils.paths import resolve_store_path
from h2mare.validators import validate_time_resolution, validate_var_key


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


# ================ Convenience functions for quick access ==========================
def get_zarr_time_coverage(var_key: str) -> DateRange | None:
    catalog = ZarrCatalog(var_key)
    return catalog.get_time_coverage()


class ZarrCatalog:
    def __init__(
        self,
        var_key: str,
        *,
        time_resolution: TimeResolution = TimeResolution.YEAR,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        metadata_root: Optional[Path] = None,
        auto_refresh: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Manages a Parquet catalog of processed zarr datasets.
        The catalog tracks metadata about zarr files including spatial extent,
        temporal coverage, variables, and file locations.

        Args:
            var_key (str): Variable key that must exist in app_config.variables
            time_resolution: Temporal granularity for file storage ('year' or 'month'). Defaults to 'year'.
            app_config (Optional[AppConfig], optional): Application configuration. If None, loads from get_settings().
            store_root (Optional[Path]): Root directory for zarr files. If None, uses get_settings().STORE_ROOT or get_settings().ZARR_DIR.
            metadata_root (Optional[Path]): Root directory for catalog parquet files. If None, uses get_settings().METADATA_DIR.
            auto_refresh (bool): Automatically check for changes on access. Defaults to True.

        Raises:
            ValueError: If var_key not found in configuration or invalid period
            ValueError: If store_root doesn't exist

        Example:
            >>> catalog = ZarrCatalog("ssh")
        """
        # Load config
        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[var_key]

        self.time_resolution = validate_time_resolution(time_resolution)

        # Setup directories
        self.store_root = resolve_store_path(self.var_config, store_root)
        self.metadata_root = metadata_root or get_settings().METADATA_DIR

        self.auto_refresh = auto_refresh
        self.verbose = verbose
        self._scanner = ZarrDirectoryScanner(
            self.store_root, self.time_resolution, self.var_config, verbose=verbose
        )
        self._df_cache: Optional[pd.DataFrame] = None

        if auto_refresh:
            self.refresh()

    def __repr__(self) -> str:
        """String representation with useful info."""
        df = self.df
        time_cov = self.get_time_coverage()

        time_str = (
            f"{time_cov.start.date()} to {time_cov.end.date()}"
            if time_cov
            else "No data"
        )

        bbox = self.get_bbox()

        return (
            f"ZarrCatalog(\n"
            f"  var_key={self.var_key},\n"
            f"  time_resolution={self.time_resolution.value},\n"
            f"  files={df['path'].nunique() if not df.empty else 0},\n"
            f"  coverage={time_str},\n"
            f"  bbox={bbox.to_label() if bbox is not None else None},\n"
            f")"
        )

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
            if self.store_root.exists():
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

    def map_dates_to_paths(
        self, dates: Union[DateLike, Sequence[DateLike]]
    ) -> dict[str, list[pd.Timestamp]]:
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
        # Normalize input to list of timestamps
        date_list = normalize_date(dates)

        if not date_list:
            return {}

        result: dict[str, list[pd.Timestamp]] = defaultdict(list)

        date_list = [date_list] if isinstance(date_list, pd.Timestamp) else date_list
        for ts in date_list:
            matches = self._find_overlapping_files(ts, ts)
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
        seen: set[str] = set()
        unique: list[str] = []
        for p in matches["path"]:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def open_dataset(
        self,
        dates: DateLike | Sequence[DateLike] | None = None,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
        bbox: BBox | Sequence[float] | None = None,
        variables: str | Sequence[str] | None = None,
        chunks: dict | str | None = "auto",
    ) -> xr.Dataset:
        """
        Open zarr dataset(s) with flexible date and spatial selection.

        Supports two modes:
        1. Sparse dates: Provide `dates` for specific dates
        2. Date range: Provide `start_date` and/or `end_date`

        Args:
            dates: Specific dates to select (sparse mode)
            start_date: Start of date range (range mode)
            end_date: End of date range (range mode)
            bbox: Bounding box [xmin, ymin, xmax, ymax]
            variables: Variables to select (None = all)
            chunks: Chunking strategy for dask ('auto' or dict)

        Returns:
            Lazy xarray Dataset, or None if no data found

        Raises:
            ValueError: If neither dates nor start_date/end_date provided,
                       or if bbox format is invalid

        Example:
            >>> # Sparse dates
            >>> ds = catalog.open_dataset(dates=['2020-01-15', '2020-06-20'])
            >>>
            >>> # Date range
            >>> ds = catalog.open_dataset(
            ...     start_date='2020-01-01',
            ...     end_date='2020-12-31',
            ...     bbox=(-180, -90, 180, 90),
            ...     variables=['mld']
            ... )
        """
        # Validate inputs
        if dates is None and start_date is None and end_date is None:
            date_range = self.get_time_coverage()
            if date_range:
                start_date, end_date = date_range.start, date_range.end
            else:
                raise ValueError(
                    "Please provide sparse 'dates' or 'start_date/end_date' range"
                )

        if dates is not None and (start_date is not None or end_date is not None):
            raise ValueError(
                "Cannot use both 'dates' and 'start_date'/'end_date'. "
                "Use one mode or the other."
            )

        if bbox is not None:
            bbox = bbox if isinstance(bbox, BBox) else BBox.from_tuple(bbox)

        # Route to appropriate method
        if dates is not None:
            return self._open_sparse_dates(
                dates=dates,
                bbox=bbox,
                variables=variables,
                chunks=chunks,
            )
        else:
            return self._open_date_range(
                start_date=start_date,
                end_date=end_date,
                bbox=bbox,
                variables=variables,
                chunks=chunks,
            )

    def _open_sparse_dates(
        self,
        dates: DateLike | Sequence[DateLike],
        bbox: BBox | None,
        variables: str | Sequence[str] | None,
        chunks: dict | str | None,
    ) -> xr.Dataset:
        """Open dataset for specific sparse dates."""
        # Normalize dates
        date_list = normalize_date(dates)

        if not date_list:
            raise ValueError("No valid dates provided")

        # Get file paths
        path_mapping = self.map_dates_to_paths(date_list)

        if not path_mapping:
            raise FileNotFoundError(f"No zarr files contain dates: {date_list}")

        paths = list(path_mapping.keys())

        # Open datasets
        try:
            ds = xr.open_mfdataset(
                paths,
                engine="zarr",
                combine="by_coords",
                parallel=True,
                data_vars="minimal",
                coords="minimal",  # type: ignore[arg-type]
                compat="override",
                chunks=chunks,
                preprocess=lambda d: self._preprocess_dataset(d, bbox, variables),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open zarr files: {e}") from e

        # Normalize time coordinates
        ds = self._normalize_time(ds)

        # Select only requested dates
        if isinstance(date_list, pd.Timestamp):
            date_list = [date_list]
        requested_dates = pd.DatetimeIndex(date_list).normalize()
        available_dates = pd.DatetimeIndex(ds.time.values).normalize()

        valid_dates = requested_dates.intersection(available_dates)

        if len(valid_dates) == 0:
            raise FileNotFoundError(
                f"None of the requested dates found in dataset. "
                f"Requested: {requested_dates.tolist()}, "
                f"Available: {available_dates.tolist()}"
            )

        if len(valid_dates) < len(requested_dates):
            missing = requested_dates.difference(available_dates)
            self._log("warning", f"Missing dates: {missing.tolist()}")

        return ds.sel(time=valid_dates.tolist())

    def _open_date_range(
        self,
        start_date: DateLike | None,
        end_date: DateLike | None,
        bbox: BBox | None,
        variables: str | Sequence[str] | None,
        chunks: dict | str | None,
    ) -> xr.Dataset:
        """Open dataset for continuous date range."""

        # Get catalog
        df = self.df

        if df.empty:
            raise FileNotFoundError("Catalog is empty")

        # Default to full range if not specified
        start: pd.Timestamp = (
            df["start_date"].min()
            if start_date is None
            else pd.to_datetime(start_date).normalize()
        )
        end: pd.Timestamp = (
            df["end_date"].max()
            if end_date is None
            else pd.to_datetime(end_date).normalize()
        )

        if pd.isna(start) or pd.isna(end):
            raise ValueError(
                "Date range contains NaT — check catalog date columns for missing values"
            )

        # Warn if requested range extends beyond what the catalog covers
        available_start = pd.Timestamp(df["start_date"].min()).normalize()
        available_end = pd.Timestamp(df["end_date"].max()).normalize()

        if start_date is not None and start < available_start:
            self._log(
                "warning",
                f"[{self.var_key}] Requested start {start.date()} not available — "
                f"opening from {available_start.date()}",
            )
            start = available_start

        if end_date is not None and end > available_end:
            self._log(
                "warning",
                f"[{self.var_key}] Requested end {end.date()} not available — "
                f"opening until {available_end.date()}",
            )
            end = available_end

        # Get overlapping paths
        paths = self.get_paths_in_range(start, end)

        if not paths:
            raise FileNotFoundError(
                f"No zarr files found for range: {start.date()} to {end.date()}"
            )

        # Open datasets
        try:
            ds = xr.open_mfdataset(
                paths,
                engine="zarr",
                combine="by_coords",
                parallel=True,
                data_vars="minimal",
                coords="minimal",  # type: ignore[arg-type]
                compat="override",
                chunks=chunks,
                preprocess=lambda d: self._preprocess_dataset(d, bbox, variables),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open zarr files: {e}") from e

        # Normalize time
        ds = self._normalize_time(ds)

        # Select time range
        return ds.sel(time=slice(start, end))

    def build_file_path(
        self,
        ds: xr.Dataset,
        date_format: Literal["year", "date", "yearmonth"],
        store_root: Optional[Path] = None,
        name_key: Optional[str] = None,
    ) -> Path:
        """
        Build zarr file path as: store_root / {source}_{name_key}_{geo_extent}_{dates}.zarr

        name_key defaults to var_key; pass dataset_id_rep (or any string) to override.
        If store_root not provided, uses self.store_root.
        """
        label = create_label_from_dataset(ds, date_format=date_format)
        store_root = store_root or self.store_root
        key = name_key if name_key is not None else self.var_key
        return store_root / f"{self.var_config.source}_{key}_{label}.zarr"

    # ==================== Helper Methods ====================
    def _normalize_time(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Normalize time coordinates to midnight (00:00:00).

        Args:
            ds: Dataset with time coordinate

        Returns:
            Dataset with normalized time
        """
        if "time" not in ds.coords:
            return ds

        normalized_time = pd.to_datetime(ds["time"].values).normalize()
        return ds.assign_coords(time=normalized_time)

    def _preprocess_dataset(
        self,
        ds: xr.Dataset,
        bbox: BBox | None,
        variables: str | Sequence[str] | None,
    ) -> xr.Dataset:
        """
        Preprocess dataset: apply bbox and variable selection.

        This runs BEFORE datasets are combined in open_mfdataset.

        Args:
            ds: Input dataset
            bbox: Bounding box to apply
            variables: Variables to select

        Returns:
            Preprocessed dataset
        """
        # Select variables
        if variables is not None:
            var_list = [variables] if isinstance(variables, str) else list(variables)
            # Only select variables that exist
            available = set(ds.data_vars.keys())
            to_select = [v for v in var_list if v in available]

            if not to_select:
                self._log(
                    "warning",
                    f"None of requested variables found. "
                    f"Requested: {var_list}, Available: {list(available)}",
                )
            else:
                ds = ds[to_select]

        # Ensure lat is monotonically increasing (ERA5 comes north→south)
        if "lat" in ds.coords and ds.lat.values[0] > ds.lat.values[-1]:
            ds = ds.sortby("lat")

        # Apply spatial subset
        if bbox is not None:
            ds = self._apply_bbox(ds, bbox)

        return ds

    def _apply_bbox(
        self,
        ds: xr.Dataset,
        bbox: BBox,
    ) -> xr.Dataset:
        """
        Apply bounding box selection.

        Args:
            ds: Input dataset
            bbox: [xmin, ymin, xmax, ymax]

        Returns:
            Spatially subset dataset

        Raises:
            ValueError: If bbox format is invalid
        """
        # Determine coordinate names (support lat/lon or y/x)
        lat_coord = "lat" if "lat" in ds.coords else "y"
        lon_coord = "lon" if "lon" in ds.coords else "x"

        if lat_coord not in ds.coords or lon_coord not in ds.coords:
            self._log(
                "warning",
                f"Cannot apply bbox: missing coordinates. "
                f"Available: {list(ds.coords.keys())}",
            )
            return ds

        # Pad by one grid cell so a sub-cell bbox (e.g. a short geometry on a
        # coarse 0.5° grid) still captures surrounding cells. Without this, a
        # bbox falling between cell centers yields an empty slice downstream.
        lat_res = (
            float(abs(ds[lat_coord][1] - ds[lat_coord][0]))
            if ds[lat_coord].size > 1
            else 0.0
        )
        lon_res = (
            float(abs(ds[lon_coord][1] - ds[lon_coord][0]))
            if ds[lon_coord].size > 1
            else 0.0
        )

        try:
            return ds.sel(
                {
                    lat_coord: slice(bbox.ymin - lat_res, bbox.ymax + lat_res),
                    lon_coord: slice(bbox.xmin - lon_res, bbox.xmax + lon_res),
                }
            )
        except Exception as e:
            logger.error(f"Failed to apply bbox: {e}")
            return ds

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

    def get_vars_nonnull_end(self, var_names: Sequence[str]) -> dict[str, pd.Timestamp]:
        """
        Last date at which each variable holds a non-null value in the store.

        Unlike :meth:`get_var_time_coverage` — which reports the *file's* date
        span for any file listing the variable — this reflects where the variable
        actually has data. That distinction matters for compiled stores where a
        lagging variable is NaN-padded out to the global end by ``xr.merge``: the
        file end overstates its real coverage, whereas this method returns the
        last date with genuine data.

        Files are opened newest-first and scanning stops once every requested
        variable has been found, so the common case (all variables current)
        touches only the most recent file. Variables never found non-null — or
        absent from the store — are omitted from the result.

        Args:
            var_names: Variables to look up (typically one representative column
                per source var_key).

        Returns:
            Mapping of variable name to its last non-null date (normalised to
            midnight). Empty when the store has no data.
        """
        df = self.df
        if df.empty or "variables" not in df.columns or "path" not in df.columns:
            return {}

        files = df.sort_values("end_date", ascending=False).drop_duplicates(
            subset="path"
        )
        remaining = set(var_names)
        result: dict[str, pd.Timestamp] = {}

        for _, row in files.iterrows():
            if not remaining:
                break
            file_vars = _variables_list(row["variables"])
            cols_here = [c for c in remaining if c in file_vars]
            if not cols_here:
                continue
            try:
                ds = xr.open_zarr(row["path"], consolidated=False)
            except Exception as e:
                logger.warning(f"Could not open {row['path']} for non-null scan: {e}")
                continue
            try:
                if "time" not in ds.coords:
                    continue
                times = pd.to_datetime(ds["time"].values)
                # Reduce each variable to a per-time "has any data" mask, batched
                # into one lazy compute so the file is read in a single pass.
                lazy = {
                    c: ds[c].notnull().any(dim=[d for d in ds[c].dims if d != "time"])
                    for c in cols_here
                }
                mask_ds = xr.Dataset(lazy).compute()
                for c in cols_here:
                    m = mask_ds[c].values
                    if m.any():
                        result[c] = pd.Timestamp(times[m].max()).normalize()
                        remaining.discard(c)
            finally:
                ds.close()

        return result

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

    def get_bbox(self) -> BBox | None:
        """
        Get overall geographic extent (bbox) across all files.
        """
        bbox = self.var_config.bbox
        if bbox is None:
            return None
        if not isinstance(bbox, BBox):
            return BBox.from_tuple(bbox)
        return bbox

    def summary(self) -> dict:
        """
        Get summary statistics about the catalog.

        Returns:
            Dictionary with catalog statistics
        """
        df = self.df

        if df.empty:
            return {
                "num_files": 0,
                "time_coverage": None,
                "bbox": "No data",
                "period": self.time_resolution,
                "variables": set(),
                "total_timesteps": 0,
                "store_root": str(self.store_root),
                "catalog_path": str(self.catalog_path),
                "last_scanned": None,
            }

        time_cov = self.get_time_coverage()
        bbox = self.get_bbox()

        return {
            "num_files": df["path"].nunique(),
            "time_coverage": time_cov if time_cov is not None else "No data",
            "bbox": bbox if bbox is not None else "No data",
            "period": self.time_resolution,
            "variables": self.get_variables(),
            "total_timesteps": (
                df["num_timesteps"].sum().item()
                if "num_timesteps" in df.columns
                else None
            ),
            "store_root": str(self.store_root),
            "catalog_path": str(self.catalog_path),
            "last_scanned": (
                df["scanned_at"].max() if "scanned_at" in df.columns else None
            ),
        }

    # ==================== Migration Helpers ====================

    def backfill_provenance(self, rep_end_date: DateLike) -> int:
        """
        Retroactively write provenance sidecars for existing Zarr files that
        pre-date automatic tracking by Netcdf2Zarr.

        For each Zarr file in store_root that has no _prov.json sidecar:

        * Entire file falls within rep period  -> single rep entry.
        * Entire file falls after rep end date -> single nrt entry
          (only written when dataset_id_nrt is configured).
        * File spans the rep/nrt boundary    -> two entries split at
          rep_end_date / rep_end_date + 1 day.

        Call once after upgrading. The rep end date is obtainable without
        re-downloading data via CMEMSDownloader(var_key).get_rep_availability().end.

        Args:
            rep_end_date: Last date covered by the reprocessed (rep) dataset.

        Returns:
            Number of sidecar files written.

        Example::

            from h2mare.storage.zarr_catalog import ZarrCatalog
            from h2mare.downloader.cmems_downloader import CMEMSDownloader

            rep_end = CMEMSDownloader("sst").get_rep_availability().end
            n = ZarrCatalog("sst").backfill_provenance(rep_end)
            print(f"Written {n} sidecars")
        """
        rep_end = pd.to_datetime(rep_end_date).normalize()
        nrt_start = rep_end + pd.Timedelta(days=1)
        has_nrt = self.var_config.dataset_id_nrt is not None

        if not self.store_root.exists():
            self._log("warning", f"Store root not found: {self.store_root}")
            return 0

        import zarr

        written = 0
        for zarr_path in sorted(self.store_root.glob("*.zarr")):
            try:
                ds = xr.open_zarr(zarr_path, consolidated=False)
                already_set = ds.attrs.get("source_datasets") is not None
                z_start = pd.to_datetime(ds.time.min().compute().item()).normalize()
                z_end = pd.to_datetime(ds.time.max().compute().item()).normalize()
                ds.close()
            except Exception as e:
                self._log("warning", f"Could not read {zarr_path.name}: {e}")
                continue

            if already_set:
                self._log(
                    "debug",
                    f"Provenance already in zarr attrs, skipping: {zarr_path.name}",
                )
                continue

            records = []

            if z_end <= rep_end or not has_nrt:
                records.append(
                    {
                        "dataset_id": self.var_config.dataset_id_rep,
                        "dataset_type": "rep",
                        "start_date": z_start.strftime("%Y-%m-%d"),
                        "end_date": z_end.strftime("%Y-%m-%d"),
                    }
                )
            elif z_start > rep_end:
                records.append(
                    {
                        "dataset_id": self.var_config.dataset_id_nrt,
                        "dataset_type": "nrt",
                        "start_date": z_start.strftime("%Y-%m-%d"),
                        "end_date": z_end.strftime("%Y-%m-%d"),
                    }
                )
            else:
                records.append(
                    {
                        "dataset_id": self.var_config.dataset_id_rep,
                        "dataset_type": "rep",
                        "start_date": z_start.strftime("%Y-%m-%d"),
                        "end_date": rep_end.strftime("%Y-%m-%d"),
                    }
                )
                records.append(
                    {
                        "dataset_id": self.var_config.dataset_id_nrt,
                        "dataset_type": "nrt",
                        "start_date": nrt_start.strftime("%Y-%m-%d"),
                        "end_date": z_end.strftime("%Y-%m-%d"),
                    }
                )

            root = zarr.open_group(str(zarr_path), mode="r+")
            root.attrs["source_datasets"] = json.dumps(records)

            # Remove any legacy sidecar now that provenance lives in zarr attrs
            prov_file = zarr_path.parent / (zarr_path.stem + "_prov.json")
            if prov_file.exists():
                prov_file.unlink()

            self._log(
                "info",
                f"Wrote backfilled provenance for {zarr_path.name} ({len(records)} source(s))",
            )
            written += 1

        if written:
            self.reload()
            self._log(
                "info",
                f"Backfill complete: {written} zarr file(s) updated, catalog reloaded",
            )
        else:
            self._log("info", "Backfill complete: no files needed provenance")

        return written


if __name__ == "__main__":
    from h2mare.config import get_settings

    var_list = get_settings().get_available_var_keys()
    for var_key in var_list:
        repo = ZarrCatalog(var_key)
        print(repo)
    # stats = repo.summary()
    # print(stats)
    # print(repo.open_dataset(
    #    start_date="2020-01-01",
    #    end_date="2022-12-31"))
