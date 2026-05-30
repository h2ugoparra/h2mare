"""
Create h2ds zarr files
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Literal, Optional

import ephem
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.models import SYSTEM_VAR_KEYS
from h2mare.storage.coverage import get_store_coverage, split_time_range
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import chunk_dataset
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox, DateLike, DateRange, TimeResolution
from h2mare.utils.date_range import resolve_date_range
from h2mare.utils.datetime_utils import normalize_date
from h2mare.utils.spatial import GridBuilder
from h2mare.validators import validate_time_resolution, validate_var_key

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Output grid resolution in degrees — matches the standard CMEMS/Copernicus
# 0.25° daily grid used across all compiled h2ds variables.
DX = 0.25
DY = 0.25


def calculate_moon_phase(
    lat: float, lon: float, dates: pd.DatetimeIndex
) -> list[float]:
    """
    Calculate moon ilumination using ephem library

    Args:
        lat (float): latitude of observation
        lon (float): longitude of observation
        dates (pd.DatetimeIndex): time index values for extraction

    Returns:
        list[float]: list of lunar ilumination values for each date values
    """
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)

    phases = []
    for date in dates:
        observer.date = date
        moon = ephem.Moon(observer)
        phases.append(moon.phase)
    return phases


def postprocess_sst_fdist(ds: xr.Dataset, var_name: str = "sst_fdist") -> xr.Dataset:
    """
    Clip sst_fdist data because interp_like gives negative values
    """
    if var_name in ds:
        ds[var_name] = ds[var_name].clip(min=0)
    return ds


class Compiler:
    def __init__(
        self,
        var_key: str = "h2ds",
        app_config: Optional[AppConfig] = None,
        remote_store_root: Optional[Path] = None,
        local_store_root: Optional[Path] = None,
        time_resolution: TimeResolution = TimeResolution.YEAR,
        date_format: Literal["year", "date", "yearmonth"] = "year",
    ):
        """
        Class function to compile zarr files from each var_key to a pre defined spatial res (set at 0.25) grid with daily interpolated data.

        Args:
            var_key (str, optional): Var key name of compiled data. Defaults to 'h2ds'.
            app_config (AppConfig, optional): Configuration data for var keys. Defaults to AppConfig.
            remote_store_root (Path, optional): Store directory where all environmental data lives (currently D:).
            local_store_root (Path], optional): Local data directory where compiled data lives (currently C:)
            time_resolution: Temporal granularity ('year' or 'month') for file storage. Defaults to 'year'.
            date_format: string date format for output file name.
        """
        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[self.var_key]

        self.local_store_root = (
            local_store_root or get_settings().ZARR_DIR / self.var_config.local_folder
        )
        resolved_remote = remote_store_root or get_settings().STORE_ROOT
        if resolved_remote is None:
            raise ValueError(
                "remote_store_root must be provided or STORE_ROOT must be set in the environment"
            )
        self.remote_store_root: Path = resolved_remote

        if self.var_config.bbox is None:
            raise ValueError(
                f"var_key '{self.var_key}' config must have a bbox for compilation"
            )
        self.bbox = BBox.from_tuple(self.var_config.bbox)

        self.time_resolution = validate_time_resolution(time_resolution)
        self.date_format: Literal["year", "date", "yearmonth"] = date_format

        self.catalog = ZarrCatalog(self.var_key)

    def run(
        self,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        var_keys: Optional[list[str]] = None,
        dx: float = DX,
        dy: float = DY,
        no_zarr_backup: bool = False,
        zarr_backup_dir: Optional[Path] = None,
    ) -> None:
        """
        Main entry point for the h2ds compilation process.

        Without explicit dates, uses incremental mode: reads the h2ds catalog
        end date and compiles ``[h2ds_end + 1 day → today]``.  To backfill a
        historical range (e.g. when adding a new variable), pass explicit
        *start_date* / *end_date*.

        Args:
            start_date: Start of the compilation window. If omitted, inferred
                from the store (see above).
            end_date: End of the compilation window. If omitted, inferred from
                the store (see above).
            var_keys: Variable keys to include. ``None`` compiles all configured
                variables (incremental mode).
            dx: Output grid cell width in degrees. Defaults to 0.25.
            dy: Output grid cell height in degrees. Defaults to 0.25.
            no_zarr_backup: Skip copying written zarr files to the local store. Defaults to False.
            zarr_backup_dir: Override destination for the zarr backup. Defaults to local_store_root.
        """
        logger.info(
            f"Initializing Zarr compilation for variable key: {self.var_key.upper()}"
        )

        start = normalize_date(start_date) if start_date else None
        end = normalize_date(end_date) if end_date else None

        self.var_keys = (
            [var_keys]
            if isinstance(var_keys, str)
            else var_keys or sorted(self.app_config.variables.keys())
        )

        # Source coverage for each variable — computed once
        self._source_coverage = self._compute_source_coverage()

        requested_range = self._resolve_compile_range(start, end)

        self.base_grid = GridBuilder(self.bbox, dx, dy).generate_grid()

        logger.info(
            f"Key variables to compile: {self.var_keys}, "
            f"for period {requested_range.start.date()} -> {requested_range.end.date()}"
        )

        # time chunks
        chunks = split_time_range(requested_range, self.time_resolution)

        logger.debug(
            f"Split into {len(chunks)} chunk(s) ({self.time_resolution} intervals)"
        )

        written_paths: list[Path] = []

        for i, chunk in enumerate(chunks, 1):
            logger.debug(
                f"Chunk {i}/{len(chunks)}: {chunk.start.date()} -> {chunk.end.date()}"
            )

            datasets = []

            for vkey in self.var_keys:
                if vkey == self.var_key:
                    continue

                ds = self._process_variable(vkey, chunk)
                if ds is not None:
                    datasets.append(ds)

            if not datasets:
                logger.warning(
                    f"No datasets collected for chunk {i}: "
                    f"Period: {chunk.start} -> {chunk.end} — Skipping."
                )
                continue

            ds_final = xr.merge(datasets, join="outer")
            assert isinstance(ds_final, xr.Dataset)
            ds_final = chunk_dataset(ds_final)
            ds_final = self._set_attrs(ds_final)

            path = self.catalog.build_file_path(
                ds_final, self.date_format, name_key=self.var_config.dataset_id_rep
            )
            write_append_zarr(self.var_key, ds_final, path)
            written_paths.append(path)

            logger.success(
                f"Finished period {chunk.start.date()} -> {chunk.end.date()}"
            )

        self.catalog.refresh()

        if not no_zarr_backup:
            # Backup all written files to local store in one pass — avoids repeated
            # large directory copies after each individual chunk
            for path in written_paths:
                self.sync_data(path, backup_dir=zarr_backup_dir)

    # =========== DATE RANGE RESOLUTION ===========
    def _compute_source_coverage(self) -> dict[str, DateRange]:
        """Return source catalog coverage for every non-system source variable."""
        result: dict[str, DateRange] = {}
        for vkey in self.var_keys:
            if vkey == self.var_key or vkey in SYSTEM_VAR_KEYS:
                continue
            cov = get_store_coverage(vkey)
            if cov is not None:
                result[vkey] = cov
            else:
                logger.warning(f"No source coverage found for '{vkey}' — skipping.")
        return result

    def _get_h2ds_var_end(self, vkey: str) -> Optional[pd.Timestamp]:
        """
        Query h2ds for the last compiled date of *vkey*.

        Tries three strategies in order:
        1. Configured source variable names (``app_config.variables[vkey].variables``)
           — covers variables processed by ``compile_default``.
        2. ``{vkey}_*`` prefix match — covers depth-sliced variables whose h2ds
           names are ``thetao_100``, ``thetao_200``, etc.
        3. Global h2ds end date — conservative fallback so an unrecognised naming
           convention never causes a 1998 recompile.

        Returns ``None`` when h2ds is empty (fresh install).
        """
        var_names: list[str] = list(
            getattr(self.app_config.variables.get(vkey), "variables", None) or []
        )
        for vn in var_names:
            cov = self.catalog.get_var_time_coverage(vn)
            if cov is not None:
                return cov.end

        # Depth-sliced: try "{vkey}_" prefix
        df = self.catalog.df
        if not df.empty and "variables" in df.columns:
            prefix = f"{vkey}_"
            mask = df["variables"].apply(
                lambda vs: (
                    isinstance(vs, list) and any(v.startswith(prefix) for v in vs)
                )
            )
            sub = df[mask]
            if not sub.empty:
                return pd.Timestamp(sub["end_date"].max())

        # Fallback: global h2ds end (avoids recompiling from source start)
        global_cov = self.catalog.get_time_coverage()
        return global_cov.end if global_cov is not None else None

    def _resolve_compile_range(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> DateRange:
        """
        Resolve the compilation date range.

        **Explicit dates** always win and are passed straight through to
        :func:`resolve_date_range`.

        **Per-variable incremental mode** (no explicit dates): for every source
        variable, the gap between its last compiled end date in h2ds (queried
        directly via :meth:`_get_h2ds_var_end`) and its current source catalog
        end is computed.  The union of all per-variable gaps becomes the
        compilation window, so a lagging variable does not hold back faster
        ones and catches up automatically when new source data arrives.

        Args:
            start: Normalised start timestamp, or ``None`` to infer.
            end: Normalised end timestamp, or ``None`` to infer.

        Returns:
            Resolved :class:`DateRange`.

        Raises:
            ValueError: If no source variable has any data to compile.
        """
        if start is not None or end is not None:
            result = resolve_date_range(self.var_key, start, end)
            if result is None:
                raise ValueError(f"Invalid date range: start ({start}) > end ({end})")
            return result

        ranges: list[DateRange] = []
        for vkey, src_cov in self._source_coverage.items():
            h2ds_var_end = self._get_h2ds_var_end(vkey)
            var_start = (
                h2ds_var_end + pd.Timedelta(days=1)
                if h2ds_var_end is not None
                else src_cov.start
            )
            if var_start <= src_cov.end:
                ranges.append(DateRange(start=var_start, end=src_cov.end))
                logger.debug(
                    f"{vkey}: compiling {var_start.date()} → {src_cov.end.date()}"
                )
            else:
                logger.debug(f"{vkey}: up to date ({src_cov.end.date()}), skipping.")

        if not ranges:
            raise ValueError(
                "All configured variables are up to date — nothing to compile. "
                "Pass explicit --start-date / --end-date to force a range."
            )

        inferred = DateRange(
            start=min(r.start for r in ranges),
            end=max(r.end for r in ranges),
        )
        logger.info(
            f"Per-variable incremental range: "
            f"{inferred.start.date()} → {inferred.end.date()}"
        )
        return inferred

    # =========== DATASET ATTRIBUTES ===========
    def _set_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Set global and variables attributes from yaml file.
        Args:
            ds: Dataset for atts assignment
        """
        ds.attrs = get_settings().global_attrs
        for var in ds.data_vars:
            var_info = get_settings().get_var_info(str(var))
            ds[var].attrs.update({key: val for key, val in var_info.items()})
        return ds

    #  ============ PROCESSING ==================
    def _process_variable(
        self,
        var_key: str,
        date_range: DateRange,
    ) -> Optional[xr.Dataset]:
        """Dispatch var_key to its registered compile processor (or the default)."""
        # Lazy import breaks the compiler.py ↔ compiler_registry.py cycle.
        from h2mare.processing.compiler_registry import (
            COMPILE_PROCESSORS,
            compile_default,
        )

        is_system = var_key in SYSTEM_VAR_KEYS
        catalog = None if is_system else ZarrCatalog(var_key)

        if not is_system and not self._has_overlap(var_key, date_range, catalog):
            return None

        processor = COMPILE_PROCESSORS.get(var_key, compile_default)
        return processor(self, catalog, date_range)

    # ============== UTILITIES ===================
    def _has_overlap(
        self, var_key: str, date_range: DateRange, catalog: ZarrCatalog
    ) -> bool:
        """Check for temporal overlap between requested date_range and catalog date_range"""
        env_daterange = catalog.get_time_coverage()

        if env_daterange:
            if date_range.overlaps(env_daterange):
                return True
            else:
                logger.warning(f"Skipping {var_key}: dates out of range.")
                return False
        return False

    def sync_data(self, remote_path: Path, backup_dir: Optional[Path] = None) -> None:
        """
        Copy a compiled zarr file to the local backup store.

        Args:
            remote_path: path built by the caller via ``ZarrCatalog.build_file_path()``
            backup_dir: destination directory; defaults to ``local_store_root``.
        """
        local_path = (backup_dir or self.local_store_root) / remote_path.name

        logger.info(f"Copying {remote_path} to {local_path}")

        try:
            shutil.copytree(remote_path, local_path, dirs_exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.exception(f"Failed to copy {remote_path} to {local_path}: {e}")

        logger.success("File copied!")


if __name__ == "__main__":
    log_path = get_settings().LOGS_DIR / f"{Path(__file__).stem}.log"
    logger.add(log_path, level="INFO")

    Compiler().run(start_date="2025-01-01", end_date="2025-01-31")
