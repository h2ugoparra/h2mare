"""
Create h2ds zarr files
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Literal, Optional

import ephem
import numpy as np
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
from h2mare.utils.spatial import GridBuilder, clip_land_data
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

        if self.var_config.bbox is not None:
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

        Date range inference (when start_date / end_date are not provided):

        - **Incremental mode** — triggered when *var_keys* is ``None`` (all
          variables).  The h2ds catalog end date is read and the range is set to
          ``[h2ds_end + 1 day → today]``.  This is the normal pipeline-update
          path: extend h2ds with the latest available data.

        - **Backfill mode** — triggered when specific *var_keys* are supplied
          (e.g. ``["thetao"]``).  The union of those source variables' catalog
          ranges is used instead of the h2ds gap.  This handles adding a new
          variable to an already-compiled historical h2ds dataset without
          requiring explicit dates.

        Explicit *start_date* / *end_date* always take priority and bypass both
        inference modes.

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

        requested_range = self._resolve_compile_range(start, end, var_keys)

        self.base_grid = GridBuilder(self.bbox, dx, dy).generate_grid()

        logger.info(
            f"Key variables to compile: {self.var_keys}, "
            f"for period {requested_range.start.date()} -> {requested_range.end.date()}"
        )

        # time chunks
        chunks = split_time_range(requested_range, self.time_resolution)

        logger.info(
            f"Split into {len(chunks)} chunk(s) ({self.time_resolution} intervals)"
        )

        written_paths: list[Path] = []

        for i, chunk in enumerate(chunks, 1):
            logger.debug(
                f"Chunk {i}/{len(chunks)}: {chunk.start.date()} -> {chunk.end.date()}"
            )

            datasets = []

            for vkey in self.var_keys:
                if vkey == "h2ds":
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
    def _resolve_compile_range(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        var_keys: Optional[list[str]],
    ) -> DateRange:
        """
        Resolve the compilation date range using one of two modes.

        **Incremental mode** (``var_keys`` is ``None``):
            Delegates to :func:`resolve_date_range` with ``var_key="h2ds"``, which
            reads the h2ds catalog and returns ``[h2ds_end + 1 day → today]``.
            Used for routine pipeline updates where all variables are compiled
            together and only the latest missing period needs to be filled.

        **Backfill mode** (specific ``var_keys`` supplied):
            Collects the time coverage of each source variable's catalog and
            returns ``[min(starts) → max(ends)]`` across all of them.  Used when
            adding a new variable to an already-compiled historical h2ds dataset;
            the inferred range covers exactly what those source variables have
            available regardless of what h2ds already contains.

        Explicit ``start`` / ``end`` timestamps short-circuit both modes and are
        passed directly to :func:`resolve_date_range`.

        Args:
            start: Normalised start timestamp, or ``None`` to infer.
            end: Normalised end timestamp, or ``None`` to infer.
            var_keys: The raw ``var_keys`` argument received by :meth:`run`
                before expansion.  ``None`` selects incremental mode; a list
                (even a single-element one) selects backfill mode.

        Returns:
            Resolved :class:`DateRange`.

        Raises:
            ValueError: If backfill mode is selected but none of the requested
                variables have data in the store.
        """
        # Explicit dates always win — standard resolver fills in missing halves
        if start is not None or end is not None:
            return resolve_date_range(self.var_key, start, end)

        # Variables that have no independent zarr store and cannot contribute
        # a date range (handled specially elsewhere in the compiler)
        _no_store = SYSTEM_VAR_KEYS

        # Normalise var_keys to a flat list, mirroring run()'s own normalisation
        if var_keys is None:
            explicit_keys: list[str] = []
        elif isinstance(var_keys, str):
            explicit_keys = [var_keys]
        else:
            explicit_keys = list(var_keys)

        source_keys = [v for v in explicit_keys if v not in _no_store]

        if not source_keys:
            # No specific source variables requested → incremental mode
            logger.debug(
                "No specific var_keys supplied: using incremental h2ds date range "
                "(h2ds end + 1 day → today)."
            )
            return resolve_date_range(self.var_key, start, end)

        # Backfill mode: derive range from the source variables' own catalogs
        ranges: list[DateRange] = []
        missing: list[str] = []
        for vkey in source_keys:
            coverage = get_store_coverage(vkey)
            if coverage is not None:
                ranges.append(coverage)
            else:
                missing.append(vkey)

        if missing:
            logger.warning(
                f"No catalog data found for {missing} — "
                "excluded from date range inference."
            )

        if not ranges:
            raise ValueError(
                f"None of the requested variables {source_keys} have data in the "
                "store. Provide explicit --start-date and --end-date."
            )

        inferred_start = min(r.start for r in ranges)
        inferred_end = max(r.end for r in ranges)
        inferred_range = DateRange(start=inferred_start, end=inferred_end)

        logger.info(
            f"Backfill mode: inferred date range from source variables {source_keys}: "
            f"{inferred_start.date()} → {inferred_end.date()}"
        )
        return inferred_range

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
        depth_range: list[float] = [0, 100, 500, 1000],
    ) -> Optional[xr.Dataset]:
        """
        Process each var_key.

        Args:
            var_key: Variable key from app_config.variables
            start_date: Start date to process
            end_date: End date to proces
            depth_range: For 'o2' var_key with depth dim
        """

        if var_key not in SYSTEM_VAR_KEYS:
            vk_catalog = ZarrCatalog(var_key)
            if not self._has_overlap(var_key, date_range, vk_catalog):
                return None

        if var_key == "bathy":
            return self._process_bathy()

        if var_key == "moon":
            return self._process_moon(date_range)

        if var_key == "o2":
            return self._process_o2(vk_catalog, date_range, depth_range)

        if var_key == "thetao":
            return self._process_thetao(vk_catalog, date_range)

        # General case
        try:
            ds = vk_catalog.open_dataset(
                start_date=date_range.start,
                end_date=date_range.end,
                bbox=self.var_config.bbox,
            )
        except FileNotFoundError:
            logger.warning(
                f"No data for {var_key} during {date_range.start.date()}–{date_range.end.date()} — skipping."
            )
            return None

        if var_key == "atm-accum-avg":
            ds = ds.drop_vars(["dayofyear", "month", "quantile"])

        if var_key == "sst":
            ds = postprocess_sst_fdist(ds)

        return ds.interp_like(self.base_grid, method="linear", assume_sorted=True)

    def _process_bathy(self):
        var_cfg = self.app_config.variables["bathy"]
        if var_cfg.data_file is None:
            raise ValueError("bathy config entry is missing required 'data_file' field")
        data_path = self.remote_store_root / var_cfg.local_folder / var_cfg.data_file
        ds = xr.open_dataset(data_path).sel(
            lon=slice(self.bbox.xmin, self.bbox.xmax),
            lat=slice(self.bbox.ymin, self.bbox.ymax),
        )
        return ds.interp_like(self.base_grid, method="linear", assume_sorted=True)

    def _process_moon(self, date_range: DateRange) -> xr.Dataset:
        """
        Moon ilumination data process for each day of a given daterange.
        Lat/lon points are the mean values of base_grid and the same for all days.
        """
        dates = pd.date_range(date_range.start, date_range.end, freq="D")
        lat = self.base_grid.lat.mean().values
        lon = self.base_grid.lon.mean().values
        moon_phase = calculate_moon_phase(lat, lon, dates)
        da = xr.DataArray(
            np.broadcast_to(
                np.array(moon_phase)[:, None, None],
                (len(dates), len(self.base_grid.lat), len(self.base_grid.lon)),
            ),
            name="moon_phase",
            dims=["time", "lat", "lon"],
            coords={
                "time": dates,
                "lat": self.base_grid.lat,
                "lon": self.base_grid.lon,
            },
        )
        return clip_land_data(da.to_dataset())

    def _process_o2(
        self, catalog: ZarrCatalog, date_range: DateRange, depths: list[float]
    ) -> xr.Dataset | None:
        """
        Create Dissolved oxygen variables for specified depth intervals
        """
        try:
            ds = catalog.open_dataset(
                start_date=date_range.start,
                end_date=date_range.end,
                bbox=self.bbox,
            )
        except FileNotFoundError:
            logger.warning(
                f"No data for o2 during {date_range.start.date()}–{date_range.end.date()} — skipping."
            )
            return None
        # Select all target depths at once and interpolate the full (time, lat, lon, depth)
        # array in one pass — avoids calling interp_like once per depth level
        ds_depths = ds.sel(depth=depths, method="nearest")
        ds_interp = ds_depths.interp_like(
            self.base_grid, method="linear", assume_sorted=True
        )
        return xr.Dataset(
            {
                f"o2_{target}": ds_interp.o2.isel(depth=i).drop_vars("depth")
                for i, target in enumerate(depths)
            }
        )

    def _process_thetao(
        self,
        catalog: ZarrCatalog,
        date_range: DateRange,
        depths: list[float] = [100, 200, 500, 1000],
    ) -> xr.Dataset | None:
        """Create potential temperature variables for specified depth intervals"""
        try:
            ds = catalog.open_dataset(
                start_date=date_range.start,
                end_date=date_range.end,
                bbox=self.bbox,
                chunks={"depth": 1},
            )
        except FileNotFoundError:
            logger.warning(
                f"No data for thetao during {date_range.start.date()}–{date_range.end.date()} — skipping."
            )
            return None
        ds_depths = ds.sel(depth=depths, method="nearest")
        ds_interp = ds_depths.interp_like(
            self.base_grid, method="linear", assume_sorted=True
        )
        return xr.Dataset(
            {
                f"thetao_{target}": ds_interp.thetao.isel(depth=i).drop_vars("depth")
                for i, target in enumerate(depths)
            }
        )

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
