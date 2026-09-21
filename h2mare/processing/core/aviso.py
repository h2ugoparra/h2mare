"""
Processes AVISO data, namely FSLE and EDDY TRAJECTORY ATLAS
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr
from global_land_mask import globe
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore

from h2mare.config import AppConfig, get_settings
from h2mare.models import KeyVarConfigEntry
from h2mare.storage.coverage import resolve_date_range
from h2mare.storage.provenance import write_provenance_for_window
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import (
    apply_cf_attrs,
    check_grid_compatible,
    chunk_dataset,
    convert360_180,
    ds_float64_to_float32,
)
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox, DateLike, DateRange, FilePeriod
from h2mare.utils.datetime_utils import normalize_date
from h2mare.utils.files_io import filter_raw_files
from h2mare.utils.paths import resolve_download_path, resolve_store_path
from h2mare.utils.spatial import (
    GridBuilder,
    haversine_min_distance_kdtree,
    to_unit_sphere,
)
from h2mare.validators import validate_file_period, validate_var_key

# ====================================================
# ================= EDDIES PROCESSOR =================
# ====================================================
#: Grid the eddy rasterisation uses when config declares none: 10 cells per
#: degree (0.1°), which is what the existing store holds. The atlas gives eddy
#: centres as continuous positions, so this is a sampling choice rather than a
#: source resolution — declare `cells_per_degree` on the eddies entry to change
#: it, and regenerate the store.
DEFAULT_CELLS_PER_DEGREE = 10

# Raw var names and respective map for processed vars
EDDY_VAR_MAP: dict[str, str] = {
    "track": "track",
    "effective_radius": "effective_radius",  # used internally, not in output
    "speed_radius": "speedrad_km",
    "amplitude": "amp",
    "speed_average": "speed",
    "observation_number": "ndays",
}

OUTPUT_VAR_SCALINGS: dict[str, float] = {
    "speedrad_km": 0.001,
}

EDDY_TYPE_MAP: dict[str, str] = {"anticyclonic": "ac", "cyclonic": "c"}


@dataclass(frozen=True)
class GridData:
    lat: NDArray
    lon: NDArray
    latlon_arr: NDArray
    sea_mask: NDArray


def find_nearest_vectorized(
    query_lats: NDArray,
    query_lons: NDArray,
    target_lats: NDArray,
    target_lons: NDArray,
) -> NDArray[np.intp]:
    """
    For each (lat, lon) query point, find the index of the nearest point
    in the target set using a KD-tree on Cartesian coordinates.

    Converts lat/lon to 3D unit-sphere Cartesian coordinates before indexing,
    which avoids distortions near the poles and across the antimeridian that
    would arise from querying on raw degree values.

    Args:
        query_lats: Latitudes of grid points to query. Shape: (N,)
        query_lons: Longitudes of grid points to query. Shape: (N,)
        target_lats: Latitudes of eddy centers. Shape: (M,)
        target_lons: Longitudes of eddy centers. Shape: (M,)

    Returns:
        NDArray[np.intp]: Indices into target arrays of the nearest point
            for each query point. Shape: (N,)
    """

    target_cartesian = to_unit_sphere(target_lats, target_lons)
    query_cartesian = to_unit_sphere(query_lats, query_lons)

    tree = cKDTree(target_cartesian)
    _, nearest_indices = tree.query(query_cartesian, workers=-1)

    return nearest_indices


def _group_dates(
    dates: pd.DatetimeIndex,
    groupby: FilePeriod,
) -> Iterator[tuple[int | tuple[int, int], pd.DatetimeIndex]]:
    """
    Yield (period_key, dates_in_period) pairs from a DatetimeIndex.

    For 'year':  period_key is the year (e.g. 2020)
    For 'month': period_key is a (year, month) tuple (e.g. (2020, 1))
    """
    if groupby == FilePeriod.YEAR:
        for year in dates.year.unique():
            yield int(year), dates[dates.year == year]
    elif groupby == FilePeriod.MONTH:
        for year in dates.year.unique():
            for month in dates[dates.year == year].month.unique():
                mask = (dates.year == year) & (dates.month == month)
                yield (int(year), int(month)), dates[mask]


def _raw_source(path: Path) -> Optional[str]:
    """
    Whether a raw file is reprocessed or near-real-time, from its location.

    The downloader stages each stream into a ``rep``/``nrt`` subfolder and the
    store keeps that layout, so the directory is the marker. Returns ``None``
    for a file sitting outside either — a legacy flat layout, or a product
    version dropped in by hand.
    """
    parts = {part.lower() for part in path.parts}
    if "nrt" in parts:
        return "nrt"
    if "rep" in parts:
        return "rep"
    return None


def _is_degenerate_axis(values: NDArray[np.float64], tol: float = 1e-6) -> bool:
    """
    True when a coordinate axis carries near-duplicate points.

    That is the signature of an axis produced by unioning two grids that differ
    only in floating-point representation: the smallest gap collapses to ~1e-16
    of the nominal spacing. Native product grids are not perfectly uniform
    either (1/12°, 1/24°, 15″ are not exactly representable), so the test is a
    ratio against the median step rather than strict uniformity.
    """
    if values.size < 3:
        return False
    steps = np.abs(np.diff(values))
    if (steps <= 0).any():
        return True
    return bool(steps.min() / np.median(steps) < tol)


class EDDIESProcessor:
    def __init__(
        self,
        *,
        var_key: str = "eddies",
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        download_root: Optional[Path] = None,
        file_period: FilePeriod = FilePeriod.YEAR,
        date_format: Literal["year", "date", "yearmonth"] = "year",
    ) -> None:
        """
        A class representing the setup to process AVISO Eddies Trajectory data.
        Convert observations from raw downloaded data to gridded data.
        Distances to eddies center (in km and normalized by radius) are computed

        Args:
            var_key: Variable key that must exist in app_config.variables. Defaults to 'eddies'.
            app_config (Optional[AppConfig], optional): Application configuration. If None, loads from get_settings().
            store_root (Optional[Path]): Root directory for zarr files. If None, uses get_settings().STORE_ROOT or get_settings().ZARR_DIR.
            download_root (Optional[Path]): Root directory with downloaded data. If None, uses get_settings().DOWNLOADS_DIR.
            grid cell size for lon and lat, (dx and dy respectively).
        """
        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[self.var_key]

        self.store_root = resolve_store_path(self.var_config, store_root)
        self.download_root = resolve_download_path(self.var_config, download_root)

        if self.var_config.pattern is None:
            raise ValueError(
                f"Variable '{self.var_key}' has no `pattern`; cannot extract dates "
                "from eddies filenames"
            )
        self.date_pattern = re.compile(self.var_config.pattern)
        self.type_pattern = re.compile(r"(anticyclonic|cyclonic)", re.IGNORECASE)

        if self.var_config.bbox is not None:
            self.bbox = BBox.from_tuple(self.var_config.bbox)

        self.file_period = validate_file_period(file_period)
        self.date_format: Literal["year", "date", "yearmonth"] = date_format

        # Initialise parent class with the Repository index class
        self.catalog = ZarrCatalog(self.var_key)

    def run(
        self,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        n_workers: int = 4,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
    ) -> None:
        """
        Process downloaded files and writes.

        Args:
            start_date (Optional[DateLike], optional): Start date to process. Defaults to None and get's date from ZarrCatalog.
            end_date (Optional[DateLike], optional): End date to process. Defaults to Noneand get's date from ZarrCatalog.
            n_workers (int, optional): Number of workers for multiprocessing daily files. Defaults to 4.
            dx, dy: x,y cell size resp., in degrees. Default to the grid the
                var_key's config declares (`cells_per_degree`).
        """

        logger.info("Starting eddies processing")

        step = 1 / (self.var_config.cells_per_degree or DEFAULT_CELLS_PER_DEGREE)
        grid = self._get_gridded_data(
            dx if dx is not None else step, dy if dy is not None else step
        )
        records = self._get_downloaded_metadata()
        requested_ranges = self._resolve_all_ranges(records, start_date, end_date)

        if not requested_ranges:
            logger.info(
                f"[{self.var_key}] No downloaded file overlaps the requested range "
                "— nothing to convert"
            )
            return

        # Determine the full set of years to process across the relevant records
        all_dates = pd.date_range(
            min(r.start for r in requested_ranges.values()),
            max(r.end for r in requested_ranges.values()),
            freq="D",
        )

        for _, period_dates in _group_dates(all_dates, self.file_period):
            ds_list = []

            for record in records:
                eddy_type, _, path = record
                eddy_type_str = EDDY_TYPE_MAP[eddy_type]
                year_range = requested_ranges.get(path)

                # Files outside the requested range carry no window at all
                if year_range is None:
                    continue

                # Skip if this record doesn't cover this year
                sel_dates = period_dates[
                    (period_dates >= pd.to_datetime(year_range.start))
                    & (period_dates <= pd.to_datetime(year_range.end))
                ]
                if len(sel_dates) == 0:
                    continue

                logger.info(
                    f"Processing {eddy_type.upper()} | {len(sel_dates)} days | {n_workers} workers"
                )

                ds_raw = self._prepare_raw_dataset(
                    path, DateRange(sel_dates[0], sel_dates[-1])
                )
                ds_year = self._process_period(
                    ds_raw, eddy_type_str, grid, sel_dates, n_workers
                )

                if ds_year is not None:
                    ds_list.append(ds_year)

            if ds_list:
                merged = xr.merge(ds_list, join="outer")
                assert isinstance(merged, xr.Dataset)
                ds_merged = chunk_dataset(merged)
                path = ZarrCatalog(self.var_key).build_file_path(
                    ds_merged, self.date_format
                )
                write_append_zarr(self.var_key, ds_merged, path)
                written = DateRange(
                    pd.to_datetime(ds_merged.time.min().values),
                    pd.to_datetime(ds_merged.time.max().values),
                )
                self._write_provenance(path, written)
                del ds_list, ds_merged

        # logger.success("Completed!")

    # ============== PREPARE DATA ===============
    def _get_gridded_data(self, dx: float, dy: float) -> GridData:
        """
        Create a base grid with land mask on the configured grid (``GridBuilder``),
        taking the store's own axes when it already holds that grid.

        Raises:
            ValueError: if the store holds a different grid than config declares.
        """

        def create_base_grid(
            lat: NDArray[np.float64], lon: NDArray[np.float64]
        ) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
            """

            Args:
                lat, lon (NDArray): arrays with N and M lengths

            Returns:
                NDArray: shape (N*M, 2)
                NDArray: shape (N, M)
            """
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            sea_mask = ~globe.is_land(lat_grid, lon_grid)
            mask_flat = sea_mask.flatten()
            return (
                np.column_stack(
                    (lat_grid.flatten()[mask_flat], lon_grid.flatten()[mask_flat])
                ),
                sea_mask,
            )

        base_grid = GridBuilder(
            self.bbox,
            dx=dx,
            dy=dy,
            values_at=self.var_config.values_at,
        ).generate_grid()
        grid = self._grid_from_store() if self.catalog.exists() else None
        if grid is None:
            lat = base_grid.coords["lat"].values
            lon = base_grid.coords["lon"].values
        else:
            lat, lon = grid
            # The store's axes are reused only when they are the configured
            # grid — they are what keeps float drift out of the store. On a
            # changed cells_per_degree / values_at they are not, and reusing
            # them silently rebuilt the store on its old grid: a full-period
            # rewrite skips the write path's own grid check.
            try:
                check_grid_compatible(
                    xr.Dataset(coords={"lat": lat, "lon": lon}), base_grid
                )
            except ValueError as e:
                raise ValueError(
                    f"[{self.var_key}] The store holds a different grid than "
                    f"config declares: {e} To regenerate it on the configured "
                    "grid, move its existing *.zarr files out of "
                    f"{self.catalog.store_root} and convert every period."
                ) from None

        latlon_arr, sea_mask = create_base_grid(lat, lon)
        return GridData(lat, lon, latlon_arr, sea_mask)

    def _read_manifest(self) -> list[dict]:
        """Read the download manifest written by AVISODownloader, or [] if absent."""
        manifest_path = self.download_root / "h2mare_manifest.json"
        if not manifest_path.exists():
            return []
        try:
            return json.loads(manifest_path.read_text())
        except Exception as e:
            logger.warning(f"Could not read download manifest: {e}")
            return []

    def _write_provenance(self, zarr_path: Path, written: DateRange) -> None:
        """
        Record which dataset covered *written* on the Zarr just produced.

        The generic converter path in Netcdf2Zarr never runs for eddies —
        ``_process_eddies`` delegates here and this class writes its own Zarr —
        so without this every eddies file lacks ``source_datasets`` and the
        catalog scanner falls back to ``dataset_id_rep``, labelling
        near-real-time data as delayed-time.
        """
        manifest = self._read_manifest()
        if not manifest:
            logger.debug(
                f"[{self.var_key}] No download manifest in {self.download_root}; "
                "skipping provenance for this period"
            )
            return

        try:
            records = write_provenance_for_window(zarr_path, manifest, written)
        except Exception as e:
            logger.warning(f"Could not write provenance for {zarr_path.name}: {e}")
            return

        if records:
            logger.debug(
                f"[{self.var_key}] Wrote provenance to {zarr_path.name}: "
                + ", ".join(
                    f"{r['dataset_type']} {r['start_date']}..{r['end_date']}"
                    for r in records
                )
            )

    def _grid_from_store(
        self,
    ) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Read the store's established grid from a single canonical Zarr file.

        Deliberately *not* ``catalog.open_dataset()``: with no arguments that
        opens every file at once, and ``combine="by_coords"`` then takes the
        union of their coordinate axes. Axes that differ only in the last
        floating-point bits therefore merge into a doubled axis full of
        near-duplicate points — and because the result is written back to the
        store, the next run reads an even worse grid. Reading one file keeps the
        grid a property of the store rather than of how many files it holds.

        Returns ``None`` when no usable grid is available, so the caller falls
        back to building one from the configured bbox.
        """
        try:
            df = self.catalog.df
            if df.empty:
                return None
            path = df.sort_values("start_date").iloc[0]["path"]
            with xr.open_zarr(path, consolidated=False) as ds:
                lat = ds.coords["lat"].values
                lon = ds.coords["lon"].values
        except Exception as e:
            logger.warning(
                f"[{self.var_key}] Could not read the store grid ({e}); "
                "falling back to the configured bbox"
            )
            return None

        for name, values in (("lat", lat), ("lon", lon)):
            if _is_degenerate_axis(values):
                logger.error(
                    f"[{self.var_key}] Store grid has a degenerate {name} axis "
                    f"({values.size} points with near-duplicate values) in "
                    f"{Path(path).name}. Falling back to the configured bbox; "
                    "that file was written on a unioned grid and needs "
                    "re-converting from raw."
                )
                return None

        return lat, lon

    def _get_downloaded_metadata(
        self, root_dir: Optional[Path] = None
    ) -> list[tuple[str, DateRange, Path]]:
        """
        Retrieve list of tuples with:
          - eddies type (anticycloninc/cyclonic)
          - start date
          - end date
          - file path

        Args:
            root_dir: Directory with downloaded files. Defaults to None, setting download_dir.

        Raises:
            FileNotFoundError: If no files are found in root_dir

        Returns:
            list: tuple (eddy type, Daterange, file path)
        """
        root_dir = root_dir or self.download_root
        files = filter_raw_files(list(root_dir.rglob("*.nc")), self.var_config)

        records = []
        for f in files:
            if date_match := self.date_pattern.search(str(f)):
                dt_ini = pd.to_datetime(date_match.group(1), format="%Y%m%d")
                dt_fin = pd.to_datetime(date_match.group(2), format="%Y%m%d")

                eddy_type = None
                if type_match := self.type_pattern.search(str(f)):
                    eddy_type = type_match.group(1).lower()
                if not type_match:
                    raise ValueError(f"Cannot infer eddy type from filename: {f}")

                records.append((eddy_type, DateRange(dt_ini, dt_fin), f))
        if not records:
            raise FileNotFoundError(f"No files to process in {root_dir}")
        return records

    def _resolve_all_ranges(
        self,
        records: list[tuple[str, DateRange, Path]],
        start_date: Optional[DateLike],
        end_date: Optional[DateLike],
    ) -> dict[Path, DateRange]:
        """
        Per-file conversion window, dropping files the request does not touch.

        Keyed by path rather than eddy type. A store holds several files per
        type — rep long/short/untracked, nrt, and any other product version
        sitting in the same tree — so keying by type collapses them and every
        record ends up processed against whichever file resolved last.

        Files whose own span does not overlap the request are omitted rather
        than aborting the run: a directory legitimately holds files outside the
        requested window, and the caller asked for a window, not for every file
        to be relevant.
        """
        resolved: dict[Path, DateRange] = {}
        eddy_type_of: dict[Path, str] = {}
        for eddy_type, download_range, path in records:
            window = self._resolve_date_range(download_range, start_date, end_date)
            if window is None:
                logger.debug(
                    f"[{self.var_key}] {path.name} ({download_range}) does not "
                    "overlap the requested range — skipping"
                )
                continue
            resolved[path] = window
            eddy_type_of[path] = eddy_type

        return self._prefer_rep(resolved, eddy_type_of)

    def _prefer_rep(
        self, resolved: dict[Path, DateRange], eddy_type_of: dict[Path, str]
    ) -> dict[Path, DateRange]:
        """
        Clip near-real-time windows to start after the reprocessed data ends.

        rep and nrt overlap on the server — AVISO's nrt trajectory file spans
        2018 to today while META3.2 delayed-time runs to 2022 — and the
        reprocessed product is the better one where both exist. The downloader
        already applies this (`nrt_start = rep_avail.end + 1 day`), but a
        conversion reads whatever files are on disk, so without the same rule
        both sources contribute to the overlap and ``xr.merge`` either combines
        two versions of the same period or raises on the cells where they
        disagree.

        Applied per eddy type, since anticyclonic and cyclonic are independent
        streams. Files with no rep/nrt directory in their path are left alone,
        which keeps legacy flat layouts working.
        """
        rep_end: dict[str, pd.Timestamp] = {}
        for path, window in resolved.items():
            if _raw_source(path) == "rep":
                key = eddy_type_of[path]
                rep_end[key] = max(rep_end.get(key, window.end), window.end)

        if not rep_end:
            return resolved

        out: dict[Path, DateRange] = {}
        for path, window in resolved.items():
            end = rep_end.get(eddy_type_of[path])
            if _raw_source(path) != "nrt" or end is None:
                out[path] = window
                continue

            start = max(window.start, pd.Timestamp(end) + pd.Timedelta(days=1))
            if start > window.end:
                logger.debug(
                    f"[{self.var_key}] {path.name} fully covered by reprocessed "
                    f"data (rep ends {pd.Timestamp(end).date()}) — skipping"
                )
                continue
            if start != window.start:
                logger.info(
                    f"[{self.var_key}] {path.name}: deferring to reprocessed data, "
                    f"converting from {start.date()} instead of {window.start.date()}"
                )
            out[path] = DateRange(start, window.end)
        return out

    def _resolve_date_range(
        self,
        download_range: DateRange,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
    ) -> Optional[DateRange]:
        """
        Overlap between the requested range and one downloaded file's own range.

        Returns ``None`` when they do not overlap, or when the store is already
        up to date.

        Args:
            path: path for file to check dates in file name
            start_date: Start date to process (None = use default or infer)
            end_date: End date to process (None = use default or infer)
        """

        start = normalize_date(start_date) if start_date else None
        end = normalize_date(end_date) if end_date else None

        requested_range = resolve_date_range(self.var_key, start, end)
        if requested_range is None:
            return None
        # None rather than raising: a raw directory may legitimately hold files
        # outside the requested window, and one of them must not abort the run.
        return requested_range.intersection(download_range)

    def _prepare_raw_dataset(self, path: Path, dates: DateRange) -> xr.Dataset:
        """Preprocess original dataframe to subset by geo_extent with +10deg for distance calculations and by time range"""
        with xr.open_dataset(path) as ds:
            missing = [v for v in self.var_config.source_vars if v not in ds.variables]
            if missing:
                # e.g. AVISO's "untracked" eddy files carry no `track` variable.
                # Without this the failure is a bare KeyError several frames down.
                raise ValueError(
                    f"{path.name} is missing source_vars {missing} — it is not a "
                    f"'{self.var_key}' file the pipeline can read. Narrow the "
                    "selection with the variable's `raw_include` in config.yaml."
                )
            ds = ds[self.var_config.source_vars]
            ds["longitude"] = ds["longitude"] - 360
            ds["time"] = ds["time"].dt.floor("D")
            ds = ds.sel(
                obs=(
                    (ds.longitude >= self.bbox.xmin - 10)
                    & (ds.longitude <= self.bbox.xmax + 10)
                    & (ds.latitude >= self.bbox.ymin - 10)
                    & (ds.latitude <= self.bbox.ymax + 10)
                    & (ds.time >= dates.start)
                    & (ds.time <= dates.end)
                )
            )

        return ds.persist()

    # ================== PROCESS DATA ============
    def _process_period(
        self,
        ds_raw: xr.Dataset,
        eddy_type_str: str,
        grid: GridData,
        dates: pd.DatetimeIndex,
        n_workers: int,
    ) -> xr.Dataset | None:
        """Process all days in a period and return a concatenated Dataset."""

        # Flattened query-point coordinates depend only on the fixed grid, so
        # build them once here instead of per day inside each worker.
        all_lats = np.repeat(grid.lat, len(grid.lon))
        all_lons = np.tile(grid.lon, len(grid.lat))

        worker = partial(
            _process_daily_static,
            ds=ds_raw,
            eddy_type_str=eddy_type_str,
            latlon1_arr=grid.latlon_arr,
            lat1=grid.lat,
            lon1=grid.lon,
            sea_mask=grid.sea_mask,
            all_lats=all_lats,
            all_lons=all_lons,
        )

        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(worker, dates)

        daily = [r for r in results if r is not None]
        if not daily:
            logger.warning(
                f"No valid results for {eddy_type_str} in year {dates[0].year}"
            )
            return None

        # Explicit join: xarray's concat default changes from "outer" to "exact".
        ds_year = xr.concat(daily, dim="time", join="outer")
        # The trajectory path bypasses NetcdfToZarr.process_dataset, so this is
        # where the eddies store gets its metadata. Same helper, so it also
        # picks up the coordinate attributes the old local _set_attrs never set.
        return apply_cf_attrs(ds_year, native_var_key="eddies")


def _process_daily_static(
    date: pd.Timestamp,
    *,
    ds: xr.Dataset,
    eddy_type_str: str,
    latlon1_arr: NDArray,
    lat1: NDArray,
    lon1: NDArray,
    sea_mask: NDArray,
    all_lats: NDArray,
    all_lons: NDArray,
) -> xr.Dataset | None:
    """Process daily files statically to avoid pickel class function."""
    try:
        ds_day = ds.sel(obs=(ds.time == date))
        lat2 = ds_day["latitude"].values
        lon2 = ds_day["longitude"].values
        latlon2_arr = np.column_stack((lat2, lon2))

        # --- Distance to nearest eddy center ---
        min_dist = haversine_min_distance_kdtree(latlon1_arr, latlon2_arr)
        dist_grid = np.full((len(lat1), len(lon1)), np.nan)
        dist_grid[sea_mask] = min_dist

        # --- Vectorised nearest-neighbour lookup ---
        nearest_indices = find_nearest_vectorized(all_lats, all_lons, lat2, lon2)
        nearest_data = ds_day.isel(obs=nearest_indices)

        # --- Build output variables from map ---
        coords = {"time": [date], "lat": lat1, "lon": lon1}
        data_vars: dict[str, tuple] = {}

        effective_radius_grid = nearest_data["effective_radius"].values.reshape(
            len(lat1), len(lon1)
        )

        for src_var, out_suffix in EDDY_VAR_MAP.items():
            if src_var == "effective_radius":
                continue  # used below, not written directly to output

            grid = (
                nearest_data[src_var].values.reshape(len(lat1), len(lon1)).astype(float)
            )
            grid = np.where(sea_mask, grid, np.nan)

            scaling = OUTPUT_VAR_SCALINGS.get(out_suffix, 1.0)
            out_name = f"{eddy_type_str}_{out_suffix}"
            data_vars[out_name] = (["time", "lat", "lon"], grid[np.newaxis] * scaling)

        # --- Derived variables ---
        dist_km_name = f"{eddy_type_str}_dist_km"
        data_vars[dist_km_name] = (["time", "lat", "lon"], dist_grid[np.newaxis])
        data_vars[f"{eddy_type_str}_normdist"] = (
            ["time", "lat", "lon"],
            dist_grid[np.newaxis] / (effective_radius_grid[np.newaxis] * 0.001),
        )

        return ds_float64_to_float32(xr.Dataset(data_vars, coords=coords))

    except Exception as e:
        logger.exception(f"Failed to process {date}: {e}")
        return None


# ================================================
# ================ FSLE PROCESSOR ================
# ================================================
def process_fsle(
    ds: xr.Dataset, var_config: KeyVarConfigEntry, var_key: str | None = None
) -> xr.Dataset:
    """
    Process AVISO FSLE raw data, which is currently downloaded globally.

    Args:
        ds (xr.Dataset): dataset for processing
        var_config (KeyVarConfigEntry): Variable configuration entry containing variable names and bounding box for subsetting
    """
    ds = ds[var_config.source_vars]
    ds = convert360_180(ds)
    if var_config.bbox is not None:
        xmin, ymin, xmax, ymax = var_config.bbox
        return ds.sel(lon=slice(xmin, xmax), lat=slice(ymin, ymax))
    return ds
