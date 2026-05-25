"""
Processes AVISO data, namely FSLE and EDDY TRAJECTORY ATLAS
"""

from __future__ import annotations

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
from h2mare.downloader.commons import resolve_date_range
from h2mare.models import KeyVarConfigEntry
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import (
    chunk_dataset,
    convert360_180,
    ds_float64_to_float32,
)
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox, DateLike, DateRange, TimeResolution
from h2mare.utils.datetime_utils import normalize_date
from h2mare.utils.paths import resolve_download_path, resolve_store_path
from h2mare.utils.spatial import GridBuilder, haversine_min_distance_kdtree
from h2mare.validators import validate_time_resolution, validate_var_key

# ====================================================
# ================= EDDIES PROCESSOR =================
# ====================================================
# GRID CELL SIZE FOR PROCESSED DATA (IN DEGREES)
DX = 0.1
DY = 0.1

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

    def to_cartesian(lats: NDArray, lons: NDArray) -> NDArray:
        lat_rad = np.deg2rad(lats)
        lon_rad = np.deg2rad(lons)
        cos_lat = np.cos(lat_rad)
        return np.column_stack(
            (
                cos_lat * np.cos(lon_rad),  # x
                cos_lat * np.sin(lon_rad),  # y
                np.sin(lat_rad),  # z
            )
        )

    target_cartesian = to_cartesian(target_lats, target_lons)
    query_cartesian = to_cartesian(query_lats, query_lons)

    tree = cKDTree(target_cartesian)
    _, nearest_indices = tree.query(query_cartesian, workers=-1)

    return nearest_indices


def _group_dates(
    dates: pd.DatetimeIndex,
    groupby: TimeResolution,
) -> Iterator[tuple[int | tuple[int, int], pd.DatetimeIndex]]:
    """
    Yield (period_key, dates_in_period) pairs from a DatetimeIndex.

    For 'year':  period_key is the year (e.g. 2020)
    For 'month': period_key is a (year, month) tuple (e.g. (2020, 1))
    """
    if groupby == TimeResolution.YEAR:
        for year in dates.year.unique():
            yield int(year), dates[dates.year == year]
    elif groupby == TimeResolution.MONTH:
        for year in dates.year.unique():
            for month in dates[dates.year == year].month.unique():
                mask = (dates.year == year) & (dates.month == month)
                yield (int(year), int(month)), dates[mask]


class EDDIESProcessor:
    def __init__(
        self,
        *,
        var_key: str = "eddies",
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        download_root: Optional[Path] = None,
        time_resolution: TimeResolution = TimeResolution.YEAR,
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

        self.date_pattern = re.compile(self.var_config.pattern)
        self.type_pattern = re.compile(r"(anticyclonic|cyclonic)", re.IGNORECASE)

        if self.var_config.bbox is not None:
            self.bbox = BBox.from_tuple(self.var_config.bbox)

        self.time_resolution = validate_time_resolution(time_resolution)
        self.date_format: Literal["year", "date", "yearmonth"] = date_format

        # Initialise parent class with the Repository index class
        self.catalog = ZarrCatalog(self.var_key)

    def run(
        self,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        n_workers: int = 4,
        dx: float = DX,
        dy: float = DY,
    ) -> None:
        """
        Process downloaded files and writes.

        Args:
            start_date (Optional[DateLike], optional): Start date to process. Defaults to None and get's date from ZarrCatalog.
            end_date (Optional[DateLike], optional): End date to process. Defaults to Noneand get's date from ZarrCatalog.
            n_workers (int, optional): Number of workers for multiprocessing daily files. Defaults to 4.
            dx, dy: x,y cell size resp., in degrees
        """

        logger.info("Starting eddies processing")

        grid = self._get_gridded_data(dx, dy)
        records = self._get_downloaded_metadata()
        requested_ranges = self._resolve_all_ranges(records, start_date, end_date)

        # Determine the full set of years to process across all records
        all_dates = pd.date_range(
            min(r.start for r in requested_ranges.values()),
            max(r.end for r in requested_ranges.values()),
            freq="D",
        )

        for _, period_dates in _group_dates(all_dates, self.time_resolution):
            ds_list = []

            for record in records:
                eddy_type, _, path = record
                eddy_type_str = EDDY_TYPE_MAP[eddy_type]
                year_range = requested_ranges[eddy_type]

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
                del ds_list, ds_merged

        # logger.success("Completed!")

    # ============== PREPARE DATA ===============
    def _get_gridded_data(self, dx: float, dy: float) -> GridData:
        """
        Create a base grid with land mask from existing data (if ZarrCatalog exists) else uses ``GridBuilder``.
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

        if self.catalog.exists():
            with self.catalog.open_dataset() as ds:
                lat = ds.coords["lat"].values
                lon = ds.coords["lon"].values
        else:
            base_grid = GridBuilder(self.bbox, dx=dx, dy=dy).generate_grid()
            lat = base_grid.coords["lat"].values
            lon = base_grid.coords["lon"].values

        latlon_arr, sea_mask = create_base_grid(lat, lon)
        return GridData(lat, lon, latlon_arr, sea_mask)

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
        files = list(root_dir.rglob("*.nc"))

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
    ) -> dict[str, DateRange]:
        """Resolve requested date ranges for all records upfront."""
        return {
            eddy_type: self._resolve_date_range(download_range, start_date, end_date)
            for eddy_type, download_range, _ in records
        }

    def _resolve_date_range(
        self,
        download_range: DateRange,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
    ) -> DateRange:
        """
        Check if input date ranges overlap with downloaded file range.

        Args:
            path: path for file to check dates in file name
            start_date: Start date to process (None = use default or infer)
            end_date: End date to process (None = use default or infer)
        """

        start = normalize_date(start_date) if start_date else None
        end = normalize_date(end_date) if end_date else None

        requested_range = resolve_date_range(self.var_key, start, end)
        overlap = requested_range.intersection(download_range)

        if overlap:
            return overlap
        else:
            raise ValueError(
                f"Requested range {requested_range} does not overlap with available datasets"
            )

    def _prepare_raw_dataset(self, path: Path, dates: DateRange) -> xr.Dataset:
        """Preprocess original dataframe to subset by geo_extent with +10deg for distance calculations and by time range"""
        with xr.open_dataset(path) as ds:
            ds = ds[self.var_config.variables]
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

        worker = partial(
            _process_daily_static,
            ds=ds_raw,
            eddy_type_str=eddy_type_str,
            latlon1_arr=grid.latlon_arr,
            lat1=grid.lat,
            lon1=grid.lon,
            sea_mask=grid.sea_mask,
        )

        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(worker, dates)

        daily = [r for r in results if r is not None]
        if not daily:
            logger.warning(
                f"No valid results for {eddy_type_str} in year {dates[0].year}"
            )
            return None

        ds_year = xr.concat(daily, dim="time")
        return self._set_attrs(ds_year)

    def _set_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Set dataset variables attributes from yaml variable_attrs."""
        for var in ds.data_vars:
            var_info = get_settings().get_var_info(str(var))
            ds[var].attrs.update({key: val for key, val in var_info.items()})
        return ds


def _process_daily_static(
    date: pd.Timestamp,
    *,
    ds: xr.Dataset,
    eddy_type_str: str,
    latlon1_arr: NDArray,
    lat1: NDArray,
    lon1: NDArray,
    sea_mask: NDArray,
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
        all_lats = np.repeat(lat1, len(lon1))
        all_lons = np.tile(lon1, len(lat1))
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
def process_fsle(ds: xr.Dataset, var_config: KeyVarConfigEntry) -> xr.Dataset:
    """
    Process AVISO FSLE raw data, which is currently downloaded globally.

    Args:
        ds (xr.Dataset): dataset for processing
        var_config (KeyVarConfigEntry): Variable configuration entry containing variable names and bounding box for subsetting
    """
    ds = ds[var_config.variables]
    ds = convert360_180(ds)
    if var_config.bbox is not None:
        xmin, ymin, xmax, ymax = var_config.bbox
    return ds.sel(lon=slice(xmin, xmax), lat=slice(ymin, ymax))


if __name__ == "__main__":
    log_path = get_settings().LOGS_DIR / f"{Path(__file__).stem}.log"
    logger.add(log_path, level="INFO")

    ed_proc = EDDIESProcessor()
    ds = ed_proc.run()
    logger.debug(ds)
