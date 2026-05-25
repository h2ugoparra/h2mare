"""
Extract data based on csv or shapefile format files from datasets in zarr format.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Union, overload

import ephem
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  # registers .rio accessor on xarray objects
import xarray as xr
from loguru import logger
from scipy.spatial import KDTree

from h2mare import AppConfig, get_settings
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox
from h2mare.utils.logging_utils import log_time
from h2mare.utils.paths import resolve_store_path

_AUTO_INDEX_SENTINEL = "__row_id__"

# Module-level KDTree cache keyed on grid identity (shape + first/last values).
# All var_keys produced by this pipeline share the same 0.25° grid, so the tree
# is built once per process and reused across every extract_from_csv call.
_kdtree_cache: dict[tuple, tuple[KDTree, int, int]] = {}


# ===== BACKUP FUNC FOR INCOMPLETE EXTRACTIONS =====
def _keys_path(tmp_path: Path) -> Path:
    return tmp_path.with_suffix(".keys.json")


def _save_completed_keys(tmp_path: Path, keys: set[str]) -> None:
    dest = _keys_path(tmp_path)
    staging = dest.with_suffix(".tmp")
    with open(staging, "w") as f:
        json.dump(list(keys), f)
    staging.replace(dest)


def _load_completed_keys(tmp_path: Path) -> set[str]:
    path = _keys_path(tmp_path)
    if not path.exists():
        return set()
    with open(path) as f:
        return set(json.load(f))


def _extract_geometry(
    id: str, date, geom, ds: xr.DataArray | xr.Dataset, index_col: str
) -> dict:
    """
    Extract data and return as dictionary for a single geometry row.
    Retries up to 5 times in case of error (e.g ISError, Zarr chunk issues).

    Args:
        id (str): index value of the geometry row.
        date (): date value of the geometry row.
        geom (): geometry of the geometry row.
        ds (xr.DataArray | xr.Dataset): xarray object with dask arrays.

    Returns:
        dict: dictionary with index, variable names and extracted values.
    """
    max_attempts = 5
    base_wait = 1.0  # seconds

    is_dataset = isinstance(ds, xr.Dataset)
    data_vars: list[str] = [str(v) for v in ds.data_vars] if is_dataset else []
    single_var_name: str = str(ds.name) if (not is_dataset and ds.name) else "value"

    if date is not None:
        ds = ds.sel(time=date, method="nearest")

    for attempt in range(1, max_attempts + 1):
        try:
            clipped = ds.rio.clip([geom], drop=True, all_touched=True).mean()

            result: dict = {index_col: id}

            if not is_dataset:
                # Single variable
                result[single_var_name] = clipped.item()
            else:
                # Dataset: extract each variables
                for var in clipped.data_vars:
                    # Ensure scalar
                    result[var] = clipped[var].item()

            return result

        except (OSError, ValueError, RuntimeError) as e:
            wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 1)

            if attempt < max_attempts:
                logger.warning(
                    f"[Attempt {attempt}/{max_attempts}] id={id}, date={date}: {e}. "
                    f"Retrying in {wait:.1f}s."
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"Extraction permanently failed for id={id}, date={date} "
                    f"after {max_attempts} attempts: {e}"
                )

    # --- Return NaNs for failed geometry to preserve structure ---
    nan_result: dict = {index_col: id}
    if is_dataset:
        nan_result.update({var: float("nan") for var in data_vars})
    else:
        nan_result[single_var_name] = float("nan")

    return nan_result


def _extract_geometry_bathy(
    id: str, geom, ds: xr.DataArray | xr.Dataset, index_col: str
) -> dict:
    """
    Extract bathymetry data and return as dictionary for a single geometry row.
    Retries up to 5 times in case of error (e.g ISError, Zarr chunk issues).

    Args:
        id (str): index value of the geometry row.
        geom (): geometry of the geometry row.
        ds (xr.DataArray | xr.Dataset): xarray object.

    Returns:
        dict: dictionary with index, variable names and extracted values.
    """
    max_attempts = 5
    base_wait = 1.0  # seconds

    is_dataset = isinstance(ds, xr.Dataset)
    data_vars: list[str] = [str(v) for v in ds.data_vars] if is_dataset else []
    single_var_name: str = str(ds.name) if (not is_dataset and ds.name) else "value"

    for attempt in range(1, max_attempts + 1):
        try:
            clipped = ds.rio.clip([geom], drop=True, all_touched=True)
            mean_ds = clipped.mean(dim=None)
            std_ds = clipped.std(dim=None)

            result: dict = {index_col: id}

            if is_dataset:
                for var in clipped.data_vars:
                    result[f"{var}"] = mean_ds[var].item()
                    result[f"{var}_std"] = std_ds[var].item()
            else:
                result[single_var_name] = mean_ds.item()
                result[f"{single_var_name}_std"] = std_ds.item()

            return result

        except (OSError, ValueError, RuntimeError) as e:
            wait = base_wait * (2 ** (attempt - 1)) + random.uniform(0, 1)

            if attempt < max_attempts:
                logger.warning(
                    f"[Attempt {attempt}/{max_attempts}] id={id}: {e}. "
                    f"Retrying in {wait:.1f}s."
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"Extraction permanently failed for id={id} "
                    f"after {max_attempts} attempts: {e}"
                )

    # --- Return NaNs for failed geometry to preserve structure ---
    nan_result: dict = {index_col: id}
    if is_dataset:
        nan_result.update({var: float("nan") for var in data_vars})
    else:
        nan_result[single_var_name] = float("nan")
        nan_result[f"{single_var_name}_std"] = float("nan")

    return nan_result


@log_time
def load_dataset_to_memory(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    return ds.compute()


class Extractor:
    def __init__(
        self,
        file_path: Union[Path, gpd.GeoDataFrame, pd.DataFrame],
        *,
        time_col: Optional[str] = None,
        index_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        lat_col: Optional[str] = None,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Union[str, Path]] = None,
        crs: int | None = 4326,
    ):
        """
        Extract data from shp/csv file_path of open file

        Parameters:
            file_path (Union[Path, gpd.GeoDataFrame, pd.DataFrame]): Data for extraction
            time_col (str): Name of time column. Defaults to "time".
            index_col (str, optional): Name of index column. Defaults to "row_id" if not provided.
            lon_col (str, optional): Name of longitude column. Defaults to "lon".
            lat_col (str, optional): Name of latitude column. Defaults to "lat".
            app_config (AppConfig, optional): Dataclass with environmental data specifics. Defaults to cfg.
            store_root (Union[str, Path], optional): Path for environmental data main folder. Defaults to STORE_ROOT.
            crs (int | None, optional): Projection EPSG code for geometry extraction. Defaults to 4326.

        """
        self.time_col = time_col if time_col is not None else "time"
        self.index_col = index_col if index_col is not None else _AUTO_INDEX_SENTINEL
        self.lon_col = lon_col if lon_col is not None else "lon"
        self.lat_col = lat_col if lat_col is not None else "lat"
        self.crs = crs

        self.app_config = app_config or get_settings().app_config

        self.store_root = (
            Path(store_root) if store_root is not None else get_settings().STORE_ROOT
        )

        data_orig = self._resolve_file_format(file_path)
        data_orig = self._resolve_index(data_orig)
        self.data = self._prepare_data(data_orig)
        self.data = self._resolve_time_col(self.data)

    # =================== DATA PREPARATION =====================

    def _resolve_time_col(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve time column to date or datetime based on time variance.

        Logic:
            - If time_col strings contain no time component → keep as date.
            - If time_col contains datetimes:
                - If time component is identical across all rows → truncate to date.
                - If time component varies → keep full datetime.
        """
        data = data.rename(columns={self.time_col: "time"})

        # Check on raw strings BEFORE parsing — avoids 00:00:00 false negative
        raw = data["time"].astype(str)
        has_time_component = raw.str.contains(r"\d{2}:\d{2}:\d{2}", regex=True).any()

        data["time"] = pd.to_datetime(data["time"], utc=True).dt.tz_convert(None)

        if has_time_component:
            time_is_uniform = data["time"].dt.time.nunique() == 1
            if time_is_uniform:
                logger.info("Uniform time component detected. Truncating to date.")
                data["time"] = data["time"].dt.normalize()
            else:
                logger.info("Variable time component detected. Keeping full datetime.")
        else:
            logger.info("No time component detected. Keeping as date.")

        return data

    def _resolve_file_format(
        self, file_path: Union[Path, gpd.GeoDataFrame, pd.DataFrame]
    ):
        """determine input type and load accordingly"""

        if isinstance(file_path, gpd.GeoDataFrame):
            data_base = file_path.copy()
            self.input_type = "shp"

        elif isinstance(file_path, pd.DataFrame):
            data_base = file_path.copy()
            self.input_type = "csv"

        else:
            file_path = Path(file_path)
            suffix = file_path.suffix.lower()

            if suffix == ".shp":
                data_base = gpd.read_file(file_path)
                self.input_type = "shp"

            elif suffix == ".csv":
                data_base = pd.read_csv(file_path)
                self.input_type = "csv"
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

        return data_base

    def _resolve_index(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Resolve index_col in shp and csv dataframes.

        If no index_col is provided (sentinel), a sequential index is created
        and stored under the private name '__row_id__', which is preserved as
        the DataFrame index throughout the pipeline — never dropped or saved
        to output files.

        Raises:
            ValueError: if provided index_col not found in data.columns
        """
        if self.index_col == _AUTO_INDEX_SENTINEL:
            if _AUTO_INDEX_SENTINEL in data.columns:
                # Reuse existing __row_id__ from a previous run — do not recreate
                logger.info(
                    f"Found existing '{_AUTO_INDEX_SENTINEL}' column. Reusing as index."
                )
                data = data.set_index(_AUTO_INDEX_SENTINEL)
            else:
                logger.info(
                    "No index column name provided. Creating sequential '__row_id__' indexation."
                )
                data = data.reset_index(drop=True)
            data.index.name = _AUTO_INDEX_SENTINEL
        else:
            if self.index_col not in data.columns:
                raise ValueError("Index column name not found in data attributes.")
            data = data.set_index(self.index_col)
        return data

    def _prepare_data(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Prepares data according to input_type and returns a subseted df/gdf with only essential cols for extraction.
            - csv: df['time', 'lon', 'lat']
            - shp: gdf['time', 'geometry']

        """
        cols = {self.time_col: "time", self.lon_col: "lon", self.lat_col: "lat"}

        if self.time_col not in data.columns:
            raise ValueError(
                f"Time column '{self.time_col}' not found in data attributes."
            )

        if self.input_type == "csv":
            if self.lon_col not in data.columns or self.lat_col not in data.columns:
                raise ValueError(
                    f"CSV must contain '{self.lon_col}' and '{self.lat_col}' columns."
                )

            data = data.rename(columns=cols)[["time", "lon", "lat"]].copy()

        elif self.input_type == "shp":
            if self.crs is None:
                raise ValueError("CRS must be provided for shapefile input.")

            if isinstance(data, gpd.GeoDataFrame):
                data = data.copy()
                if data.crs is None:
                    data.set_crs(self.crs, inplace=True)
                elif data.crs.to_epsg() != self.crs:
                    data = data.to_crs(self.crs)
            data = data.rename(columns=cols)[["time", "geometry"]].copy()

        else:
            raise ValueError(
                f"Invalid input_type: {self.input_type}. Must be 'csv' or 'shp'."
            )

        return data

    def _define_bbox(self, data: pd.DataFrame | gpd.GeoDataFrame) -> BBox:
        """
        Returns bbox from data according to input_type ('csv' or 'shp').
        """
        if self.input_type == "csv":
            bounds = (
                data["lon"].min(),
                data["lat"].min(),
                data["lon"].max(),
                data["lat"].max(),
            )

        elif self.input_type == "shp":
            bounds = tuple(data.total_bounds)

        return BBox.from_tuple(bounds)

    def _resolve_coverage(self, catalog: ZarrCatalog) -> list[pd.Timestamp]:
        """
        Resolve input/store data space/time coverage limits.

        Raises:
            ValueError: if ``catalog.get_time_coverage()`` or ``get_bbox()`` returns None

        Returns:
            list[pd.Timestamp]: List of unique dates within store limits if out of range, else returns None.
        """
        # get storage coverage
        store_dates = catalog.get_time_coverage()
        store_bbox = catalog.get_bbox()

        if store_dates is None or store_bbox is None:
            raise ValueError(f"No coverage data for {catalog.var_key}")

        start_store = pd.Timestamp(store_dates.start)
        end_store = pd.Timestamp(store_dates.end)

        # Input Data coverage
        dates = self._extract_unique_dates(self.data)
        start, end = dates.min(), dates.max()
        bounds = self._define_bbox(self.data)

        if not bounds.overlaps(store_bbox):
            logger.warning(
                f"Data input bbox does not overlap with store data for {catalog.var_key}"
            )

        start_date = pd.to_datetime(max(start, start_store))
        end_date = pd.to_datetime(min(end, end_store))

        if start < start_store:
            clipped = dates[dates < start_store]
            logger.warning(
                f"{catalog.var_key}: {len(clipped)} date(s) before store coverage clipped "
                f"({clipped.min().date()} -> {clipped.max().date()} | store starts {start_store.date()})"
            )
        if end > end_store:
            clipped = dates[dates > end_store]
            logger.warning(
                f"{catalog.var_key}: {len(clipped)} date(s) after store coverage clipped "
                f"({clipped.min().date()} -> {clipped.max().date()} | store ends {end_store.date()})"
            )

        return sorted(dates[(dates >= start_date) & (dates <= end_date)])

    def _extract_unique_dates(
        self, data: gpd.GeoDataFrame | pd.DataFrame
    ) -> pd.DatetimeIndex:
        """Extract unique dates from the GeoDataFrame's time column."""
        if "time" in data.columns:
            data["time"] = pd.to_datetime(data["time"])
            return pd.DatetimeIndex(data["time"]).drop_duplicates()
        else:
            raise ValueError("Time column 'time' not found in shapefile attributes.")

    # ===================  PROCESS DATA ===================

    def process_single_varkey(
        self, var_key: str, vars: str | list[str] | None = None, n_workers: int = 8
    ) -> pd.DataFrame:
        """
        Run extraction process for a single var_key.

        Parameters:
            var_key : str
                Key to identify variable in config.
            vars : str, list[str], None
                Specific variables for extraction associated with the specified var_key. This avoids extracting all vars inside the var_key.
            n_workers : int, optional
                Number of parallel workers for geometries (shp) extraction, by default 8.

        Returns:
            pd.DataFrame with extracted values.
        """
        vars = [vars] if isinstance(vars, str) else vars

        # Moon and bathy first since they do not need data from ZarCatalog
        if var_key == "moon":
            return self._extract_moon_phase(self.data)

        if var_key == "bathy":
            return self._extract_bathy(self.data)

        # Remaining Key Variable
        vr_catalog = ZarrCatalog(var_key)

        # Resolve coverage and subset data if necessary
        dates_resolved = self._resolve_coverage(vr_catalog)
        mask = self.data["time"].between(min(dates_resolved), max(dates_resolved))
        data_resolved = self.data.loc[mask]
        bounds = self._define_bbox(data_resolved)

        logger.info(f"Extracting {var_key} data")
        logger.info(
            f"{data_resolved.shape[0]} samples | "
            f"{min(dates_resolved).date()} -> {max(dates_resolved).date()} | "
            f"{bounds}"
        )

        ## Get datasets covering the dates values and bounds
        ds = vr_catalog.open_dataset(dates=dates_resolved, bbox=bounds)

        if vars:
            ds = ds[vars]

        ds = ds.sortby("time")

        # Variable-specific preprocessing
        if var_key == "o2":
            ds = self._preprocess_o2(ds)

        if self.input_type == "shp":
            if not isinstance(data_resolved, gpd.GeoDataFrame):
                raise TypeError("Data must be a GeoDataFrame for shapefile extraction")

            ds = self.ensure_crs(data_resolved, ds)

            # Rename coordinates to avoid error in rioxarray clip
            if var_key in ["fsle", "eddies"]:
                ds = ds.rename({"lon": "x", "lat": "y"})

            return self.extract_from_shp(
                data_resolved, ds, self.index_col, n_workers=n_workers
            )

        elif self.input_type == "csv":
            return self.extract_from_csv(data_resolved, ds, self.index_col)

        raise ValueError(f"Unsupported input_type: {self.input_type}")

    @overload
    def run(
        self,
        var_dict: Optional[
            Union[str, list[str], dict[str, str | list[str] | None]]
        ] = ...,
        output_path: None = ...,
        n_workers: int = ...,
    ) -> pd.DataFrame: ...

    @overload
    def run(
        self,
        var_dict: Optional[
            Union[str, list[str], dict[str, str | list[str] | None]]
        ] = ...,
        output_path: str | Path = ...,
        n_workers: int = ...,
    ) -> None: ...

    @log_time
    def run(
        self,
        var_dict: Optional[
            Union[str, list[str], dict[str, str | list[str] | None]]
        ] = None,
        output_path: Optional[str | Path] = None,
        n_workers: int = 8,
    ) -> pd.DataFrame | None:
        """
        Extract all or specified var_key and respective variables, and save dataframe with extracted data.

        Args:
            var_dict (str | list[str] | dict[str, str  |  list[str]  |  None] | None, optional: Var_key str or list of strings or dict specifiying vars in var_key.
                Defaults to None, extracting all available var_keys and respective variables.
            output_path (str | Path | None): Path to save file. If None, it returns a dataframe with all results.
            n_workers (int, optional): Workers for shp parallel processing. Defaults to 8.

        Example:
            >>> var_dict = {
            >>>     'seapodym': [],
            >>>     'radiation': ['tisr', 'ssrd', 'slhf'],
            >>>     }
            >>>
            >>> extractor = Extractor(file_path=input_path, time_col='ls_date', index_col='idlance')
            >>> results = extractor.run(output_path, var_dict=var_dict, n_workers=12)
        """
        logger.info(f"Starting extraction process with {n_workers} workers.")

        var_dict = self._normalize_var_dict(var_dict)

        tmp_path = get_settings().INTERIM_DIR / "extraction_checkpoint.feather"

        if tmp_path.exists():
            logger.warning(f"Found checkpoint file: {tmp_path}, resuming.")
            df_processed = pd.read_feather(tmp_path).set_index(self.index_col)
            if df_processed.index.duplicated().any():
                logger.warning(
                    "Duplicate index values found in checkpoint — keeping first occurrence."
                )
                df_processed = df_processed[
                    ~df_processed.index.duplicated(keep="first")
                ]
            completed_keys = _load_completed_keys(tmp_path)
        else:
            df_processed = self.data.copy()
            completed_keys = set()

        all_succeeded = True

        for var_key, vars_ in var_dict.items():
            if var_key == "h2ds":
                continue

            if var_key in completed_keys:
                logger.info(f"Skipping {var_key}: already extracted.")
                continue

            try:
                result = self.process_single_varkey(
                    var_key=var_key, vars=vars_, n_workers=n_workers
                )

                result.drop(
                    columns=["time", "lat", "lon", "geom"],
                    errors="ignore",
                    inplace=True,
                )
                if result.index.duplicated().any():
                    logger.warning(
                        f"Duplicate index values in '{var_key}' result — keeping first occurrence."
                    )
                    result = result[~result.index.duplicated(keep="first")]
                df_processed = df_processed.join(result)

                # Mark var_key as completed and save checkpoint atomically
                completed_keys.add(var_key)
                staging = tmp_path.with_suffix(".tmp")
                df_processed.reset_index().to_feather(staging)
                staging.replace(tmp_path)
                _save_completed_keys(tmp_path, completed_keys)
                logger.info(f"Checkpoint saved to {tmp_path}")

            except Exception as e:
                logger.error(f"Error processing '{var_key}': {e}")
                all_succeeded = False
                continue

        logger.success("Extraction completed!")
        logger.info("=" * 60)
        logger.info("  Number of null values per variable:")
        result_cols = [c for c in df_processed.columns if c not in self.data.columns]
        for col, count in df_processed[result_cols].isnull().sum().items():
            logger.info(f"  {col}: {count}")
        logger.info("=" * 60)

        if all_succeeded:
            tmp_path.unlink(missing_ok=True)
            _keys_path(tmp_path).unlink(missing_ok=True)
        else:
            logger.warning(
                "Some var_keys failed. Checkpoint files preserved for resume."
            )

        if output_path is not None:
            self._save_results(df_processed, Path(output_path))
            return

        return df_processed

    @staticmethod
    def _nearest_grid_indices(
        ds: xr.Dataset | xr.DataArray,
        query_lons: np.ndarray,
        query_lats: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat_idx, lon_idx) arrays for each query point using a KDTree.

        The tree is cached at module level keyed on grid identity (shape + boundary
        values), so it is built only once per unique grid across all var_key calls.
        """
        lons = ds.lon.values  # (n_lon,)
        lats = ds.lat.values  # (n_lat,)

        cache_key = (
            lons.shape,
            float(lons[0]),
            float(lons[-1]),
            lats.shape,
            float(lats[0]),
            float(lats[-1]),
        )

        if cache_key not in _kdtree_cache:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            tree = KDTree(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]))
            _kdtree_cache[cache_key] = (tree, len(lats), len(lons))

        tree, n_lats, n_lons = _kdtree_cache[cache_key]
        _, flat_idx = tree.query(np.column_stack([query_lons, query_lats]))
        lat_idx, lon_idx = np.unravel_index(flat_idx, (n_lats, n_lons))
        return np.asarray(lat_idx), np.asarray(lon_idx)

    @staticmethod
    def _nearest_time_indices(
        ds: xr.Dataset | xr.DataArray,
        query_times: np.ndarray,
    ) -> np.ndarray:
        """Return the nearest time index for each query timestamp via searchsorted.

        Picks whichever grid step (left or right of the insertion point) is
        closer, matching xarray's method='nearest' semantics exactly.
        """
        grid_times = ds.time.values.astype("int64")
        q = pd.to_datetime(query_times).to_numpy().astype("int64")

        right = np.searchsorted(grid_times, q).clip(0, len(grid_times) - 1)
        left = (right - 1).clip(0, len(grid_times) - 1)
        return np.where(
            np.abs(grid_times[right] - q) <= np.abs(grid_times[left] - q),
            right,
            left,
        )

    @staticmethod
    def extract_from_csv(
        data: pd.DataFrame, ds: xr.Dataset | xr.DataArray, index_col: str
    ) -> pd.DataFrame:
        """
        Point extraction from a dataframe. If run as staticmethod, time, lat and lon cols should be named 'time', 'lat' and 'lon', resp.

        Uses a KDTree for spatial nearest-neighbour lookup (works on regular and
        irregular grids) and numpy searchsorted for time, then selects with
        isel() (integer indexing) which is faster than coordinate-based sel().

        Parameters:
            ds (xr.Dataset | xr.DataArray): dataset with coords lon, lat and optionally time.

        Returns:
            pd.DataFrame: extracted variables with previous index set
        """
        valid = data[data["lon"].notna() & data["lat"].notna()]
        coords = {index_col: valid.index}

        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, valid["lon"].to_numpy(), valid["lat"].to_numpy()
        )

        isel_kwargs: dict = {
            "lon": xr.DataArray(lon_idx, dims=index_col, coords=coords),
            "lat": xr.DataArray(lat_idx, dims=index_col, coords=coords),
        }

        if "time" in ds.coords:
            time_idx = Extractor._nearest_time_indices(ds, valid["time"].values)  # type: ignore
            isel_kwargs["time"] = xr.DataArray(time_idx, dims=index_col, coords=coords)

        ds = load_dataset_to_memory(ds.isel(**isel_kwargs))
        result = ds.to_dataframe()
        return result.reindex(data.index)

    @staticmethod
    def extract_from_shp(
        data: gpd.GeoDataFrame,
        ds: xr.Dataset | xr.DataArray,
        index_col: str,
        n_workers: int = 8,
    ) -> pd.DataFrame:
        """
        Extract data from shapefile using multiprocessing starmap.

        Args:
            gdf (gpd.GeoDataFrame): geodataframe with geometries and time column.
            ds (xr.Dataset): xarray dataset with dask arrays.
            n_workers (int, optional): Number of workers for parallel processing of geometries. Defaults to 8.

        Returns:
            pd.DataFrame with extracted values.
        """

        # Clip to the combined spatial envelope of all geometries before pulling
        # data into memory — reduces what gets computed from the dask graph
        xmin, ymin, xmax, ymax = data.total_bounds
        lat_coord = "lat" if "lat" in ds.coords else "y"
        lon_coord = "lon" if "lon" in ds.coords else "x"
        if lat_coord in ds.coords and lon_coord in ds.coords:
            ds = ds.sel({lat_coord: slice(ymin, ymax), lon_coord: slice(xmin, xmax)})

        ds_computed = load_dataset_to_memory(ds)

        has_time = "time" in ds.coords

        if has_time:
            tasks = [
                (id, date, geom, ds_computed, index_col)
                for id, date, geom in zip(data.index, data.time, data.geometry)
            ]
        else:
            tasks = [
                (id, None, geom, ds_computed, index_col)
                for id, geom in zip(data.index, data.geometry)
            ]

        out = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_extract_geometry, *task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    out.append(result)
        return pd.DataFrame(out).set_index(index_col)

    def _extract_bathy(
        self, data: pd.DataFrame | gpd.GeoDataFrame, n_workers: int = 8
    ) -> pd.DataFrame:
        """
        Extract bathymetry data for geometries (shp - original 15s res, calculates mean and std where the geom touches)
        and points (csv - from coarser 0.25deg res with mean and std already calculated).
        """
        vkey = "bathy"
        var_cfg = self.app_config.variables[vkey]
        store_root = resolve_store_path(var_cfg)

        if self.input_type == "shp":
            if var_cfg.data_file_hires is None:
                raise ValueError(
                    "bathy config entry is missing required 'data_file_hires' field"
                )
            data_path = store_root / var_cfg.data_file_hires
        elif self.input_type == "csv":
            if var_cfg.data_file is None:
                raise ValueError(
                    "bathy config entry is missing required 'data_file' field"
                )
            data_path = store_root / var_cfg.data_file

        bounds = self._define_bbox(data)
        logger.info(
            f"Extracting {vkey.upper()} data | {self.data.shape[0]} rows | {bounds}"
        )

        ds = xr.open_dataset(data_path)
        ds_bbox = BBox.from_dataset(ds)

        if not bounds.overlaps(ds_bbox):
            logger.warning(
                f"Data input bbox does not overlap with store data for {vkey}"
            )

        if isinstance(data, gpd.GeoDataFrame):
            ds = (
                ds.sel(
                    lon=slice(bounds.xmin, bounds.xmax),
                    lat=slice(bounds.ymin, bounds.ymax),
                ).rename({"z": "bathy", "lon": "x", "lat": "y"})
            ).compute()

            ds = self.ensure_crs(data, ds)

            tasks = [
                (id, geom, ds, self.index_col)
                for id, geom in zip(data.index, data.geometry)
            ]

            out = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_extract_geometry_bathy, *task) for task in tasks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        out.append(result)

        else:
            ds = (
                ds.sel(
                    lon=slice(bounds.xmin, bounds.xmax),
                    lat=slice(bounds.ymin, bounds.ymax),
                )
            ).compute()

            out = ds.sel(
                lon=xr.DataArray(
                    data["lon"].values,
                    dims=self.index_col,
                    coords={self.index_col: data.index},
                ),
                lat=xr.DataArray(
                    data["lat"].values,
                    dims=self.index_col,
                    coords={self.index_col: data.index},
                ),
                method="nearest",
            ).to_dataframe()

        if isinstance(out, list):
            return pd.DataFrame(out).set_index(self.index_col)
        return out

    def _preprocess_o2(
        self, ds: xr.Dataset | xr.DataArray, depth_intervals: list[int] = [0, 100, 500]
    ) -> xr.Dataset:
        """Preprocess and return a dataset with o2 variables divided by depth_intervals values."""
        o2_sel = ds["o2"].sel(depth=depth_intervals, method="nearest")
        ds_o2 = xr.Dataset(
            {
                f"o2_{int(d.values)}": o2_sel.sel(depth=d)
                .squeeze(drop=True)
                .drop_vars("depth")
                for d in o2_sel.depth
            }
        )
        rename_map = {
            f"o2_{int(d.values)}": f"o2_{target}"
            for d, target in zip(o2_sel.depth, depth_intervals)
        }
        return ds_o2.rename(rename_map)

    def _extract_moon_phase(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Extract moon ilumination from ephem library. Lat/lon values are averaged.

        Returns:
            pd.DataFrame: _description_
        """
        bounds = self._define_bbox(data)

        logger.info(f"Extracting 'MOON' data | {data.shape[0]} rows | {bounds}")

        lat = (bounds.ymin + bounds.ymax) / 2
        lon = (bounds.xmin + bounds.xmax) / 2

        observer = ephem.Observer()
        observer.lat = str(lat)
        observer.lon = str(lon)

        result = []
        for id, date in zip(data.index, data.time):
            observer.date = date
            moon = ephem.Moon(observer)
            result.append({self.index_col: id, "moon_phase": moon.phase})
        return pd.DataFrame(result).set_index(self.index_col)

    # ======================= HELPERS =========================
    def _normalize_var_dict(
        self,
        var_dict: Optional[
            Union[str, list[str], dict[str, str | list[str] | None]]
        ] = None,
    ) -> dict[str, str | list[str] | None]:
        """
        Helper function to resolves var_dict arg from ``run()``

        Args:
            var_dict (Optional[Union[str, list[str], dict[str, str  |  list[str]  |  None]]], optional): _description_. Defaults to None.

        Raises:
            TypeError: if type list[str] but elements not str
            TypeError: No valid var_dict

        Returns:
            dict[str, str | list[str] | None]: _description_
        """
        if var_dict is None:
            all_var_keys = self.app_config.variables.keys()
            logger.info(
                f"No variables provided. Using all key variables from config: "
                f"{list(all_var_keys)}"
            )
            return {k: None for k in all_var_keys}

        elif isinstance(var_dict, dict):
            return var_dict

        # single var_key
        elif isinstance(var_dict, str):
            return {var_dict: None}

        # list of var_keys
        elif isinstance(var_dict, list):
            if not all(isinstance(v, str) for v in var_dict):
                raise TypeError("All elements in var_dict list must be strings")
            return {vd: None for vd in var_dict}
        else:
            raise TypeError("Provide a valid var_dict")

    def ensure_crs(
        self, data: gpd.GeoDataFrame, ds: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        """Ensure the CRS of the dataset is the same as the prepared GeoDataFrame's."""
        if ds.rio.crs != data.crs:
            return ds.rio.write_crs(data.crs, inplace=True)
        return ds

    def remove_duplicated_cols(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compares column names from two dataframes and removes duplicated columns from df1.

        Parameters:
            df1, df2 (pd.DataFrame): Older/existing data (df1) from which columns will be removed if present in newer data (df2)

        Returns:
            (pd.DataFrame) with removed duplicated cols
        """
        overlapping_cols = df1.columns.intersection(df2.columns)

        if len(overlapping_cols) > 0:
            logger.warning(
                f"Removing overlapping columns from existing dataframe: {list(overlapping_cols)}"
            )
            return df1.drop(columns=overlapping_cols)
        else:
            return df1

    # ========================  I/O =========================
    def _save_results(self, result: pd.DataFrame, output_path: Path) -> None:
        """
        Save result dataframe to output_path. Checks if exists, and if so, remove duplicated columns.

        Parameters:
            result (pd.DataFrame): Dataframe with extracted data
            output_path (Path): Path to save csv file.
        """
        logger.info(f"Saving results to {output_path}")

        if output_path.exists():
            existing_df = pd.read_csv(output_path, index_col=self.index_col)
            logger.warning(
                f"Output_path already exists. Loading {output_path} with {len(existing_df)} observations."
            )
            existing_df = self.remove_duplicated_cols(existing_df, result)
            result = existing_df.join(result, how="left")

        result = result.reset_index(drop=False)

        result.to_csv(output_path, index=False)
        logger.success("Results saved")
