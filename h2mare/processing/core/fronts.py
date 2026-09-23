"""
Detection of oceanic fronts with the Belkin–O'Reilly algorithm (BOA), and the
distance from every sea cell to the nearest front detected that day.

Which variables get a front-distance layer, and at what gradient threshold, is
declared per var_key in ``boa_fronts`` (:class:`h2mare.models.BOAFrontSpec`)
rather than decided here: the algorithm is the same for sst and chl, only the
field and the threshold differ. :func:`apply_boa_fronts` is what the convert
step calls; :class:`FrontProcessor` runs one layer.

Detection is eager — one BOA pass and one KD-tree query per day, across a
process pool — while everything around it in convert is lazy. Each month is
written to a staging Zarr under ``INTERIM_DIR`` and the layer is handed back as
a lazy view of it, so a whole period's days are never held in memory at once. A
year of sst at 0.05° is ~3.7 GB of distances; collecting it in one list, as this
used to, cost that again to concatenate.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil
import time
from collections.abc import Iterator
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from global_land_mask import globe
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter, median_filter, minimum_filter, sobel

from h2mare import get_settings
from h2mare.models import BOAFrontSpec
from h2mare.utils.spatial import haversine_min_distance_kdtree

#: Pool size when a ``boa_fronts`` entry does not set ``n_workers``.
DEFAULT_N_WORKERS = 10

#: Marks the staging stores under INTERIM_DIR, so they can be swept by name.
_STAGE_SUFFIX = ".zarr.stage"

# ============================================
#   FUNCTIONS FOR FRONTAL DISTANCES CALCULATION
# ==============================================


def create_base_grid(
    lat: NDArray[np.float64], lon: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool]]:
    """
    Create a base grid with land mask.

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
        np.column_stack((lat_grid.flatten()[mask_flat], lon_grid.flatten()[mask_flat])),
        sea_mask,
    )


def filt5(ingrid: NDArray[np.float64]) -> NDArray[np.int8]:
    """
    Identify local extrema (maxima or minima) within a 5x5 window.

    For each pixel, it checks whether it's the
    maximum or minimum value within a 5x5 neighborhood. If so, it is flagged
    with 1, otherwise 0.

    Args:
        ingrid: 2D input array (e.g., SST field) containing values to be analyzed.
            NaN values are ignored by treating them as constant fill values.

    Returns:
        Binary mask array of the same shape as `ingrid` where:
        - 1 indicates the pixel is a local maximum or minimum within its 5x5
            neighborhood.
        - 0 otherwise.
    """
    max5 = maximum_filter(ingrid, size=5, mode="constant", cval=np.nan)
    min5 = minimum_filter(ingrid, size=5, mode="constant", cval=np.nan)
    return ((ingrid == max5) | (ingrid == min5)).astype(np.int8)


def filt3(ingrid: NDArray[np.float64], grid5: NDArray[np.int8]) -> NDArray[np.float64]:
    """
    Apply a 3x3 filter to refine front detection after filt5.

    This function smooths the input field with a median filter (3x3), then
    checks whether each pixel is a local maximum or minimum within its 3x3
    neighborhood. Only pixels not already flagged by `filt5` are considered.
    If a pixel meets the condition, its value is replaced with the smoothed
    median value; otherwise, it retains its original value.

    Args:
        ingrid: 2D input array (e.g., SST field).
        grid5 : Binary mask from `filt5`, used to exclude pixels already flagged
        as extrema in the 5x5 window.

    Returns:
        Output array of the same shape as `ingrid`, where some values are
        replaced with their 3x3 median-filtered equivalents if they represent
        a local extremum not captured by `filt5`.
    """
    smoothed = median_filter(ingrid, size=3, mode="nearest")
    max3 = maximum_filter(ingrid, size=3, mode="constant", cval=np.nan)
    min3 = minimum_filter(ingrid, size=3, mode="constant", cval=np.nan)
    center = ingrid
    out = np.where(
        (grid5 == 0) & ((center == max3) | (center == min3)), smoothed, center
    )
    return out


def boa(
    lon: NDArray[np.float64],
    lat: NDArray[np.float64],
    ingrid: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    """
    Detect oceanic fronts based on Belkin O'Reilly algorithm.

    This function applies the following steps:
    1. Replace NaNs with zeros.
    2. Detect candidate extrema with `filt5` and `filt3`.
    3. Compute gradients in x and y directions using Sobel filters.
    4. Combine gradients to form a front-intensity map.
    5. Apply a threshold to binarize the front detection.
    6. Extract coordinates of detected front pixels.

    Args:
        lon: 1D array of longitudes corresponding to the x-axis of `ingrid`.
        lat: 1D array of latitudes corresponding to the y-axis of `ingrid`.
        ingrid: 2D input field (e.g., SST).
        threshold: Threshold value for gradient magnitude, in `ingrid`'s own
            units. Pixels with magnitude above this threshold are classified as
            part of a front.

    Returns:
        np.ndarray shape (N, 2): Array of coordinates (lat, lon) for the N detected frontal pixels.
    """
    ingrid = np.nan_to_num(ingrid, nan=0)
    grid5 = filt5(ingrid)
    grid35 = filt3(ingrid, grid5)

    tgx = sobel(grid35, axis=1)
    tgy = sobel(grid35, axis=0)
    front = np.hypot(tgx, tgy)

    front = np.where(front >= threshold, 1, 0)
    iy, ix = np.where(front == 1)
    return np.column_stack((lat[iy], lon[ix]))


def BOA_application(data_xarray: xr.DataArray, threshold: float):
    """
    Apply Belkin O'Reilly front-detection algorithm to an xarray DataArray.

    This wrapper function extracts lat/lon grids and SST values from an
    xarray DataArray and applies `boa`.

    Parameters
    ----------
    data_xarray : xr.DataArray
        Input SST data with dimensions (lat, lon).
    threshold : float
        Threshold for front detection (gradient magnitude).

    Returns
    -------
    np.ndarray, shape (N, 2)
        Array of coordinates (lat, lon) for the N detected frontal pixels.
    """
    lat = data_xarray["lat"].values
    lon = data_xarray["lon"].values

    ingrid = data_xarray.values

    return boa(lon=lon, lat=lat, ingrid=ingrid, threshold=threshold)


# ==============================================
#   STAGING
# ==============================================


def stage_path(var_key: str, name: str) -> Path:
    """Staging store one layer of *var_key* is written to."""
    return get_settings().INTERIM_DIR / f".{var_key}_{name}{_STAGE_SUFFIX}"


def clear_staging(var_key: str) -> int:
    """
    Remove *var_key*'s staging stores, and report how many there were.

    Called once a period has been written, and again before a run starts so
    that whatever a killed run left behind goes with it — these are full
    copies of a period's layer, so leaving them costs real disk.

    Like the eddies staging, this assumes no second conversion of the same
    var_key is running alongside, which would already be unsafe: both would
    write the same period file.
    """
    removed = 0
    for leftover in get_settings().INTERIM_DIR.glob(f".{var_key}_*{_STAGE_SUFFIX}"):
        logger.debug(f"[{var_key}] Removing front staging {leftover.name}")
        shutil.rmtree(leftover, ignore_errors=True)
        removed += 1
    return removed


def _month_batches(times: pd.DatetimeIndex) -> Iterator[pd.DatetimeIndex]:
    """*times* split into calendar months, in axis order."""
    months = times.to_period("M")
    for month in months.unique():
        yield times[months == month]


def _stage_chunks(da: xr.DataArray) -> tuple[int, ...]:
    """
    Staging chunks: one day of time, spatial tiles as the store uses.

    A day per chunk means every month appends on a chunk boundary whatever its
    length, and tiles mean reading the layer back into the store's own layout
    is a concatenation of tiles rather than a reshuffle of whole fields.
    """
    return tuple(1 if d == "time" else min(256, n) for d, n in da.sizes.items())


def _detect_day(
    date: pd.Timestamp,
    *,
    da: xr.DataArray,
    threshold: float,
    name: str,
    latlon1_arr: NDArray[np.float64],
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    sea_mask: NDArray[np.bool],
) -> xr.DataArray:
    """
    One day's distances to the nearest front, as a (1, lat, lon) DataArray.

    Module level, and taking everything it needs as arguments, so a pool worker
    pickles a plain function rather than a processor holding config.
    """
    latlon2_arr = BOA_application(da.sel(time=date), threshold)

    min_distance = haversine_min_distance_kdtree(latlon1_arr, latlon2_arr)
    # float32, like the field the distances describe: a distance in km has far
    # fewer digits than float64 offers, and the staged month is half the size.
    distances = np.full((len(lat), len(lon)), np.nan, dtype="float32")
    distances[sea_mask] = min_distance

    return (
        xr.DataArray(
            distances,
            coords={"lat": lat, "lon": lon},
            dims=["lat", "lon"],
            name=name,
        )
        .assign_coords({"time": date})
        .expand_dims(dim="time")
    )


class FrontProcessor:
    """
    One ``boa_fronts`` layer: detection and distances over a dataset's days.

    Args:
        var_key: Variable key whose store the layer belongs to. Names the
            staging stores and the log lines.
        name: Name the layer is written under (e.g. ``sst_fdist``).
        spec: Source field, threshold and worker count.
    """

    def __init__(self, var_key: str, name: str, spec: BOAFrontSpec):
        self.var_key = var_key
        self.name = name
        self.spec = spec

    @property
    def n_workers(self) -> int:
        return self.spec.n_workers or DEFAULT_N_WORKERS

    def from_dataset(self, ds: xr.Dataset) -> xr.DataArray:
        """
        Distances to the nearest front for every day on *ds*'s time axis.

        The days come from the axis itself rather than from a calendar rebuilt
        between its ends: a source that skipped a day would otherwise be asked
        for a date it does not carry, and fail deep inside a pool worker.

        Returns a lazy DataArray backed by the layer's staging store, which is
        the caller's to remove with :func:`clear_staging` once it has been
        written.
        """
        src = ds[self.spec.source]
        times = pd.DatetimeIndex(np.atleast_1d(src["time"].values))
        if times.empty:
            raise ValueError(
                f"[{self.var_key}] boa_fronts.{self.name}: '{self.spec.source}' "
                f"has no time steps to detect fronts on"
            )
        if not times.is_monotonic_increasing:
            # The layer is staged in axis order and its coordinates are taken
            # from the source at the end, so an out-of-order axis would pair
            # each day's distances with the wrong date.
            raise ValueError(
                f"[{self.var_key}] boa_fronts.{self.name}: '{self.spec.source}' "
                f"has an unsorted time axis; sort it before detecting fronts."
            )

        lat = src["lat"].values
        lon = src["lon"].values
        # The base grid (sea-point coordinates + land mask) depends only on
        # lat/lon, which are identical for every day, so build it once here
        # instead of recomputing it per day inside each worker.
        latlon1_arr, sea_mask = create_base_grid(lat, lon)

        source_stage = stage_path(self.var_key, f"{self.name}_source")
        layer_stage = stage_path(self.var_key, self.name)
        for path in (source_stage, layer_stage):
            shutil.rmtree(path, ignore_errors=True)
        source_stage.parent.mkdir(parents=True, exist_ok=True)

        # Workers read the field from a Zarr rather than from the dataset
        # open_mfdataset handed us: that one is a dask graph over the raw
        # files, which would be pickled to every worker and re-read per day.
        src.astype("float32").chunk({"time": 1, "lat": -1, "lon": -1}).to_dataset(
            name=self.spec.source
        ).to_zarr(source_stage, consolidated=False)
        staged_src = xr.open_zarr(source_stage, consolidated=False)

        logger.info(
            f"[{self.var_key}] BOA front detection for {self.name}: {len(times)} "
            f"day(s), threshold {self.spec.threshold}, {self.n_workers} workers"
        )
        worker = partial(
            _detect_day,
            da=staged_src[self.spec.source],
            threshold=self.spec.threshold,
            name=self.name,
            latlon1_arr=latlon1_arr,
            lat=lat,
            lon=lon,
            sea_mask=sea_mask,
        )
        t0 = time.perf_counter()
        try:
            with mp.Pool(processes=self.n_workers) as pool:
                appending = False
                for batch in _month_batches(times):
                    self._stage_batch(pool, worker, batch, layer_stage, appending)
                    appending = True
        finally:
            staged_src.close()
            shutil.rmtree(source_stage, ignore_errors=True)

        logger.success(
            f"[{self.var_key}] {self.name}: {len(times)} day(s) in "
            f"{time.perf_counter() - t0:.1f}s"
        )
        return self._open_staged(layer_stage, src)

    def _stage_batch(
        self,
        pool: Pool,
        worker,
        batch: pd.DatetimeIndex,
        layer_stage: Path,
        appending: bool,
    ) -> None:
        """Detect one month's days and append them to the staging store."""
        t0 = time.perf_counter()
        # Explicit join: every day is built on the same lat/lon arrays, so an
        # outer join would only mask a bug.
        days: list[xr.DataArray] = pool.map(worker, batch)
        block = xr.concat(days, dim="time", join="exact").to_dataset(name=self.name)
        if appending:
            block.to_zarr(layer_stage, append_dim="time", consolidated=False)
        else:
            block.to_zarr(
                layer_stage,
                consolidated=False,
                encoding={self.name: {"chunks": _stage_chunks(block[self.name])}},
            )
        logger.debug(
            f"[{self.var_key}] {self.name} {batch[0]:%Y-%m}: {len(batch)} day(s) "
            f"detected and staged in {time.perf_counter() - t0:.1f}s"
        )

    def _open_staged(self, layer_stage: Path, src: xr.DataArray) -> xr.DataArray:
        """Lazy view of the staged layer, on the source's own coordinates."""
        layer = xr.open_zarr(layer_stage, consolidated=False)[self.name]
        # The staging layout must not leak into the store: without this the
        # one-day chunks it was staged with follow the layer into to_zarr and
        # fight the layout chunk_dataset asks for.
        layer.encoding = {}
        # Coordinates come back from the source rather than from the round
        # trip through Zarr. The layer is merged into the dataset it was
        # detected in, and that merge aligns on coordinate *values* — a time
        # axis decoded back at another resolution would align to nothing.
        return layer.assign_coords(
            {"time": src["time"], "lat": src["lat"], "lon": src["lon"]}
        )


def apply_boa_fronts(
    ds: xr.Dataset, fronts: dict[str, BOAFrontSpec] | None, owner: str
) -> xr.Dataset:
    """
    Add each declared front-distance layer to *ds*, in declaration order.

    Unlike :func:`h2mare.processing.derived.apply_derived_vars`, which it sits
    beside in the convert step, the work happens here: each layer is detected
    day by day and staged to disk, and what lands on *ds* is a lazy view of
    that staging store. ``owner`` is the var_key named in errors and in the
    staging paths, and whose conversion clears them.
    """
    for name, spec in (fronts or {}).items():
        if spec.source not in ds.data_vars:
            raise ValueError(
                f"[{owner}] boa_fronts.{name} reads '{spec.source}', which the "
                f"dataset does not hold. Variables: "
                f"{sorted(map(str, ds.data_vars))}."
            )

        src = ds[spec.source]
        dims = set(map(str, src.dims))
        missing = {"time", "lat", "lon"} - dims
        if missing:
            raise ValueError(
                f"[{owner}] boa_fronts.{name} needs '{spec.source}' over "
                f"time/lat/lon; it has {tuple(map(str, src.dims))}."
            )
        extra = sorted(dims - {"time", "lat", "lon"})
        if extra:
            raise ValueError(
                f"[{owner}] boa_fronts.{name} reads '{spec.source}', which also "
                f"has {extra}. BOA detects fronts in one lat×lon field per day; "
                f"reduce it to a single level first and point the entry at that."
            )

        ds[name] = FrontProcessor(owner, name, spec).from_dataset(ds)
    return ds
