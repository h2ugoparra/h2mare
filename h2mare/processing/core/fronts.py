"""
Detection of oceanic fronts using the Belkin O'Reilly algorithm, applied to sea surface temperature (SST) and chlorophyll (CHL) data.
This module includes functions for identifying frontal pixels and calculating distances to the nearest front for each grid cell in a given dataset.
The main class, `FrontProcessor`, orchestrates the processing of xarray Datasets, allowing for parallel computation across multiple time steps.
"""

from __future__ import annotations

import multiprocessing as mp
import shutil

import numpy as np
import pandas as pd
import xarray as xr
from global_land_mask import globe
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter, median_filter, minimum_filter, sobel

from h2mare import get_settings
from h2mare.utils.spatial import haversine_min_distance_kdtree

# Thresholds for different variables (based on literature)
threshold_dict = {"sst": 0.4, "chl": 0.06}

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
        threshold: Threshold value for gradient magnitude. Pixels with magnitude above
            this threshold are classified as part of a front.

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


class FrontProcessor:
    def __init__(self, var_key: str):

        self.var_key = var_key
        if self.var_key not in threshold_dict:
            raise ValueError(f"var_key must be one of {list(threshold_dict)}")

    def from_dataset(
        self, ds: xr.Dataset, n_workers: int = 10
    ) -> xr.DataArray | xr.Dataset:
        """
        Calculate frontal distances from an xarray.Dataset, typically loaded from a NetCDF file.

        This method processes the input dataset day by day in parallel to compute the distance
        to the nearest detected front for each grid cell.

        Args:
            var_key (str): The variable key to process, which must be either 'sst' (sea surface
                           temperature) or 'chl' (chlorophyll).
            ds (xr.Dataset): The input dataset containing the variable data over a time range.
                             It must have 'time', 'lat', and 'lon' coordinates.
            n_workers (int, optional): The number of parallel worker processes to use for
                                       daily processing. Defaults to 10.

        Returns:
            xr.DataArray | xr.Dataset: An xarray object containing the frontal distance data,
                                        combined across all processed dates. If no data is
                                        successfully processed, an empty DataArray is returned.
        """

        da = (
            ds[self.var_key]
            .astype("float32")
            .chunk({"time": 1, "lat": 500, "lon": 500})
        )

        zarr_path = get_settings().INTERIM_DIR / f"{self.var_key}_tmp.zarr"
        da.to_dataset(name=self.var_key).to_zarr(zarr_path, mode="w")

        _zarr_ds = xr.open_zarr(zarr_path)
        da = _zarr_ds[self.var_key]

        start_date = pd.Timestamp(ds.time.min().values)
        end_date = pd.Timestamp(ds.time.max().values)
        dates = pd.date_range(start_date, end_date, freq="D")

        tasks = [(da, date) for date in dates]
        out: list[xr.DataArray] = []

        logger.info(
            f"Front detection process for {self.var_key.upper()}: {len(dates)} days using {n_workers} workers"
        )
        try:
            with mp.Pool(processes=n_workers) as pool:
                results = pool.starmap(self._process_daily, tasks)
                out.extend([r for r in results if r is not None])
        finally:
            _zarr_ds.close()
            shutil.rmtree(zarr_path, ignore_errors=True)

        logger.success("Completed")

        if not out:
            return xr.DataArray(
                np.empty((0, 0)),
                coords={"lat": [], "lon": []},
                dims=["lat", "lon"],
                name=f"{self.var_key}_fdist",
            )

        return xr.combine_by_coords(out)

    def _process_daily(self, da: xr.DataArray, date: pd.Timestamp):

        da_tmp = da.sel(time=date)
        lat = da_tmp.lat.values
        lon = da_tmp.lon.values

        latlon1_arr, sea_mask = create_base_grid(lat, lon)
        latlon2_arr = BOA_application(da_tmp, threshold_dict[self.var_key])

        min_distance = haversine_min_distance_kdtree(latlon1_arr, latlon2_arr)
        reshaped_min_distance = np.full((len(lat), len(lon)), np.nan)
        reshaped_min_distance[sea_mask] = min_distance

        return (
            xr.DataArray(
                reshaped_min_distance,
                coords={"lat": lat, "lon": lon},
                dims=["lat", "lon"],
                name=f"{self.var_key}_fdist",
            )
            .assign_coords({"time": date})
            .expand_dims(dim="time")
        )


# def main():
#    mp.freeze_support()
#
#    var_key = 'sst'
#    FrontProcessor().run(
#        var_key=var_key,
#        start_date="1998-01-01",
#        end_date="1998-12-31",
#        n_workers= 10,
#        )
#
# if __name__ == "__main__":
#    main()
