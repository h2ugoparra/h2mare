"""
Functions for geographic distance calculations, Grid creation and land cover cliping
"""

from typing import Optional

import numpy as np
import xarray as xr
from global_land_mask import globe
from numpy.typing import NDArray
from scipy.spatial import KDTree

from h2mare.types import BBox

_EARTH_RADIUS_KM: float = 6371.0


def haversine_min_distance_kdtree(
    coords1: NDArray[np.float64],
    coords2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    For each point in ``coords1``, compute the great-circle distance (km) to
    the nearest point in ``coords2`` using a KD-tree nearest-neighbour search.

    Points are projected to radians before indexing. The KD-tree uses Euclidean
    distance on radian coordinates, which approximates the haversine distance
    closely enough for nearest-neighbour ranking over typical oceanographic
    spatial scales.

    Args:
        coords1: Query points as (lat, lon) pairs in decimal degrees.
            Shape: ``(N, 2)``
        coords2: Target points as (lat, lon) pairs in decimal degrees.
            Shape: ``(M, 2)``

    Returns:
        Minimum great-circle distance in km from each point in ``coords1``
        to its nearest neighbour in ``coords2``. Shape: ``(N,)``

    Raises:
        ValueError: If either input array is not 2-dimensional or does not
            have exactly 2 columns.

    Example:
        >>> grid_points = np.array([[40.0, -10.0], [41.0, -9.5]])
        >>> eddy_centers = np.array([[40.1, -10.1], [45.0, 0.0]])
        >>> haversine_min_distance_kdtree(grid_points, eddy_centers)
        array([13.48, 14.2])
    """
    if coords1.ndim != 2 or coords1.shape[1] != 2:
        raise ValueError(f"coords1 must have shape (N, 2), got {coords1.shape}")
    if coords2.ndim != 2 or coords2.shape[1] != 2:
        raise ValueError(f"coords2 must have shape (M, 2), got {coords2.shape}")

    tree = KDTree(np.radians(coords2))
    distances, _ = tree.query(np.radians(coords1), k=1)
    return distances * _EARTH_RADIUS_KM


class GridBuilder:
    def __init__(
        self, bbox: BBox, dx: float, dy: float, attributes: Optional[dict | None] = None
    ):
        """
        Creates grid with given geoextent and grid cell size (dx, dy). Grid cells are centered.

        Args:
            xmin, ymin, xmax, ymax (float): lon min, lat min, lon max and lat max for geo extent
            dx, dy (float): grid cell size for lon and lat, (dx and dy respectively).
            attributes (dict): global attributes for dataset
        """
        self.xmin = bbox.xmin
        self.ymin = bbox.ymin
        self.xmax = bbox.xmax
        self.ymax = bbox.ymax
        self.dx = dx
        self.dy = dy
        self.attributes = attributes

    def generate_grid(self) -> xr.Dataset:
        # Generate the latitude and longitude arrays
        lat = np.arange(self.ymin + (self.dy / 2), self.ymax + (self.dy / 2), self.dy)
        lon = np.arange(self.xmin + (self.dx / 2), self.xmax + (self.dx / 2), self.dx)

        return xr.Dataset(coords={"lat": lat, "lon": lon})

    def generate_grid_with_attributes(self) -> xr.Dataset:
        grid = self.generate_grid()
        # Add attributes to the grid
        if self.attributes is not None:
            grid.attrs.update(self.attributes)
        return grid


def sel_padded_bbox(
    ds: xr.Dataset | xr.DataArray,
    bounds: tuple[float, float, float, float],
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> xr.Dataset | xr.DataArray:
    """
    Select a bounding box padded by one grid cell on each side.

    A sub-cell bbox (e.g. a short geometry on a coarse 0.5° grid) can fall
    between cell centers and yield an empty slice; padding by one cell keeps
    the surrounding cells in the selection.

    Args:
        ds: Dataset with monotonically increasing lat/lon coordinates.
        bounds: (xmin, ymin, xmax, ymax) in coordinate units.
        lat_coord: Latitude coordinate name. Defaults to "lat".
        lon_coord: Longitude coordinate name. Defaults to "lon".
    """
    xmin, ymin, xmax, ymax = bounds
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
    return ds.sel(
        {
            lat_coord: slice(ymin - lat_res, ymax + lat_res),
            lon_coord: slice(xmin - lon_res, xmax + lon_res),
        }
    )


def clip_land_data(ds: xr.Dataset) -> xr.Dataset:
    """Clip land values from a dataset

    Args:
        ds (xr.Dataset): dataset to clip

    Returns:
        xr.Dataset: dataset with land values as np.nan
    """
    lat1 = ds.coords["lat"].values
    lon1 = ds.coords["lon"].values
    lon1_grid, lat1_grid = np.meshgrid(lon1, lat1)
    sea_mask = ~globe.is_land(lat1_grid, lon1_grid)
    mask = xr.DataArray(
        sea_mask, dims=("lat", "lon"), coords={"lat": lat1, "lon": lon1}
    )
    return ds.where(mask)
