"""
Functions for geographic distance calculations, Grid creation and land cover cliping
"""

from typing import Mapping, Optional

import numpy as np
import xarray as xr
from global_land_mask import globe
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial import KDTree

from h2mare.types import BBox, RegridMethod

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


#: Tolerance on the target/native step ratio before a variable counts as being
#: coarsened. Stored axes are rounded to ``GRID_COORD_DECIMALS`` on write, so a
#: 0.25° store measures as 0.2500xx rather than 0.25 and must not read as
#: "slightly coarser than the target" and flip to the aggregating path.
_RATIO_TOL = 0.01

#: How far a stored label may sit from the regular axis fitted through the ends
#: before the axis is refused as irregular. Rounding to 4 decimals moves a label
#: by at most 5e-5, so this is twice the largest legitimate offset.
_AXIS_FIT_TOL = 1e-4


def axis_step(values: NDArray[np.float64] | xr.DataArray) -> float:
    """
    Mean spacing of a regular coordinate axis.

    Measured end to end rather than from ``np.diff``, because stored axes carry
    labels rounded to ``GRID_COORD_DECIMALS``: consecutive differences on a
    1/12° store alternate between 0.0833 and 0.0834, so their median reads
    0.083302 and their spread is 1e-4. The two endpoints carry the same rounding
    but divide it by the number of cells, which puts the result back within a
    float of the true step.

    Args:
        values: Coordinate labels, strictly increasing.

    Raises:
        ValueError: If the axis has fewer than two points, is not strictly
            increasing, or is not regularly spaced.
    """
    v = np.asarray(values, dtype="float64")
    if v.size < 2:
        raise ValueError(f"need at least 2 points to measure a step; got {v.size}")
    if not np.all(np.diff(v) > 0):
        raise ValueError("axis must be strictly increasing; sort it first")

    step = float((v[-1] - v[0]) / (v.size - 1))
    drift = float(np.abs(v[0] + np.arange(v.size) * step - v).max())
    if drift > _AXIS_FIT_TOL:
        raise ValueError(
            f"axis is not regularly spaced: labels sit up to {drift:.2e}° from a "
            f"regular {step:.6g}° grid. Regridding assumes a rectilinear axis."
        )
    return step


def _cell_edges(values: NDArray[np.float64], spherical: bool) -> NDArray[np.float64]:
    """
    Cell boundaries of a regular axis, built from the fitted step.

    From the fit rather than from the labels themselves, so rounded labels
    cannot leak their jitter into the weights. ``spherical`` converts latitude
    edges to sine, which turns a length overlap into the true area fraction of
    the zone between two parallels.
    """
    step = axis_step(values)
    edges = float(values[0]) - step / 2 + np.arange(len(values) + 1) * step
    if spherical:
        edges = np.sin(np.deg2rad(np.clip(edges, -90.0, 90.0)))
    return np.asarray(edges, dtype="float64")


def _overlap_weights(
    source: NDArray[np.float64], target: NDArray[np.float64], spherical: bool = False
) -> NDArray[np.float64]:
    """
    Per-axis overlap matrix ``(n_source, n_target)``.

    ``W[i, j]`` is how much of source cell *i* falls inside target cell *j* —
    length for longitude, area fraction for latitude. Zero where they do not
    meet. This is the whole of conservative remapping for a rectilinear grid:
    the 2-D weight is the product of the two axes' matrices, so they are applied
    one axis at a time and never formed as a 4-D tensor.
    """
    src_edges = _cell_edges(source, spherical)
    tgt_edges = _cell_edges(target, spherical)
    lower = np.maximum(src_edges[:-1, None], tgt_edges[None, :-1])
    upper = np.minimum(src_edges[1:, None], tgt_edges[None, 1:])
    return np.clip(upper - lower, 0.0, None)


def _weight_array(
    source: xr.DataArray, target: xr.DataArray, dim: str, spherical: bool
) -> xr.DataArray:
    """Overlap weights as a DataArray indexed ``(dim, <dim>_out)`` for ``xr.dot``."""
    weights = _overlap_weights(
        np.asarray(source.values, dtype="float64"),
        np.asarray(target.values, dtype="float64"),
        spherical=spherical,
    )
    return xr.DataArray(
        weights,
        dims=(dim, f"{dim}_out"),
        coords={dim: source.values},
    )


def _conservative_regrid(
    ds: xr.Dataset, target: xr.Dataset, min_coverage: float
) -> xr.Dataset:
    """
    Area-weighted mean of each variable over the target cells.

    Each output cell is the mean of the source cells overlapping it, weighted by
    how much of the cell each one covers, skipping the ones that are NaN. The
    normalisation is the weight of the *valid* source area rather than the whole
    cell, so a cell that is half land still reports the mean of its water —
    which is why this keeps coastal cells that ``interp`` drops.
    """
    w_lat = _weight_array(ds["lat"], target["lat"], "lat", spherical=True)
    w_lon = _weight_array(ds["lon"], target["lon"], "lon", spherical=False)
    # Weight available per target cell if every source cell contributing to it
    # were valid — the denominator ``coverage`` is measured against.
    full = w_lat.sum("lat") * w_lon.sum("lon")

    renames = {"lat_out": "lat", "lon_out": "lon"}
    out: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        if "lat" not in da.dims or "lon" not in da.dims:
            out[str(name)] = da
            continue

        valid = da.notnull()
        # fillna(0) rather than masking: a NaN cell contributes nothing to the
        # sum and nothing to the weight, which is the same as not being there.
        # It is also what avoids interp's 0 x NaN = NaN, which drops a target
        # cell over a zero-weight land neighbour.
        numerator = xr.dot(xr.dot(da.fillna(0.0), w_lat, dim="lat"), w_lon, dim="lon")
        denominator = xr.dot(
            xr.dot(valid.astype("float64"), w_lat, dim="lat"), w_lon, dim="lon"
        )
        result = numerator / denominator.where(denominator > 0)
        if min_coverage > 0:
            result = result.where(denominator >= min_coverage * full)

        result = result.rename(renames).transpose(*da.dims)
        result.attrs = dict(da.attrs)
        out[str(name)] = result.astype(da.dtype)

    return xr.Dataset(out, attrs=ds.attrs)


def regrid_to(
    ds: xr.Dataset,
    target: xr.Dataset,
    *,
    method: RegridMethod = "auto",
    methods: Optional[Mapping[str, RegridMethod]] = None,
    min_coverage: float = 0.0,
) -> xr.Dataset:
    """
    Put *ds* on the *target* grid, choosing how by comparing the two resolutions.

    ``interp`` samples the field at the target cell centres, which is right when
    the target is the same resolution or finer and wrong when it is coarser: it
    reads the 2x2 source cells around each centre and ignores the rest of the
    cell. Going from 0.05° to 0.25° that is 4 of 25 source cells, or exactly 1
    where the centres coincide. So the direction decides the method:

    * target step <= native step — ``linear``, as before.
    * target step > native step — ``conservative``, the area-weighted mean of
      every source cell in the target footprint.

    ``nearest`` is never chosen automatically. It is for fields a mean would
    destroy — an identifier, a class — and has to be asked for, per variable,
    through *methods*: an eddy track ID averaged with its neighbour's names a
    third eddy that does not exist.

    Args:
        ds: Dataset on a rectilinear lat/lon grid, both axes increasing.
        target: Dataset whose ``lat``/``lon`` define the output grid.
        method: Override the choice described above, for every variable.
        methods: Per-variable overrides, keyed by the variable's name in *ds*.
            Variables not named here follow *method*.
        min_coverage: Fraction of a target cell that must be valid (not NaN) for
            it to carry a value, for the conservative path only. ``0.0`` (the
            default) gives a value to any cell with some valid area.

    Returns:
        *ds* on the target grid, carrying the target's own coordinate objects so
        that separately regridded variables merge rather than union.
    """
    if methods:
        groups: dict[RegridMethod, list[str]] = {}
        for name in ds.data_vars:
            groups.setdefault(methods.get(str(name), method), []).append(str(name))
        if len(groups) > 1:
            # join="exact": each part already carries the target's coordinates,
            # so anything but an exact match means one of them was not regridded.
            merged = xr.merge(
                [
                    regrid_to(ds[names], target, method=m, min_coverage=min_coverage)
                    for m, names in groups.items()
                ],
                join="exact",
            )
            merged.attrs = dict(ds.attrs)
            return merged
        method = next(iter(groups), method)

    if method == "auto":
        ratio = max(
            axis_step(target["lat"].values) / axis_step(ds["lat"].values),
            axis_step(target["lon"].values) / axis_step(ds["lon"].values),
        )
        method = "conservative" if ratio > 1 + _RATIO_TOL else "linear"
        logger.debug(
            f"regrid: {axis_step(ds['lat'].values):.6g}° → "
            f"{axis_step(target['lat'].values):.6g}° (ratio {ratio:.3g}) via {method}"
        )

    if method == "linear":
        out = ds.interp_like(target, method="linear", assume_sorted=True)
    elif method == "nearest":
        out = ds.sel(lat=target["lat"], lon=target["lon"], method="nearest")
    elif method == "conservative":
        out = _conservative_regrid(ds, target, min_coverage)
    else:
        raise ValueError(f"unknown regrid method {method!r}")

    # Assigned rather than assumed: ``sel`` carries the source's own labels, and
    # the compiler merges the per-variable results with an outer join, where
    # axes differing in the last bit union into a doubled grid instead of
    # aligning.
    return out.assign_coords(lat=target["lat"].values, lon=target["lon"].values)


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
