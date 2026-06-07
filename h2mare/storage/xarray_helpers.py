from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger


def xr_float64_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Convert float64 variables to float32 in a Dataset."""
    return ds.map(lambda da: da.astype(np.float32) if da.dtype == np.float64 else da)


# Backward-compatible alias kept so existing call sites continue to work
ds_float64_to_float32 = xr_float64_to_float32


def get_dataset_encoding(ds: xr.Dataset) -> dict:
    """
    Get the chunking configuration for all variables in a Dataset. To be fed to function to_zarr encoding argument.

    Args:
        ds (xr.Dataset): dataset to encode

    Returns:
        dict: encoding configuration for dask array
    """
    ds = xr_float64_to_float32(ds)
    dim_dict = {dim: val for dim, val in ds.sizes.items() if dim != "time"}
    time_chunk = unified_time_chunk(ds)
    chunks = {"time": time_chunk, **dim_dict}

    return {
        var: {"chunks": tuple(chunks[dim] for dim in ds[var].sizes)}
        for var in ds.data_vars
    }


def chunk_dataset(
    ds: xr.Dataset,
    target_mb: int = 32,
    time_dim: str = "time",
    spatial_chunk: int = 256,
) -> xr.Dataset:
    """
    Convert all variables from float64 to float32 and chunk for storage,
    keeping each chunk close to target_mb.

    Spatial dims (lat/lon/x/y) are tiled to ``spatial_chunk`` cells (capped at the
    dim size). Tiling is what makes point/geometry extraction cheap: a small bbox
    reads only the overlapping tiles instead of decompressing the full grid for
    every timestep. The time chunk then fills the remaining budget up to ~target_mb.
    Non-spatial, non-time dims (e.g. depth) are chunked to 1 when a full-grid
    per-step payload exceeds target_mb, preventing oversized chunks on 4-D datasets.

    Trade-off: tiling speeds up subset reads but produces more, smaller chunk files
    and makes full-grid single-timestep reads (e.g. a global daily map) costlier,
    since the larger time chunk pulls neighbouring timesteps per tile.

    Note: appends rewrite a period file at its *existing* chunking
    (``write_append_zarr`` reads ``ds_old.chunksizes``), so changing this only
    affects newly created files — existing stores keep their layout until
    re-chunked explicitly.

    Args:
        ds: dataset to chunk.
        target_mb : Target uncompressed chunk size in MB.
        time_dim : Time dimension name.
        spatial_chunk : Max cells per chunk along each spatial dim (lat/lon/x/y).
    """
    ds = xr_float64_to_float32(ds)

    target_bytes = target_mb * 1024 * 1024
    spatial_dims = {"lat", "lon", "latitude", "longitude", "x", "y"}

    time_vars = [v for v in ds.data_vars if time_dim in ds[v].dims]
    if not time_vars or time_dim not in ds.sizes:
        raise ValueError(f"No variables contain dimension '{time_dim}'")

    main_var = max(
        time_vars, key=lambda v: ds[v].sizes[time_dim] * ds[v].dtype.itemsize
    )
    da = ds[main_var]
    time_idx = da.dims.index(time_dim)
    bytes_per_step = (
        int(np.prod([s for i, s in enumerate(da.shape) if i != time_idx]))
        * da.dtype.itemsize
    )

    dim_dict: dict[str, int] = {}
    for dim, size in ds.sizes.items():
        if dim == time_dim:
            continue
        if dim.lower() in spatial_dims:
            # Tile spatial dims so a small bbox reads only the overlapping tiles.
            dim_dict[dim] = min(spatial_chunk, int(size))
        elif bytes_per_step <= target_bytes:
            dim_dict[dim] = size
        else:
            dim_dict[dim] = 1

    non_time_size = int(np.prod(list(dim_dict.values()))) if dim_dict else 1
    time_chunk = max(
        1,
        min(
            int(target_bytes // (non_time_size * da.dtype.itemsize)), ds.sizes[time_dim]
        ),
    )

    return ds.chunk({time_dim: time_chunk} | dim_dict)


def unified_time_chunk(
    ds: xr.Dataset, target_mb: int = 32, time_dim: str = "time"
) -> int:
    """
    Suggest an integer chunk size along the time dimension
    so that the resulting chunk is close to target_mb.

    Args:
        ds: dataset to interpolate chunk size.
        target_mb : Target uncompressed chunk size in MB.
        time_dim : Time dimension name.

    Returns:
        int: Chunk size along the time dimension.
    """
    target_bytes = target_mb * 1024 * 1024

    time_vars = [v for v in ds.data_vars if time_dim in ds[v].dims]
    if not time_vars or time_dim not in ds.sizes:
        raise ValueError(f"No variables contain dimension '{time_dim}'")

    main_var = max(
        time_vars,
        key=lambda v: ds[v].sizes[time_dim] * ds[v].dtype.itemsize,
    )
    da = ds[main_var]

    time_idx = da.dims.index(time_dim)
    non_time_elems = int(np.prod([s for i, s in enumerate(da.shape) if i != time_idx]))
    bytes_per_step = non_time_elems * da.dtype.itemsize

    if bytes_per_step == 0:
        chunk_len = ds.sizes.get(time_dim, 1)
    else:
        chunk_len = int(target_bytes // bytes_per_step)

    chunk_len = max(1, min(chunk_len, da.sizes[time_dim]))
    return chunk_len


def have_vars_unique_values(ds: Path | xr.Dataset) -> bool:
    """
    Return ``True`` if **any** variable in the given Zarr path or dataset has only one
    unique value in its last time slice. Used to detect corrupt merge output.

    Parameters
    ----------
    ds : pathlib.Path or xarray.Dataset
        Either a path to a Zarr store or an already opened Dataset.
    """
    own_ds = False
    if isinstance(ds, Path):
        try:
            ds = xr.open_zarr(ds)
            own_ds = True
        except Exception as e:
            logger.error(f"Could not open Zarr store at {ds}: {e}")
            return False

    unique_found = False

    try:
        for var in ds.data_vars:
            if "time" not in ds[var].dims:
                continue
            last_slice = ds[var].isel(time=-1)
            uniq = np.unique(last_slice.values)
            if len(uniq) == 1:
                t = str(last_slice.time.values)[:10]
                logger.warning(
                    f"{var} has a single unique value ({uniq[0]:.4g}) at time={t} "
                    f"in {ds.encoding.get('source', 'unknown')}"
                )
                unique_found = True
    finally:
        if own_ds:
            ds.close()
    return unique_found


def convert360_180(_ds: xr.Dataset) -> xr.Dataset:
    """Convert 0-360 lon to -180-180 (FSLE)."""
    if _ds["lon"].min() >= 0:
        with xr.set_options(keep_attrs=True):
            _ds.coords["lon"] = (_ds["lon"] + 180) % 360 - 180
        _ds = _ds.sortby("lon")
    return _ds


def rename_dims(ds: xr.Dataset) -> xr.Dataset:
    """Rename 'longitude', 'latitude', and 'valid_time' (CDS-ERA5) to lon, lat, time."""
    mapping = {}
    if "longitude" in ds.sizes:
        mapping["longitude"] = "lon"
    if "latitude" in ds.sizes:
        mapping["latitude"] = "lat"
    if "valid_time" in ds.sizes:
        mapping["valid_time"] = "time"
    return ds.rename(mapping)
