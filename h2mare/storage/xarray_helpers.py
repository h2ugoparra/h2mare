from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from loguru import logger


def xr_float64_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Convert float64 variables to float32 in a Dataset."""
    return ds.map(lambda da: da.astype(np.float32) if da.dtype == np.float64 else da)


# Backward-compatible alias kept so existing call sites continue to work
ds_float64_to_float32 = xr_float64_to_float32


#: How much wider than the observed range to make the int16 scale.
#:
#: A store's scale is fixed when it is *created* and every later append
#: inherits it, so the range this batch happens to span becomes the range the
#: store can represent for its whole life. At 1.0 the observed data fills
#: ±32500 of int16's ±32767 — about 0.4% headroom, which a single unusually
#: cold day or deep low is enough to exceed, and an exceedance wraps rather
#: than clips.
#:
#: 1.6 spreads the same span over 1.6× the value range, leaving roughly 60%
#: headroom beyond the observed half-range on each side at 1.6× coarser
#: resolution — msl ~0.24 Pa rather than ~0.15 Pa, still far inside ERA5's own
#: precision. Appends outside even this are refused by
#: ``storage._check_packed_range`` rather than silently wrapped.
_INT16_HEADROOM = 1.6


def int16_scale(lo: float, hi: float) -> dict | None:
    """
    ``scale_factor``/``add_offset`` packing [lo, hi] into int16, widened by
    :data:`_INT16_HEADROOM`.

    None for a degenerate or all-NaN range: the scale would be zero or
    non-finite, so there is nothing to pack.
    """
    span = hi - lo
    if not np.isfinite(span) or span == 0:
        return None
    return {
        "scale_factor": span * _INT16_HEADROOM / 65000.0,
        "add_offset": (hi + lo) / 2.0,
    }


def int16_encoding(ds: xr.Dataset, level: int = 9) -> dict:
    """
    Scale/offset int16 encoding, one scale per data variable.

    The scale spans each variable's own range, widened by
    :data:`_INT16_HEADROOM`, over 65000 levels — which for ERA5 lands well
    inside the source's own precision (msl ~0.24 Pa, wind ~0.002 m/s). NaN —
    land, or a gap — round-trips through ``_FillValue``.

    The headroom is not slack. This encoding is applied only when a store is
    *created*; appends inherit it, so a scale derived from one batch has to
    hold for every later one. Without it, data outside the first batch's range
    overflows int16 and wraps back into the middle of the range — worse than
    clipping, because a wrapped value looks like ordinary data.

    Computing the range forces one pass over the data. That is deliberate: a
    fixed guess would clip, and clipping is silent.

    One pass, though, not one per reduction: every min and max goes into a
    single graph so each source chunk is decoded once and fed to both. Computed
    separately they cost a full re-read apiece — on a year of hourly ERA5 that
    is tens of GB of GRIB decoded again for every extra call, all of it before
    the write starts and so invisible in the log.
    """
    import dask
    import zarr

    logger.info(
        f"int16_encoding: scanning {len(ds.data_vars)} variable(s) for their "
        f"value range (one pass over the source, no output until it completes)"
    )
    bounds: dict[tuple[str, str], xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        bounds[(str(name), "lo")] = da.min()
        bounds[(str(name), "hi")] = da.max()
    (computed,) = dask.compute(bounds)

    encoding: dict = {}
    for name in ds.data_vars:
        lo = float(computed[(str(name), "lo")])
        hi = float(computed[(str(name), "hi")])
        scale = int16_scale(lo, hi)
        if scale is None:
            # Degenerate or all-NaN: packing buys nothing, so leave this
            # variable on the default encoding.
            continue
        encoding[name] = {
            "dtype": "int16",
            **scale,
            "_FillValue": -32767,
            "compressors": [zarr.codecs.ZstdCodec(level=level)],
        }
    return encoding


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


def _log_chunk_layout(ds: xr.Dataset, layout: str, time_dim: str = "time") -> None:
    """Log the chunk layout, shape and uncompressed size of a chunked dataset.

    Reports a representative time-bearing variable (the one with the most
    timesteps); every gridded variable shares the same per-chunk shape, so this
    one line tells you how big a single read is and which layout produced it.
    """
    time_vars = [v for v in ds.data_vars if time_dim in ds[v].dims]
    if not time_vars:
        return
    rep = max(time_vars, key=lambda v: ds[v].sizes[time_dim])
    da = ds[rep]
    chunk_shape = {str(d): int(da.chunksizes[d][0]) for d in da.dims}
    n_cells = 1
    for c in chunk_shape.values():
        n_cells *= c
    mb = n_cells * da.dtype.itemsize / 1024**2
    logger.info(
        f"chunk_dataset: layout='{layout}' chunk {chunk_shape} "
        f"~{mb:.2f} MB ({da.dtype}, e.g. '{rep}')"
    )


def chunk_dataset(
    ds: xr.Dataset,
    target_mb: int = 32,
    time_dim: str = "time",
    spatial_chunk: int = 256,
    layout: Literal["timeseries", "map"] = "timeseries",
    map_time_chunk: int = 14,
) -> xr.Dataset:
    """
    Convert all variables from float64 to float32 and chunk for storage,
    keeping each chunk close to target_mb.

    Two layouts trade off opposite access patterns; pick by how the store is read:

    ``"timeseries"`` (default) tiles spatial dims (lat/lon/x/y) to ``spatial_chunk``
    cells (capped at the dim size) and fills the remaining byte budget with time.
    Tiling is what makes point/geometry extraction cheap: a small bbox reads only
    the overlapping tiles instead of decompressing the full grid for every
    timestep. Non-spatial, non-time dims (e.g. depth) are chunked to 1 when a
    full-grid per-step payload exceeds target_mb, preventing oversized chunks on
    4-D datasets. Trade-off: tiling speeds up subset reads but makes full-grid
    single-timestep reads (e.g. a global daily map) costlier, since the larger
    time chunk pulls neighbouring timesteps per tile.

    ``"map"`` keeps spatial dims contiguous and pins the time chunk to
    ``map_time_chunk`` (default 14), so a small block of full-grid fields is the
    fewest possible chunks while a single-day field still reads just one chunk.
    This is the layout an interactive map / animation wants. Trade-off: long
    time-series reads are now costly (few timesteps per chunk). Spatial dims are
    only tiled down here if a *single* full-grid timestep already exceeds
    target_mb (hi-res grids).

    Note: appends rewrite a period file at its *existing* chunking
    (``write_append_zarr`` reads ``ds_old.chunksizes``), so changing this only
    affects newly created files — existing stores keep their layout until
    re-chunked explicitly.

    Args:
        ds: dataset to chunk.
        target_mb : Target uncompressed chunk size in MB.
        time_dim : Time dimension name.
        spatial_chunk : Max cells per chunk along each spatial dim (lat/lon/x/y);
            ``"timeseries"`` layout only.
        layout : ``"timeseries"`` (time-contiguous, extraction) or ``"map"``
            (space-contiguous, interactive display).
        map_time_chunk : Time chunk for the ``"map"`` layout. A single-day field
            still reads one chunk regardless; larger values pack more days per
            chunk, cutting the chunk-file count (at the cost of pulling
            neighbouring days per read). 1 is the smallest/most-random-friendly.
            Ignored for ``"timeseries"``.
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

    if layout == "map":
        # Asymmetry vs "timeseries" is deliberate, do NOT "fix" it to mirror the
        # budget-fill-with-time behaviour. Both layouts fill the byte budget along
        # the axis read *contiguously* and minimise the axis indexed *into*:
        # timeseries reads all time for a small tile (fill time), map reads one
        # timestep across the full grid (fill space, minimise time). The 32 MB
        # budget therefore caps the *spatial* extent here (see the oversized-step
        # guard below), while time is pinned to the small ``map_time_chunk``.
        # Filling time to the budget instead would force a single-day viz read to
        # decompress dozens of unwanted days per frame (read amplification),
        # degrading the interactive scrubbing this layout exists for.
        #
        # Keep spatial dims contiguous so a single full-grid field is one chunk;
        # collapse any other non-time dim (e.g. depth) to 1.
        map_dims: dict[str, int] = {}
        for dim, size in ds.sizes.items():
            name = str(dim)
            if name == time_dim:
                continue
            map_dims[name] = int(size) if name.lower() in spatial_dims else 1

        spatial = [d for d in map_dims if d.lower() in spatial_dims]
        if not spatial:
            logger.warning(
                "chunk_dataset(layout='map'): no spatial dims (lat/lon/x/y) found "
                f"in {list(ds.sizes)}; only the time chunk is being set. The map "
                "layout is meant for gridded data — check the input dataset."
            )
        # Guard a single oversized timestep (hi-res grids): if one full field
        # exceeds the budget, shrink spatial tiles uniformly so the chunk fits.
        step_bytes = int(np.prod(list(map_dims.values()) or [1])) * da.dtype.itemsize
        if step_bytes > target_bytes and spatial:
            scale = (target_bytes / step_bytes) ** (1 / len(spatial))
            for d in spatial:
                map_dims[d] = max(1, int(ds.sizes[d] * scale))
        tchunk = max(1, min(map_time_chunk, ds.sizes[time_dim]))
        out = ds.chunk({time_dim: tchunk} | map_dims)
        _log_chunk_layout(out, "map", time_dim)
        return out

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

    out = ds.chunk({time_dim: time_chunk} | dim_dict)
    _log_chunk_layout(out, "timeseries", time_dim)
    return out


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


# Decimal places lon/lat labels are rounded to. The finest grid in the pipeline
# is CMEMS' 1/12° (~0.0833°) product; its cells sit ~5500× farther apart than
# the ~1.5e-5° float noise a source introduces when it reprocesses/re-grids a
# product. Rounding to 4 dp (~11 m at the equator) erases that noise without ever
# merging two real cells.
GRID_COORD_DECIMALS = 4


def snap_grid_coords(ds: xr.Dataset, decimals: int = GRID_COORD_DECIMALS) -> xr.Dataset:
    """Round lon/lat coordinate labels to a canonical precision.

    Source providers occasionally reprocess a product and shift its grid by
    floating-point noise (CMEMS seapodym moved its longitudes by ~1.5e-5° from
    2023 on). When period files carrying such near-but-unequal labels are later
    combined — ``xr.open_mfdataset(join="outer")`` on read, or ``xr.concat`` on
    append — the mismatched labels get *unioned* instead of aligned, doubling the
    axis and NaN-filling each block at the other grid's phantom cells. Snapping
    every file's labels to the same rounded grid keeps them bit-identical so they
    align rather than union.
    """
    new_coords = {}
    for name in ("lon", "lat"):
        if name not in ds.coords:
            continue
        rounded = np.round(ds[name].values, decimals)
        if np.unique(rounded).size != rounded.size:
            # Rounding would collapse genuinely distinct cells — leave it alone
            # rather than silently corrupt a finer grid than we assumed.
            logger.warning(
                f"snap_grid_coords: rounding '{name}' to {decimals} dp would merge "
                f"distinct cells ({rounded.size} -> {np.unique(rounded).size}); "
                "leaving it unchanged."
            )
            continue
        new_coords[name] = rounded
    return ds.assign_coords(new_coords) if new_coords else ds


#: Attribute name prefixes carrying the source file's own encoding.
_SOURCE_ENCODING_PREFIXES = ("GRIB_",)

#: Attribute names carrying the source file's packing bounds. These read as
#: physical limits and are not: CMEMS ships int16-packed data, so thetao_100
#: arrives declaring [-32766, 21306] "degrees_C" against real values of
#: [-2.3, 28.2]. Some variables' bounds happen to be plausible (mld's [1, 4525]
#: metres), which is what makes keeping them worse than dropping them — a
#: consumer cannot tell the nonsense from the sensible.
_SOURCE_ENCODING_ATTRS = ("valid_min", "valid_max")


def drop_source_encoding_attrs(ds: xr.Dataset, *, drop_grib: bool = True) -> xr.Dataset:
    """
    Strip attributes that describe the source file rather than this dataset.

    They were true of the file the data came from and stopped being true the
    moment it was regridded and re-encoded, but nothing removes them, so they
    travel into the compiled product asserting the wrong thing. On h2ds — 280 x
    360 at 0.25 deg — the wave variables still claimed ``GRIB_Nx=181,
    GRIB_Ny=141, iDirectionIncrement=0.5``, ERA5's coarser wave grid; the upwell
    counts claimed ``GRIB_units='N m**-2'`` inherited from the stress field they
    are derived from; and ``GRIB_missingValue`` named a sentinel h2ds does not
    use, having NaN.

    Only data variables are touched, and only these families: ``standard_name``,
    ``cell_methods``, ``unit_long`` and the rest still describe the quantity
    itself and survive. Provenance is not lost with ``GRIB_paramId`` — the
    variable's ``product_id``/``dataset_id`` carry it.

    Args:
        ds: Dataset to strip, modified in place and returned.
        drop_grib: Whether to drop the ``GRIB_*`` family as well as the packing
            bounds. True for the compiled product, whose regrid is what makes
            those attributes lie. False on the native path: a CDS store is
            written at ERA5's own grid and cadence, so ``GRIB_Nx``/``GRIB_Ny``
            still describe it, and :func:`~h2mare.processing.core.cds.hourly_radiation`
            deliberately *rewrites* ``GRIB_units`` and ``GRIB_stepType`` there to
            stop the accumulation being differenced twice. Dropping them would
            throw that correction away along with the ``GRIB_paramId``
            provenance. The packing bounds go either way — they are wrong on a
            native store too, where the values have been unpacked to float.
    """
    prefixes = _SOURCE_ENCODING_PREFIXES if drop_grib else ()
    for var in ds.data_vars:
        attrs = ds[var].attrs
        for name in list(attrs):
            if (
                prefixes and name.startswith(prefixes)
            ) or name in _SOURCE_ENCODING_ATTRS:
                del attrs[name]
    return ds


#: CF attributes for the axes every store in the pipeline shares.
#:
#: ``time`` deliberately carries no ``units``: xarray owns the time encoding and
#: writes ``units``/``calendar`` into ``.encoding`` at ``to_zarr``. Setting them
#: in ``.attrs`` as well makes the write raise rather than merely disagree.
_CF_COORD_ATTRS: dict[str, dict[str, str]] = {
    "lon": {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    },
    "lat": {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "axis": "Y",
    },
    "depth": {
        "standard_name": "depth",
        "long_name": "depth",
        "units": "m",
        "positive": "down",
        "axis": "Z",
    },
    "time": {"standard_name": "time", "long_name": "time", "axis": "T"},
}


def resolve_cf_attrs(var_name: str, native_var_key: str | None = None) -> dict:
    """
    Config's attributes for one variable, with the native overrides layered on.

    Split out of :func:`apply_cf_attrs` so the repair script can compute what a
    stored variable *should* say without opening the data — and, more to the
    point, so it cannot compute it differently. A ``None`` value survives into
    the result and means remove, which is the caller's to act on.
    """
    from h2mare.config import get_settings

    settings = get_settings()
    overrides: dict = {}
    if native_var_key is not None:
        overrides = settings.native_attr_overrides.get(native_var_key, {})
    return {**settings.get_var_info(var_name), **overrides.get(var_name, {})}


def apply_cf_attrs(ds: xr.Dataset, native_var_key: str | None = None) -> xr.Dataset:
    """
    Put config's metadata onto a dataset's variables and axes.

    The single place both write paths get their attributes from, so a native
    store and h2ds cannot drift into describing the same quantity differently.

    Coordinates get :data:`_CF_COORD_ATTRS`. Without them a store is not merely
    under-documented: ``rio.clip`` resolves spatial dims by name and only falls
    back to lon/lat when they carry CF attributes, which is why geometry
    extraction against the CDS stores and h2ds used to clip to nothing but NaN.

    Args:
        ds: Dataset to annotate, modified in place and returned.
        native_var_key: The var_key whose *native* store is being written, or
            None for the compiled product. Two things follow from it: the
            ``GRIB_*`` attributes are kept (see
            :func:`drop_source_encoding_attrs`), and that key's entry in
            ``native_attr_overrides`` is layered over the shared table. The
            overrides exist because a native store is not always in the units it
            publishes downstream — ``msl`` is Pa here and hPa in h2ds, ``tp`` is
            m here and mm there — and because an hourly field is not the daily
            reduction its ``cell_methods`` describes. An override of ``null``
            removes the attribute rather than setting it, which is how those
            ``cell_methods`` come off.
    """
    ds = drop_source_encoding_attrs(ds, drop_grib=native_var_key is None)

    for var in ds.data_vars:
        merged = resolve_cf_attrs(str(var), native_var_key)
        attrs = ds[var].attrs
        for key, value in merged.items():
            if value is None:
                attrs.pop(key, None)
            else:
                attrs[key] = value

    for name, coord_attrs in _CF_COORD_ATTRS.items():
        if name in ds.coords:
            ds[name].attrs.update(coord_attrs)

    return ds
