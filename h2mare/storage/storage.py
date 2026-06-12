"""
Zarr Save/Temporal-overlap-check Logic
"""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.storage.xarray_helpers import snap_grid_coords
from h2mare.types import BBox, DateRange


def write_append_zarr(
    var_key: str,
    ds: xr.Dataset,
    path: Path,
) -> None:
    """
    Write dataset, checking temporal overlap and appending data if path exists.

    Args:
        var_key: Variable key, must exist in app_config.variables (used for overlap resolution)
        ds: New dataset to write/append
        path: Destination zarr path, built by the caller via ``ZarrCatalog.build_file_path()``
    """
    # Canonicalize grid labels before any write/append so float-noise drift —
    # between a source's reprocessed periods, or between new data and a legacy
    # store on a slightly different grid — can't union into a doubled axis. The
    # old side is snapped too in _append_data, so an append self-heals the store.
    # Idempotent: a no-op for data already on the rounded grid.
    ds = snap_grid_coords(ds)

    if path.exists():
        logger.debug(f"{path.name} exists — appending.")
        _append_data(var_key, ds, path)

    else:
        logger.info(f"Saving new dataset at {path}")
        t0 = time.perf_counter()
        ds.to_zarr(path)
        try:
            xr.open_zarr(path, consolidated=False).close()
        except Exception as e:
            shutil.rmtree(path, ignore_errors=True)
            raise RuntimeError(f"Zarr write verification failed for {path}") from e
        ds.close()
        logger.success(f"Saved in {time.perf_counter() - t0:.1f}s")


def _append_data(var_key: str, ds_new: xr.Dataset, path: Path) -> None:
    """
    Append new data to existing zarr file, handling two distinct cases:

    **Variable-addition** — all variables in *ds_new* are absent from the
    existing zarr (disjoint variable sets).  The new variables are merged into
    the existing dataset with an outer join so no existing variable is lost.
    This is the correct path when backfilling a brand-new variable (e.g.
    ``thetao``) into an already-compiled h2ds file that has many other variables.

    **Time-extension** — *ds_new* shares at least one variable with the existing
    zarr.  Temporal overlap is resolved via :func:`_resolve_overlap` and the
    non-overlapping head of the existing data is concatenated with *ds_new* along
    the time dimension.

    Both paths write via an atomic tmp → backup-swap to avoid corrupt state on
    failure.

    Args:
        var_key: The key for the variable to be processed and must exist in app_config.variables
        ds_new: New dataset to append.
        path: file path created by ``ZarrCatalog(var_key).build_file_path()``

    Raises:
        ValueError: If corrupted dataset is detected with unique values after concatenation.
    """
    ds_old = xr.open_zarr(path, consolidated=False)
    ds_old_vars = set(ds_old.data_vars)
    ds_new_vars = set(ds_new.data_vars)

    # src_to_close tracks every dataset that holds open handles into path's zarr
    # store.  They must all be closed (and gc'd) before the backup-swap so that
    # Windows releases its file locks on the directory.
    src_to_close: list[xr.Dataset] = []

    if ds_new_vars.isdisjoint(ds_old_vars):
        # Variable-addition: none of the incoming variables exist in the zarr yet.
        # Merge so that all existing variables are preserved alongside the new ones.
        logger.info(
            f"Variable-addition: merging {sorted(ds_new_vars)} into {path.name}."
        )
        # Snap the on-disk grid too: ds_new is already snapped (write_append_zarr),
        # so aligning ds_old here keeps the outer join from unioning a legacy
        # noise-drifted grid against the rounded one.
        ds_out = xr.merge([snap_grid_coords(ds_old), ds_new], join="outer")
        chunk_sizes = {dim: sizes[0] for dim, sizes in ds_old.chunksizes.items()}
        ds_out = ds_out.chunk(chunk_sizes)
        src_to_close.append(ds_old)
    else:
        # Time-extension: at least one shared variable — resolve temporal overlap
        # then concatenate along the time dimension.
        # ds_old is reopened inside _resolve_overlap; closing here avoids a
        # redundant open handle (zarr stores are lazy so this is safe).
        ds_old.close()

        # Clean trailing append (same vars, same grid, strictly after the
        # stored dates) extends the zarr in place — by year's end the rewrite
        # path would otherwise copy ~a full year of data per incremental run.
        if _try_append_fast_path(ds_new, path):
            return None

        ds_resolved = _resolve_overlap(ds_new, path)

        if ds_resolved is not None:
            # ds_new is already snapped (write_append_zarr); snap the retained
            # head of the existing store so concat aligns instead of unioning a
            # legacy noise-drifted grid against the rounded one.
            ds_out = xr.concat(
                [snap_grid_coords(ds_resolved), ds_new], dim="time", data_vars="minimal"
            )
            # Rechunk to match the existing zarr layout and avoid dask chunk-alignment errors.
            chunk_sizes = {
                dim: sizes[0] for dim, sizes in ds_resolved.chunksizes.items()
            }
            ds_out = ds_out.chunk(chunk_sizes)
            src_to_close.append(ds_resolved)
        else:
            ds_out = ds_new

    # Drop chunk encodings inherited from the on-disk store: when the retained
    # head is shorter than one zarr chunk, the stale encoding conflicts with
    # the rechunked concat and to_zarr rejects the write as unsafe.
    for var in ds_out.variables:
        ds_out[var].encoding.pop("chunks", None)
        ds_out[var].encoding.pop("preferred_chunks", None)

    # Co-locate tmp with destination so rename stays on the same drive (atomic on Windows/NTFS)
    tmp_path = path.with_name(path.name + ".tmp")

    logger.debug(f"Saving concatenated dataset to {tmp_path}")

    t0 = time.perf_counter()
    for attempt in range(1, 4):
        shutil.rmtree(tmp_path, ignore_errors=True)
        try:
            ds_out.to_zarr(tmp_path)
            break
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(
                    f"Failed saving concatenated dataset to {tmp_path}"
                ) from e
            logger.warning(
                f"[Attempt {attempt}/3] Failed saving to {tmp_path}: {e}. Retrying."
            )
            time.sleep(2**attempt)

    # Release all file handles on path before the swap.  On Windows, open zarr
    # handles prevent shutil.move from renaming the directory ([WinError 32]).
    ds_out.close()
    for ds in src_to_close:
        ds.close()
    del ds_out, src_to_close
    gc.collect()

    # Backup-swap: keep original until new file is confirmed in place
    backup_path = path.with_name(path.name + ".bak")
    logger.debug(f"Atomic swap: {path.name}")
    shutil.move(str(path), str(backup_path))
    try:
        shutil.move(str(tmp_path), str(path))
        shutil.rmtree(str(backup_path), ignore_errors=True)
    except Exception as e:
        shutil.rmtree(str(path), ignore_errors=True)
        shutil.move(str(backup_path), str(path))
        raise RuntimeError(
            f"Failed to swap {tmp_path} → {path}; original restored from backup"
        ) from e
    logger.success(f"Saved in {time.perf_counter() - t0:.1f}s")
    return None


def _try_append_fast_path(ds_new: xr.Dataset, path: Path) -> bool:
    """
    Extend the zarr at *path* in place via ``to_zarr(append_dim="time")``.

    Only attempted when provably safe: identical variable set, identical
    non-time coordinates, matching per-variable dims, and ds_new strictly
    after the stored dates. Returns False otherwise — or when the append or
    its post-write verification fails — and the caller falls back to the
    rewrite path. That fallback also self-heals a partially appended store:
    the overlap resolver discards everything from ds_new's start onward and
    rewrites it from ds_new.
    """
    ds_old = xr.open_zarr(path, consolidated=False)
    try:
        if set(ds_new.data_vars) != set(ds_old.data_vars):
            return False
        if "time" not in ds_new.dims or "time" not in ds_old.dims:
            return False
        for coord in set(ds_old.coords) | set(ds_new.coords):
            if coord == "time":
                continue
            if coord not in ds_old.coords or coord not in ds_new.coords:
                return False
            if not np.array_equal(ds_old[coord].values, ds_new[coord].values):
                return False
        for var in ds_old.data_vars:
            if ds_old[var].dims != ds_new[var].dims:
                return False

        old_end = pd.Timestamp(ds_old.time.values[-1])
        new_start = pd.Timestamp(ds_new.time.values[0])
        if new_start <= old_end:
            return False

        old_n = ds_old.sizes["time"]
        chunk_sizes = {dim: sizes[0] for dim, sizes in ds_old.chunksizes.items()}
    finally:
        ds_old.close()

    # Align dask chunks to the stored zarr chunk grid: the first new chunk
    # exactly fills the partially-written boundary chunk, so no two dask
    # chunks write into the same zarr chunk (which to_zarr rejects as unsafe).
    new_n = ds_new.sizes["time"]
    tchunk = chunk_sizes.get("time") or old_n
    time_chunks: list[int] = []
    boundary = (-old_n) % tchunk
    if boundary:
        time_chunks.append(min(boundary, new_n))
    remaining = new_n - sum(time_chunks)
    while remaining > 0:
        time_chunks.append(min(tchunk, remaining))
        remaining -= time_chunks[-1]

    target = {d: c for d, c in chunk_sizes.items() if d != "time" and d in ds_new.dims}
    target["time"] = tuple(time_chunks)
    ds_append = ds_new.chunk(target)
    # Stale chunk encodings from whatever store ds_new was read from would
    # conflict with the destination's layout — the append uses the store's.
    for var in ds_append.variables:
        ds_append[var].encoding.pop("chunks", None)
        ds_append[var].encoding.pop("preferred_chunks", None)

    logger.info(f"Appending {new_n} timestep(s) to {path.name} in place.")
    t0 = time.perf_counter()
    try:
        ds_append.to_zarr(path, append_dim="time")
    except Exception as e:
        logger.warning(
            f"In-place append to {path.name} failed ({e}) — falling back to rewrite."
        )
        return False

    # Verify the time axis grew as expected and stayed strictly increasing.
    check = xr.open_zarr(path, consolidated=False)
    try:
        times = pd.DatetimeIndex(check.time.values)
        ok = (
            len(times) == old_n + new_n
            and times.is_monotonic_increasing
            and times.is_unique
        )
    finally:
        check.close()
    if not ok:
        logger.warning(f"Post-append verification failed for {path.name} — rewriting.")
        return False

    logger.success(f"Appended in {time.perf_counter() - t0:.1f}s")
    return True


def _resolve_overlap(ds_new: xr.Dataset, path: Path) -> Optional[xr.Dataset]:
    """
    Checks temporal and spatial overlap between the existing zarr and new data.
    Returns the slice of existing data to keep, or None if the existing file
    should be discarded entirely.

    Args:
        ds_new: New dataset to append.
        path: Path to the existing zarr store.

    Raises:
        AssertionError: If geographic extents do not overlap.
    """
    # Open once with chunking — all subsequent slicing is lazy
    ds_old = xr.open_zarr(path, consolidated=False)

    ds_old_vars = set(ds_old.data_vars)
    ds_new_vars = set(ds_new.data_vars)

    if ds_old_vars != ds_new_vars:
        only_in_old = ds_old_vars - ds_new_vars
        only_in_new = ds_new_vars - ds_old_vars
        # A new chunk carrying a subset of the existing variables is the normal
        # backfill case (only_in_old populated, only_in_new empty) — log it at
        # debug. Only genuinely new/unexpected variables warrant a warning.
        if only_in_new:
            logger.warning(
                f"New variables not in existing zarr: {sorted(only_in_new)}."
            )
        else:
            logger.debug(
                f"Backfilling subset; columns held back as existing: "
                f"{len(only_in_old)} not in new data."
            )

    daterange_old = DateRange.from_dataset(ds_old)
    daterange_new = DateRange.from_dataset(ds_new)

    if not BBox.from_dataset(ds_old).overlaps(BBox.from_dataset(ds_new)):
        raise AssertionError(
            f"Geographic extents from stored zarr file {path} and new data does not overlap."
        )

    if daterange_old == daterange_new and ds_old_vars == ds_new_vars:
        logger.info(f"Full overlap with {path.name} — replacing entirely.")
        ds_old.close()
        return None

    if daterange_old.overlaps(daterange_new):
        logger.debug(f"Temporal overlap with {path.name} — merging.")

        if (
            daterange_new.start <= daterange_old.start
            and daterange_new.end >= daterange_old.end
        ):
            logger.info("New data fully contains existing data — replacing entirely.")
            ds_old.close()
            return None

        # Keep the non-overlapping head of ds_old — slice directly, no second zarr open
        cutoff_date = daterange_new.start - pd.Timedelta(days=1)
        start_date = min(daterange_old.start, daterange_new.start)
        ds_subset = ds_old.sel(time=slice(start_date, cutoff_date))
        return ds_subset if len(ds_subset.time) > 0 else None

    return ds_old
