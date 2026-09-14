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

from h2mare.storage.xarray_helpers import int16_scale, snap_grid_coords
from h2mare.types import BBox


def _read_root_attrs(path: Path) -> dict:
    """Zarr root-group attributes, or ``{}`` when unreadable."""
    try:
        import zarr

        return dict(zarr.open_group(str(path), mode="r").attrs)
    except Exception as e:
        logger.debug(f"Could not read root attrs of {path.name}: {e}")
        return {}


def _restore_root_attrs(path: Path, preserved: dict) -> None:
    """
    Put back root attributes the write dropped, without clobbering new ones.

    Only keys absent after the write are restored, so a value the write
    deliberately set still wins.
    """
    if not preserved:
        return
    try:
        import zarr

        root = zarr.open_group(str(path), mode="r+")
        missing = {k: v for k, v in preserved.items() if k not in root.attrs}
        if missing:
            root.attrs.update(missing)
            logger.debug(f"Restored {sorted(missing)} on {path.name} after append")
    except Exception as e:
        logger.warning(f"Could not restore root attrs on {path.name}: {e}")


def write_append_zarr(
    var_key: str,
    ds: xr.Dataset,
    path: Path,
    encoding: Optional[dict] = None,
) -> None:
    """
    Write dataset, checking temporal overlap and appending data if path exists.

    Args:
        var_key: Variable key, must exist in app_config.variables (used for overlap resolution)
        ds: New dataset to write/append
        path: Destination zarr path, built by the caller via ``ZarrCatalog.build_file_path()``
        encoding: Per-variable zarr encoding, applied only when *path* does not
            exist yet. An existing store keeps the encoding it was created with:
            appends inherit it from disk, and passing one to an append is an
            error. None (the default) writes exactly what it always has.
    """
    # Canonicalize grid labels before any write/append so float-noise drift —
    # between a source's reprocessed periods, or between new data and a legacy
    # store on a slightly different grid — can't union into a doubled axis. The
    # old side is snapped too in _append_data, so an append self-heals the store.
    # Idempotent: a no-op for data already on the rounded grid.
    ds = snap_grid_coords(ds)

    # Recover an orphaned backup before choosing the write path. An append that
    # crashed mid-swap (after moving path → .bak, before moving tmp → path)
    # leaves the only full copy in .bak with path missing. Without this, the
    # branch below would take the new-file path and overwrite the store with
    # just the increment, silently dropping the store's history.
    _restore_orphaned_backup(path)

    if path.exists():
        # No log here: each append path announces itself (in-place append,
        # variable-addition, or overlap merge).
        #
        # The in-place fast path ends in to_zarr(append_dim="time"), which
        # rewrites the group attrs from the incoming dataset — and that carries
        # none, so the store's own metadata was wiped. The rewrite paths keep
        # them (xr.concat/merge carry ds_old's attrs through), which is why this
        # only ever bit strictly-after appends: the common case.
        #
        # source_datasets is the casualty that matters. Provenance stamped by
        # one run vanished on the next, so merge_records never found anything to
        # merge with and backfill_provenance's "skip files that already have the
        # attr" was undone by the following append.

        # Before anything is written: a packed store's scale was fixed at
        # creation and an append inherits it, so data outside that range has no
        # encoding and would wrap rather than clip. Widen the scale first by
        # repacking the store. No-op for float32 stores.
        overflow = _check_packed_range(ds, path)
        if overflow:
            _repack_store(path, overflow)

        preserved = _read_root_attrs(path)
        _append_data(var_key, ds, path)
        _restore_root_attrs(path, preserved)

    else:
        logger.info(f"Saving new dataset at {path}")
        t0 = time.perf_counter()
        # Write to a sibling tmp and promote it only after verification. A hard
        # kill mid-write then leaves a *.zarr.tmp (which the catalog scanner
        # ignores — it globs *.zarr) instead of a partial *.zarr that gap
        # detection would trust as complete and never re-convert.
        tmp_path = path.with_name(path.name + ".tmp")
        shutil.rmtree(tmp_path, ignore_errors=True)
        try:
            ds.to_zarr(tmp_path, encoding=encoding or None)
            xr.open_zarr(tmp_path, consolidated=False).close()
        except Exception as e:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise RuntimeError(f"Zarr write verification failed for {path}") from e
        finally:
            ds.close()
        shutil.move(str(tmp_path), str(path))
        logger.success(f"Saved in {time.perf_counter() - t0:.1f}s")


def _restore_orphaned_backup(path: Path) -> None:
    """
    Restore a ``*.zarr.bak`` left by an append that crashed mid-swap.

    The backup-swap in :func:`_append_data` moves ``path`` → ``path.bak`` and
    then ``tmp`` → ``path``. A hard kill between those two moves leaves ``path``
    missing with the only complete copy in ``path.bak``. Restoring it here lets
    the caller take the append (history-preserving) path instead of the
    new-file path. No-op when ``path`` exists or no backup is present.
    """
    backup_path = path.with_name(path.name + ".bak")
    if not path.exists() and backup_path.exists():
        logger.warning(
            f"Restoring orphaned backup {backup_path.name} → {path.name} "
            "(previous append was interrupted mid-swap)."
        )
        shutil.move(str(backup_path), str(path))


# Smallest step pandas can represent at datetime64[ns], so `t - _ONE_TICK` is
# the largest instant strictly before t. Used to make a window boundary
# exclusive without assuming anything about the axis's cadence.
_ONE_TICK = pd.Timedelta(nanoseconds=1)


def _time_bounds(ds: xr.Dataset) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last actual timestamps on *ds*'s time axis."""
    times = ds.time.values
    return pd.Timestamp(times[0]), pd.Timestamp(times[-1])


def _check_packed_range(
    ds_new: xr.Dataset, path: Path
) -> dict[str, tuple[float, float]]:
    """
    Find the variables whose values the store's frozen packing cannot represent.

    A scale/offset store fixes ``scale_factor`` and ``add_offset`` when it is
    *created*, from the range of whatever batch was written first, and every
    later append inherits them (see :func:`write_append_zarr`'s ``encoding``
    argument). A value outside that range has no encoding — and it does not
    clip, it overflows the integer dtype and wraps back into the middle of the
    range. A store scaled for msl 99000-102000 Pa, appended with a storm year
    reaching 92000 Pa, read back with a 9074 Pa error and no warning: the
    wrapped values land *inside* the plausible band, so there are no outliers
    to notice and no NaNs to count. The caller repacks the store over the wider
    range (:func:`_repack_store`) before appending.

    Costs one pass over *ds_new* to find its bounds, and only for stores that
    are actually packed — a float32 store returns before reading anything.
    Every variable's min and max go into a single graph, so the source is
    decoded once rather than twice per variable.

    Note this checks the representable *range* only. A value that happens to
    encode exactly onto ``_FillValue`` still reads back as NaN; that is a far
    rarer single-value collision, not the systematic wraparound above.

    Returns:
        The incoming ``(min, max)`` of each variable that falls outside the
        store's packing; empty when everything fits.
    """
    import dask

    ds_old = xr.open_zarr(path, consolidated=False)
    try:
        packed = {}
        for name in ds_new.data_vars:
            if name not in ds_old.data_vars:
                continue
            enc = ds_old[name].encoding
            scale = enc.get("scale_factor")
            dtype = np.dtype(enc.get("dtype", "f4"))
            if scale is None or dtype.kind not in "iu":
                continue
            packed[name] = (float(scale), float(enc.get("add_offset", 0.0)), dtype)
    finally:
        ds_old.close()

    if not packed:
        return {}

    logger.info(
        f"Checking {len(packed)} packed variable(s) against {path.name}'s stored "
        "scale (one pass over the incoming data)"
    )
    bounds: dict[tuple[str, str], xr.DataArray] = {}
    for name in packed:
        bounds[(str(name), "lo")] = ds_new[name].min()
        bounds[(str(name), "hi")] = ds_new[name].max()
    (computed,) = dask.compute(bounds)

    overflow: dict[str, tuple[float, float]] = {}
    for name, (scale, offset, dtype) in packed.items():
        lo = float(computed[(str(name), "lo")])
        hi = float(computed[(str(name), "hi")])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            # All-NaN increment: nothing to encode, nothing to overflow.
            continue
        info = np.iinfo(dtype)
        repr_lo = offset + info.min * scale
        repr_hi = offset + info.max * scale
        if lo < repr_lo or hi > repr_hi:
            logger.warning(
                f"{name} ({dtype.name}): incoming [{lo:.6g}, {hi:.6g}] is outside "
                f"{path.name}'s representable [{repr_lo:.6g}, {repr_hi:.6g}]"
            )
            overflow[str(name)] = (lo, hi)
    return overflow


def _repack_store(path: Path, overflow: dict[str, tuple[float, float]]) -> None:
    """
    Rewrite *path* with each overflowing variable's scale widened to fit.

    The new scale spans the stored data's actual range joined with the
    incoming one, through :func:`int16_scale` — so the usual headroom is
    applied once. Deriving it from the old *representable* window instead would
    compound that headroom on every repack. Variables that already fit keep
    their encoding untouched, so their values round-trip exactly.

    Costs a pass over the overflowing variables and a rewrite of the whole
    store: a yearly file, and only when its range is exceeded. Existing values
    move by at most half a quantisation step of the old and new scales. The
    write goes to a sibling tmp and is swapped in as :func:`_append_data` does,
    so a failure leaves the store as it was.

    Raises:
        ValueError: If a variable cannot be repacked — not int16, or a range
            with no span to derive a scale from. Nothing is written.
    """
    import dask

    ds_old = xr.open_zarr(path, consolidated=False)
    try:
        bounds: dict[tuple[str, str], xr.DataArray] = {}
        for name in overflow:
            bounds[(name, "lo")] = ds_old[name].min()
            bounds[(name, "hi")] = ds_old[name].max()
        (computed,) = dask.compute(bounds)

        problems: list[str] = []
        for name, (in_lo, in_hi) in overflow.items():
            var = ds_old.variables[name]
            dtype = np.dtype(var.encoding["dtype"])
            lo = float(np.nanmin([float(computed[(name, "lo")]), in_lo]))
            hi = float(np.nanmax([float(computed[(name, "hi")]), in_hi]))
            scale = int16_scale(lo, hi) if dtype == np.int16 else None
            if scale is None:
                problems.append(
                    f"  {name} ({dtype.name}): incoming [{in_lo:.6g}, {in_hi:.6g}] "
                    f"cannot be repacked (only int16 with a non-degenerate range)"
                )
                continue
            logger.warning(
                f"Repacking {name} in {path.name} over [{lo:.6g}, {hi:.6g}] "
                f"(rewrites the whole store)"
            )
            var.encoding = {**var.encoding, **scale}

        if problems:
            raise ValueError(
                f"{path.name} is scale/offset-packed with a scale fixed when it "
                f"was created, and cannot represent the incoming data:\n"
                + "\n".join(problems)
                + f"\nAppending would silently wrap these values back into range. "
                f"Re-convert this store so the scale is re-derived over the full "
                f"range (delete {path.name} and re-run the convert step for it)."
            )

        tmp_path = path.with_name(path.name + ".tmp")
        shutil.rmtree(tmp_path, ignore_errors=True)
        t0 = time.perf_counter()
        try:
            ds_old.to_zarr(tmp_path)
            xr.open_zarr(tmp_path, consolidated=False).close()
        except Exception as e:
            shutil.rmtree(tmp_path, ignore_errors=True)
            raise RuntimeError(f"Repack of {path.name} failed") from e
    finally:
        # Release the handles on path before the swap ([WinError 32]).
        ds_old.close()
        del ds_old
        gc.collect()

    _swap_into_place(tmp_path, path)
    logger.success(f"Repacked {path.name} in {time.perf_counter() - t0:.1f}s")


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
        missing_vars = sorted(ds_old_vars - ds_new_vars)
        old_chunk_sizes = {dim: sizes[0] for dim, sizes in ds_old.chunksizes.items()}
        ds_old.close()

        # Clean trailing append (same vars, same grid, strictly after the
        # stored dates) extends the zarr in place — by year's end the rewrite
        # path would otherwise copy ~a full year of data per incremental run.
        if _try_append_fast_path(ds_new, path):
            return None

        ds_resolved = _resolve_overlap(ds_new, path)

        # ds_new is already snapped (write_append_zarr); snap the retained
        # pieces of the existing store so concat aligns instead of unioning a
        # legacy noise-drifted grid against the rounded one.
        # Head and tail contribute only variables that ds_new carries AND that
        # have a time dimension. Variables ds_new lacks are re-merged in full
        # below (concatenating them here would NaN-fill them over ds_new's
        # window); time-less statics (e.g. bathy) are not concatenated by
        # xr.concat but merged with an exact-equality check, so a float-level
        # recompute difference raises MergeError — the freshly compiled copy
        # in ds_new wins instead.
        def _shared_time_vars(d: xr.Dataset) -> xr.Dataset:
            return d[
                [v for v in d.data_vars if v in ds_new_vars and "time" in d[v].dims]
            ]

        parts = [ds_new]
        if ds_resolved is not None:
            parts.insert(0, _shared_time_vars(snap_grid_coords(ds_resolved)))
            src_to_close.append(ds_resolved)

        # Retain the existing tail beyond ds_new's window: an explicit
        # middle-window compile (e.g. --start/--end covering March against a
        # file that extends through June) would otherwise drop April-June.
        new_end = pd.Timestamp(ds_new.time.values[-1])
        ds_old_tail = xr.open_zarr(path, consolidated=False)
        # Strictly after the incoming window's last instant. A `+ 1 day` bound
        # here would skip every stored step falling later the same day as
        # new_end — 23 of them on an hourly axis.
        tail = ds_old_tail.sel(time=slice(new_end + _ONE_TICK, None))
        if tail.sizes.get("time", 0):
            parts.append(_shared_time_vars(snap_grid_coords(tail)))
            src_to_close.append(ds_old_tail)
        else:
            ds_old_tail.close()

        if len(parts) > 1:
            # join is explicit because xarray's concat default changes from
            # "outer" to "exact"; without it this call starts raising once that
            # lands. "outer" pins today's behaviour — note it is also the
            # failure mode the snapping above guards against, so a genuine grid
            # mismatch here unions into a NaN-holed axis rather than failing.
            ds_out = xr.concat(parts, dim="time", data_vars="minimal", join="outer")
            # Rechunk to match the existing zarr layout and avoid dask
            # chunk-alignment errors.
            ds_out = ds_out.chunk(
                {d: c for d, c in old_chunk_sizes.items() if d in ds_out.dims}
            )
        else:
            ds_out = ds_new

        if missing_vars:
            # Variables absent from ds_new keep their stored values over the
            # full existing span — including ds_new's window. Without this, a
            # subset compile (e.g. `run -v ssh` appending only ssh columns)
            # NaN-wiped every other variable across the appended range.
            ds_old_keep = xr.open_zarr(path, consolidated=False)[missing_vars]
            ds_out = xr.merge([ds_out, snap_grid_coords(ds_old_keep)], join="outer")
            ds_out = ds_out.chunk(
                {d: c for d, c in old_chunk_sizes.items() if d in ds_out.dims}
            )
            src_to_close.append(ds_old_keep)

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

    _swap_into_place(tmp_path, path)
    logger.success(f"Saved in {time.perf_counter() - t0:.1f}s")
    return None


def _swap_into_place(tmp_path: Path, path: Path) -> None:
    """Replace *path* with *tmp_path*, restoring the original on failure."""
    # Backup-swap: keep original until new file is confirmed in place
    backup_path = path.with_name(path.name + ".bak")
    logger.debug(f"Atomic swap: {path.name}")
    # Clear a stale backup first. A crash between the tmp → path move below and
    # the rmtree that follows it leaves *both* path and path.bak present, and
    # _restore_orphaned_backup only handles the other direction (path missing).
    # shutil.move onto an existing directory moves *into* it rather than
    # replacing it, so without this the store lands at path.bak/<name>.zarr and
    # the rollback branch restores a directory mixing stale files with a nested
    # store — losing the history in the one branch that exists to preserve it.
    # Reaching here means path exists, so any surviving backup is known-stale.
    if backup_path.exists():
        logger.warning(
            f"Discarding stale backup {backup_path.name} left by an earlier "
            "interrupted append."
        )
        shutil.rmtree(backup_path, ignore_errors=True)
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
        chunk_sizes: dict[str, int] = {
            str(dim): sizes[0] for dim, sizes in ds_old.chunksizes.items()
        }
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

    target: dict[str, int | tuple[int, ...]] = {
        d: c for d, c in chunk_sizes.items() if d != "time" and d in ds_new.dims
    }
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

    # Boundaries come from the actual timestamps, not from DateRange, which
    # normalizes both ends to midnight. Day-granular boundaries silently drop
    # data whenever a store's stamps are not at midnight: 23 hours per boundary
    # on an hourly axis, and the whole preceding day on a noon-stamped daily one.
    old_first, old_last = _time_bounds(ds_old)
    new_first, new_last = _time_bounds(ds_new)

    if not BBox.from_dataset(ds_old).overlaps(BBox.from_dataset(ds_new)):
        # A plain ValueError (not assert — assertions vanish under `python -O`)
        # with an actionable message: this usually means the config bbox changed
        # between the stored history and the new run, so the two no longer share
        # any geography to merge.
        ds_old.close()
        raise ValueError(
            f"Geographic extents of stored zarr {path.name} "
            f"({BBox.from_dataset(ds_old)}) and new data "
            f"({BBox.from_dataset(ds_new)}) do not overlap — has the configured "
            "bbox changed since this store was written?"
        )

    if (old_first, old_last) == (new_first, new_last) and ds_old_vars == ds_new_vars:
        logger.info(f"Full overlap with {path.name} — replacing entirely.")
        ds_old.close()
        return None

    if old_first <= new_last and old_last >= new_first:
        logger.debug(f"Temporal overlap with {path.name} — merging.")

        if new_first <= old_first and new_last >= old_last:
            logger.info("New data fully contains existing data — replacing entirely.")
            ds_old.close()
            return None

    # Keep the head of ds_old that sits strictly before the incoming window
    # — slice directly, no second zarr open. The bound is one tick short of
    # new_first rather than a day short, so a step on the same day as the
    # incoming start but earlier than it is retained rather than discarded.
    #
    # Disjoint windows land here too, and must: returning ds_old whole for them
    # put it in the head *and* in the tail _append_data selects after ds_new,
    # so a store holding June-December that was backfilled with January came
    # back with June-December written twice and a non-monotonic time axis. When
    # ds_new is the earlier side this slice is empty (None) and the tail alone
    # contributes ds_old; when it is the later side the slice is all of ds_old
    # and the tail is empty. One expression covers overlap and disjoint alike,
    # which is what keeps the two from drifting apart again.
    ds_subset = ds_old.sel(time=slice(None, new_first - _ONE_TICK))
    return ds_subset if len(ds_subset.time) > 0 else None
