"""
Zarr Save/Temporal-overlap-check Logic
"""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.storage.xarray_helpers import have_vars_unique_values
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
    if path.exists():
        logger.warning(f"{path} already exists.")
        _append_data(var_key, ds, path)

    else:
        logger.info(f"Saving new dataset at {path}")
        ds.to_zarr(path)
        try:
            xr.open_zarr(path, consolidated=False).close()
        except Exception as e:
            shutil.rmtree(path, ignore_errors=True)
            raise RuntimeError(f"Zarr write verification failed for {path}") from e
        ds.close()
        logger.success("Saved")


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
        ds_out = xr.merge([ds_old, ds_new], join="outer")
        chunk_sizes = {dim: sizes[0] for dim, sizes in ds_old.chunksizes.items()}
        ds_out = ds_out.chunk(chunk_sizes)
        src_to_close.append(ds_old)
    else:
        # Time-extension: at least one shared variable — resolve temporal overlap
        # then concatenate along the time dimension.
        # ds_old is reopened inside _resolve_overlap; closing here avoids a
        # redundant open handle (zarr stores are lazy so this is safe).
        ds_old.close()
        ds_resolved = _resolve_overlap(ds_new, path)

        if ds_resolved is not None:
            ds_out = xr.concat([ds_resolved, ds_new], dim="time", data_vars="minimal")
            # Rechunk to match the existing zarr layout and avoid dask chunk-alignment errors.
            chunk_sizes = {
                dim: sizes[0] for dim, sizes in ds_resolved.chunksizes.items()
            }
            ds_out = ds_out.chunk(chunk_sizes)
            src_to_close.append(ds_resolved)
        else:
            ds_out = ds_new

    # Check if file is corrupted (warning only — does not abort)
    have_vars_unique_values(ds_out)

    # Co-locate tmp with destination so rename stays on the same drive (atomic on Windows/NTFS)
    tmp_path = path.with_name(path.name + ".tmp")

    logger.debug(f"Saving concatenated dataset to {tmp_path}")

    for attempt in range(1, 4):
        try:
            ds_out.to_zarr(tmp_path, align_chunks=True)
            break
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(
                    f"Failed saving concatenated dataset to {tmp_path}"
                ) from e
            logger.warning(
                f"[Attempt {attempt}/3] Failed saving to {tmp_path}: {e}. Retrying."
            )
            shutil.rmtree(tmp_path, ignore_errors=True)
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
    logger.success("Completed")
    return None


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
        logger.warning(
            f"Variable mismatch between existing zarr and new data. "
            f"Only in existing: {only_in_old}. Only in new: {only_in_new}."
        )

    daterange_old = DateRange.from_dataset(ds_old)
    daterange_new = DateRange.from_dataset(ds_new)

    if not BBox.from_dataset(ds_old).overlaps(BBox.from_dataset(ds_new)):
        raise AssertionError(
            f"Geographic extents from stored zarr file {path} and new data does not overlap."
        )

    if daterange_old == daterange_new and ds_old_vars == ds_new_vars:
        logger.warning(
            f"Full temporal overlap between {path} and new data. Replacing entirely."
        )
        ds_old.close()
        return None

    if daterange_old.overlaps(daterange_new):
        logger.warning(f"Temporal overlap between {path} and new data.")

        if (
            daterange_new.start <= daterange_old.start
            and daterange_new.end >= daterange_old.end
        ):
            logger.warning("New data fully contains existing data. Replacing entirely.")
            ds_old.close()
            return None

        # Keep the non-overlapping head of ds_old — slice directly, no second zarr open
        cutoff_date = daterange_new.start - pd.Timedelta(days=1)
        start_date = min(daterange_old.start, daterange_new.start)
        ds_subset = ds_old.sel(time=slice(start_date, cutoff_date))
        return ds_subset if len(ds_subset.time) > 0 else ds_old

    return ds_old
