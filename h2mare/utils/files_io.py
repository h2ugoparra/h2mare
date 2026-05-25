"""
Input/Output Help functions
"""

from __future__ import annotations

import os
import shutil
import stat
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger

# ========================== IO ==========================================


def _force_remove(func, path, exc_info):
    """
    Error handler for shutil.rmtree. Tries to make the file writable and retries.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError as e:
        logger.debug(f"_force_remove: could not remove {path}: {e}")


def safe_rmtree(path: Path, retries=10, delay=0.5) -> None:
    """
    Remove a directory tree with retries (prevent Windows file locks).

    Args:
        path: Directory to remove
        retries: Number of retries. Defaults to 10.
        delay: Delay between retries. Defaults to 0.5s.
    """
    last_err = None

    for i in range(retries):
        try:
            if not path.exists():
                return

            shutil.rmtree(path, onerror=_force_remove)
            return

        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(delay * (i + 1))

    raise RuntimeError(
        f"Failed to remove {path} after {retries} attempts"
    ) from last_err


def safe_move_files(
    paths: Path | list[Path], dest_dir: Path, retries=10, delay=0.5
) -> None:
    """
    Move a list of files paths with retries.

    Args:
        paths: File path or List of files paths to move.
        dest_dir: Directory to move file.
        retries: Number of retries. Defaults to 10.
        delay: Delay between retries. Defaults to 0.5s.
    """
    paths = [paths] if isinstance(paths, Path) else paths
    for path in paths:
        dest_path = dest_dir / path.name

        last_err = None

        for i in range(retries):
            try:
                # Avoid errors if file aready exits in dest_dir
                if dest_path.exists():
                    dest_path.unlink()

                logger.debug(
                    f"Moving {path} -> {dest_path} (exists={dest_path.exists()})"
                )
                shutil.move(path, dest_path)
                break

            except (PermissionError, OSError) as e:
                last_err = e
                time.sleep(delay * (i + 1))
        else:
            raise RuntimeError(
                f"Failed to move {path} after {retries} attempts"
            ) from last_err


def move_files(
    source_dir: str | Path, destination_dir: str | Path, file_extension: Optional[str]
):
    """
    Function to move files from source_fir to destination_dir based on file_extension (e.g. 'nc', 'zarr')
    """
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    destination_dir.mkdir(parents=True, exist_ok=True)

    for file_path in source_dir.glob(f"*.{file_extension}"):
        destination_file = destination_dir / file_path.name
        try:
            logger.info(f"Moving {file_path} to {destination_dir}")
            shutil.move(file_path, destination_file)
            logger.success(f"File saved at {destination_file}")

        except Exception as e:
            logger.exception(f"Error moving file {file_path}: {e}")
    return


# ------------------
# Utilities
# ----------------
def unizp_files(zip_path: str | Path, extract_dir: str | Path) -> None:
    """Unzip files that may be downloaded from CLS"""
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
        logger.info(f"Extracted all files to: {extract_dir}")


def clean_era_dataset(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Check for corrupted varibales in ERA5 files"""
    # 1. Ensure time is datetime64
    if not np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = xr.decode_cf(ds)

    # 2. Drop invalid (NaT-like) times
    valid_time_mask = ~np.isnat(ds["time"].values)
    ds = ds.isel(time=valid_time_mask)

    # 3. Drop duplicate times
    _, index = np.unique(ds["time"].values, return_index=True)
    ds = ds.isel(time=np.sort(index))

    # 4. Safely check timesteps one by one
    good_times = []
    for t in ds["time"].values:
        try:
            da = ds[var].sel(time=t)
            if np.isfinite(da).any():
                good_times.append(t)
        except Exception:
            logger.warning(f"Corrupted time index {t} — skipping.")
            continue

    ds = ds.sel(time=good_times)
    return ds
