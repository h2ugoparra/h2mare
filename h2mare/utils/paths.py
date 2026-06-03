"""
Path resolution utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from h2mare.config import get_settings
from h2mare.models import KeyVarConfigEntry


def resolve_download_path(
    var_config: KeyVarConfigEntry,
    download_root: Optional[Path] = None,
    warn_if_missing: bool = True,
) -> Path:

    if download_root is not None:
        path = Path(download_root)
    else:
        path = get_settings().DOWNLOADS_DIR / var_config.local_folder

    path = path.resolve()

    if warn_if_missing and not path.exists():
        logger.warning(
            f"Store directory does not exist: {path}. "
            f"Will be created when data is added."
        )

    return path


def resolve_store_path(
    var_config: KeyVarConfigEntry,
    store_root: Optional[Path] = None,
    warn_if_missing: bool = True,
) -> Path:
    """
    Resolve store directory path with fallback hierarchy.
    Adds local_folder in var_config to STORE_ROOT or ZARR_DIR.

    Priority:
        1. Explicit store_root argument
        2. STORE_ROOT environment variable
        3. get_settings().ZARR_DIR

    Args:
        var_config: Variable configuration (for local_folder)
        store_root: Explicit path override
        warn_if_missing: Log warning if path doesn't exist

    Returns:
        Resolved absolute path

    Example:
        >>> path = resolve_store_path(var_config, store_root="/custom/path")
    """
    settings = get_settings()
    if store_root is not None:
        path = Path(store_root)
    elif settings.STORE_ROOT is not None:
        path = settings.STORE_ROOT / var_config.local_folder
    else:
        path = settings.ZARR_DIR / var_config.local_folder

    path = path.resolve()

    if warn_if_missing and not path.exists():
        logger.warning(
            f"Store directory does not exist: {path}. "
            f"Will be created when data is added."
        )

    return path
