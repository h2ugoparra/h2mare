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


def store_root_for(
    var_config: KeyVarConfigEntry,
    default_root: Optional[Path] = None,
) -> Path:
    """
    The root *above* this variable's ``local_folder``.

    Returns a root, never a store directory — callers join ``local_folder``
    themselves. The two are not interchangeable and confusing them has caused
    real bugs (see ``PipelineManager._store_dir``), so this helper commits to
    the root reading and says so in its name.

    Priority:
        1. ``--store-path`` (``Settings.store_root_overridden``)
        2. ``var_config.store_root`` from config.yaml
        3. *default_root*, the root the calling step was handed
        4. ``STORE_ROOT`` from .env
        5. ``get_settings().ZARR_DIR``

    A variable's own root beats the configured ``STORE_ROOT`` — that is the
    point of the field — but not an explicit ``--store-path``, which relocates
    a whole run deliberately.

    *default_root* exists for the steps that already carry a root of their own
    (``PipelineManager.store_root``, ``Compiler.remote_store_root``). Passing it
    keeps those steps behaving exactly as before for variables that declare no
    root, instead of silently re-resolving them from settings.

    Args:
        var_config: Variable configuration (for its optional ``store_root``)
        default_root: Root to use when the variable names none

    Returns:
        Root directory (``local_folder`` NOT appended)
    """
    settings = get_settings()

    if settings.store_root_overridden and settings.STORE_ROOT is not None:
        return settings.STORE_ROOT

    # getattr, not attribute access: stand-in configs in tests and scripts
    # predate the field, the same allowance step_freq makes for time_step.
    if (var_root := getattr(var_config, "store_root", None)) is not None:
        return Path(var_root)

    if default_root is not None:
        return Path(default_root)

    if settings.STORE_ROOT is not None:
        return settings.STORE_ROOT

    return settings.ZARR_DIR


def resolve_store_path(
    var_config: KeyVarConfigEntry,
    store_root: Optional[Path] = None,
    warn_if_missing: bool = True,
) -> Path:
    """
    Resolve store directory path with fallback hierarchy.
    Adds local_folder in var_config to the root chosen by ``store_root_for``.

    Priority:
        1. Explicit store_root argument (used verbatim — see below)
        2. ``--store-path`` override
        3. var_config.store_root from config.yaml
        4. STORE_ROOT environment variable
        5. get_settings().ZARR_DIR

    Note the asymmetry in level 1: an explicit ``store_root`` argument is *one
    store's exact directory* and ``local_folder`` is not appended to it, while
    every other level is a root that ``local_folder`` is joined onto. Callers
    holding a root must not pass it here — that is how every variable once
    ended up pointed at the same directory.

    Args:
        var_config: Variable configuration (for local_folder)
        store_root: Explicit path override — this store's directory
        warn_if_missing: Log warning if path doesn't exist

    Returns:
        Resolved absolute path

    Example:
        >>> path = resolve_store_path(var_config, store_root="/custom/path")
    """
    if store_root is not None:
        path = Path(store_root)
    else:
        path = store_root_for(var_config) / var_config.local_folder

    path = path.resolve()

    if warn_if_missing and not path.exists():
        logger.warning(
            f"Store directory does not exist: {path}. "
            f"Will be created when data is added."
        )

    return path


def static_layer_path(
    var_config: KeyVarConfigEntry, layer: Optional[str], store_dir: Path
) -> Path:
    """
    File of the static *layer* (a key of ``layers`` in config.yaml) under *store_dir*.

    *store_dir* is this variable's own directory (``local_folder`` already
    joined), the shape ``Extractor._store_dir`` and ``resolve_store_path``
    return. A layer that is not declared raises, naming the ones that are.
    """
    layers = getattr(var_config, "layers", None) or {}
    if layer is None or layer not in layers:
        raise ValueError(
            f"Layer {layer!r} is not declared for '{var_config.local_folder}'; "
            f"`layers` in config.yaml has {sorted(layers)}"
        )
    return Path(store_dir) / layers[layer]
