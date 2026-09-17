"""
Extract data based on csv or shapefile format files from datasets in zarr format.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional, Sequence, Union, overload

import ephem
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  # registers .rio accessor on xarray objects
import xarray as xr
from loguru import logger
from rasterio.errors import NotGeoreferencedWarning
from scipy.spatial import KDTree

from h2mare import AppConfig, get_settings
from h2mare.models import check_depth_levels, depth_levels_for, step_freq
from h2mare.storage.var_routing import compiled_var_key
from h2mare.storage.xarray_helpers import select_depth_levels
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox, DateRange, ReadFrom
from h2mare.utils.datetime_utils import end_of_day
from h2mare.utils.logging import configure_extraction_logging, log_time
from h2mare.utils.paths import store_root_for
from h2mare.utils.spatial import sel_padded_bbox

#: How the input's own timestamps are read. ``auto`` infers it from the data;
#: ``daily`` and ``hourly`` state it outright. Purely about parsing ``time_col``
#: — which store answers is :data:`~h2mare.types.ReadFrom`.
TimeCadence = Literal["auto", "daily", "hourly"]

#: What ``run()`` accepts per var_key: one variable, a list of them, ``None``
#: for everything, or ``{variable: depths | None}`` to also choose the depth
#: levels (metres) of 3-D variables for this request only.
VarSelection = Union[str, list[str], Mapping[str, Optional[Sequence[int]]], None]

#: Coordinate columns that ride out of ``to_dataframe()`` alongside the real
#: values. Carried by every engine result, so they are stripped before a join
#: and again from the final frame.
_COORD_COLS = ["time", "lat", "lon", "geom"]

#: Half-width added to a bbox axis that came out with no width. Only has to
#: clear ``BBox``'s ``xmin < xmax`` check: the read pads by a whole grid cell on
#: each side anyway (:func:`~h2mare.utils.spatial.sel_padded_bbox`), so the
#: cells surrounding the point are in the subset either way. It does not make a
#: one-point query identical to a wide one — a sample landing exactly halfway
#: between two cell centres is a tie, and which of them the nearest-neighbour
#: search returns depends on the subset it was given. That was already true of
#: any two differently-sized bboxes.
_DEGENERATE_BBOX_PAD = 1e-6

# Module-level KDTree cache keyed on grid identity (shape + first/last values).
# All var_keys produced by this pipeline share the same 0.25° grid, so the tree
# is built once per process and reused across every extract_from_csv call.
_kdtree_cache: dict[tuple, tuple[KDTree, int, int]] = {}


# ===== BACKUP FUNC FOR INCOMPLETE EXTRACTIONS =====
def _keys_path(tmp_path: Path) -> Path:
    return tmp_path.with_suffix(".keys.json")


def input_fingerprint(data: pd.DataFrame, index_col: str) -> str:
    """
    Stable digest of the frame an extraction consumes.

    The checkpoint lives at one fixed path, so the next run finds whatever the
    last one left there. Without a fingerprint it is resumed on faith: a
    different input of the same shape has its rows replayed from the previous
    run, silently and with the right index. ``ensure_row_id`` makes that easy
    to hit — its key is positional ``0..n-1``, so any two frames of the same
    length collide perfectly.

    Covers the key column's name and values plus every prepared column, since
    those are exactly what extraction reads. Geometries go in as WKB, which
    pandas can hash and shapely objects cannot.
    """
    parts: list[bytes] = [index_col.encode(), str(len(data)).encode()]

    def digest(values) -> bytes:
        return pd.util.hash_pandas_object(
            pd.Series(list(values)), index=False
        ).values.tobytes()

    parts.append(digest(data.index))
    for col in sorted(map(str, data.columns)):
        parts.append(col.encode())
        series = data[col]
        if col == "geometry" and hasattr(series, "to_wkb"):
            parts.append(digest(series.to_wkb()))
        else:
            parts.append(digest(series.astype(str)))

    return hashlib.sha256(b"".join(parts)).hexdigest()


def _save_completed_keys(tmp_path: Path, keys: set[str], fingerprint: str) -> None:
    dest = _keys_path(tmp_path)
    staging = dest.with_suffix(".tmp")
    with open(staging, "w") as f:
        json.dump({"fingerprint": fingerprint, "completed": sorted(keys)}, f)
    staging.replace(dest)


def _load_completed_keys(tmp_path: Path, fingerprint: str) -> set[str] | None:
    """
    The var_keys already done for *fingerprint*, or None if the checkpoint is
    not this input's.

    None means "discard and start over": a sidecar that is missing, unreadable,
    written by an older version without a fingerprint, or written for a
    different input. Resuming across any of those replays another run's values.
    """
    path = _keys_path(tmp_path)
    if not path.exists():
        return None

    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Checkpoint sidecar unreadable ({e}); starting over.")
        return None

    if not isinstance(payload, dict) or "fingerprint" not in payload:
        logger.warning(
            "Checkpoint predates input fingerprinting, so it cannot be matched "
            "to this input; starting over."
        )
        return None

    if payload["fingerprint"] != fingerprint:
        logger.warning(
            "Checkpoint was written for a different input — same path, "
            "different data. Discarding it and starting over rather than "
            "replaying that run's rows onto yours."
        )
        return None

    return set(payload.get("completed", []))


def null_summary_lines(result: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    One line per extracted variable: its null count, and what share that is.

    The share is shown only where something is actually null. A clean run
    should read as a column of zeros rather than a column of "(0.0%)" to scan
    past, and the point of the percentage is to tell a couple of stray
    geometries from a variable that came back mostly empty.
    """
    total = len(result)
    lines = []

    for col, count in result[columns].isnull().sum().items():
        share = f" ({count / total:.1%})" if count and total else ""
        lines.append(f"  {col}: {count}{share}")

    return lines


def _warn_if_wholly_failed(result: pd.DataFrame, errors: list[Exception]) -> None:
    """
    Say something when *every* geometry came back empty.

    A few NaN rows are ordinary — geometries outside the grid clip to nothing,
    and that is data, not a fault. Every row failing is not: it means the
    dataset could not be clipped at all, usually because rioxarray cannot
    identify the spatial dims or there is no CRS. That used to surface only as
    a DEBUG line per geometry, leaving an all-null column looking like absent
    data rather than a broken precondition.
    """
    if result.empty or not errors:
        return

    values = result.select_dtypes("number")
    if values.empty or not bool(values.isna().all().all()):
        return

    logger.warning(
        f"Every geometry returned NaN across {len(result)} row(s) — the dataset "
        f"could not be clipped at all, rather than the geometries falling "
        f"outside it. First error: {type(errors[0]).__name__}: {errors[0]}"
    )


def _extract_geometry(
    id: str,
    date,
    geom,
    ds: xr.DataArray | xr.Dataset,
    index_col: str,
    errors: list[Exception] | None = None,
) -> dict:
    """
    Extract data and return as dictionary for a single geometry row.

    The dataset is already loaded in memory by the caller, so a clip failure
    (e.g. geometry outside the grid) is deterministic — log it and return
    NaNs for the row rather than retrying.

    Args:
        id (str): index value of the geometry row.
        date (): date value of the geometry row.
        geom (): geometry of the geometry row.
        ds (xr.DataArray | xr.Dataset): in-memory xarray object.

    Returns:
        dict: dictionary with index, variable names and extracted values.
    """
    is_dataset = isinstance(ds, xr.Dataset)
    data_vars: list[str] = [str(v) for v in ds.data_vars] if is_dataset else []
    single_var_name: str = str(ds.name) if (not is_dataset and ds.name) else "value"

    if date is not None:
        ds = ds.sel(time=date, method="nearest")

    try:
        clipped = ds.rio.clip([geom], drop=True, all_touched=True).mean()

        result: dict = {index_col: id}

        if not is_dataset:
            # Single variable
            result[single_var_name] = clipped.item()
        else:
            # Dataset: extract each variables
            for var in clipped.data_vars:
                # Ensure scalar
                result[var] = clipped[var].item()

        return result

    except (OSError, ValueError, RuntimeError) as e:
        # Per-geometry detail only — thousands of geometries would flood the
        # log at ERROR. The end-of-run null summary carries the aggregate, and
        # `errors` lets the caller tell "all of them failed" (a broken
        # precondition) from "some fell outside the grid" (ordinary).
        logger.debug(f"Extraction failed for id={id}, date={date}: {e}")
        if errors is not None:
            errors.append(e)

    # --- Return NaNs for failed geometry to preserve structure ---
    nan_result: dict = {index_col: id}
    if is_dataset:
        nan_result.update({var: float("nan") for var in data_vars})
    else:
        nan_result[single_var_name] = float("nan")

    return nan_result


def _extract_geometry_bathy(
    id: str, geom, ds: xr.DataArray | xr.Dataset, index_col: str
) -> dict:
    """
    Extract bathymetry data (mean and std over the clipped geometry) and
    return as dictionary for a single geometry row. As in
    :func:`_extract_geometry`, failures return NaNs without retrying.

    Args:
        id (str): index value of the geometry row.
        geom (): geometry of the geometry row.
        ds (xr.DataArray | xr.Dataset): in-memory xarray object.

    Returns:
        dict: dictionary with index, variable names and extracted values.
    """
    is_dataset = isinstance(ds, xr.Dataset)
    data_vars: list[str] = [str(v) for v in ds.data_vars] if is_dataset else []
    single_var_name: str = str(ds.name) if (not is_dataset and ds.name) else "value"

    try:
        clipped = ds.rio.clip([geom], drop=True, all_touched=True)
        mean_ds = clipped.mean(dim=None)
        std_ds = clipped.std(dim=None)

        result: dict = {index_col: id}

        if is_dataset:
            for var in clipped.data_vars:
                result[f"{var}"] = mean_ds[var].item()
                result[f"{var}_std"] = std_ds[var].item()
        else:
            result[single_var_name] = mean_ds.item()
            result[f"{single_var_name}_std"] = std_ds.item()

        return result

    except (OSError, ValueError, RuntimeError) as e:
        # Per-geometry detail only — see _extract_geometry.
        logger.debug(f"Extraction failed for id={id}: {e}")

    # --- Return NaNs for failed geometry to preserve structure ---
    nan_result: dict = {index_col: id}
    if is_dataset:
        nan_result.update({var: float("nan") for var in data_vars})
    else:
        nan_result[single_var_name] = float("nan")
        nan_result[f"{single_var_name}_std"] = float("nan")

    return nan_result


def _declared_vars(var_config) -> list[str]:
    """Variables this var_key publishes, per ``compiled_vars`` in config."""
    return list(getattr(var_config, "compiled_vars", None) or [])


def _widen_degenerate(
    bounds: Sequence[float],
) -> tuple[float, float, float, float]:
    """
    Nudge apart any axis of *bounds* whose min equals its max.

    Only exact equality, so an inverted box still reaches ``BBox`` and is still
    rejected — that one is a real defect, not a shape the input can take.
    """
    xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    if xmin == xmax:
        xmin, xmax = xmin - _DEGENERATE_BBOX_PAD, xmax + _DEGENERATE_BBOX_PAD
    if ymin == ymax:
        ymin, ymax = ymin - _DEGENERATE_BBOX_PAD, ymax + _DEGENERATE_BBOX_PAD
    return xmin, ymin, xmax, ymax


def resolve_read_from(var_config, *, read_from: ReadFrom, subdaily_input: bool) -> str:
    """
    Which store answers this var_key.

    ``native`` and ``compiled`` are honoured as given. ``auto`` decides per
    var_key from the cadence the input asked for:

    - A **daily** store holds everything its var_key publishes, so it always
      answers for itself.
    - An **hourly** store is the raw *source*: it holds neither the daily
      reduction nor the features derived from it, and snapping a date-only row
      to the nearest stored step lands it on one arbitrary hour. A date-only
      query is therefore answered from the compiled store, where those numbers
      actually live — the same ones ``ParquetIndexer.scan`` returns. A sub-daily
      query reads the hourly store, since that is the only thing that has hours.

    Returns:
        ``"native"`` for the per-variable Zarr, ``"compiled"`` for h2ds.
    """
    if read_from != "auto":
        return read_from
    if step_freq(var_config) != "h":
        return "native"
    return "native" if subdaily_input else "compiled"


def warn_on_subdaily_store(var_key: str, var_config, ds: xr.Dataset) -> None:
    """
    Say out loud that native hourly values do not read like daily ones.

    Only fires when the hourly store is actually being served — a date-only
    query routes to the compiled store (see :func:`resolve_read_from`), where the
    semantics and units are the daily ones the caller already expects, so
    warning there would be noise.

    Each sample snaps to the nearest stored step, which here is one
    instantaneous hour rather than the day's aggregate. Units differ too,
    because an hourly store holds the raw source rather than the pipeline's
    daily reduction (ERA5 ``msl`` is Pa here and hPa in h2ds) — so the stored
    units are reported instead of being left for the caller to discover in the
    numbers.
    """
    if step_freq(var_config) != "h":
        return

    units = {str(v): ds[v].attrs.get("units", "?") for v in ds.data_vars}
    logger.warning(
        f"[{var_key}] hourly store: each sample snaps to the nearest hour and "
        f"returns that instantaneous value, NOT a daily aggregate. Units are "
        f"the raw source's and may differ from the daily store: {units}"
    )


def split_depth_request(
    var_key: str, vars: VarSelection
) -> tuple[str | list[str] | None, dict[str, list[int]]]:
    """
    Separate a ``{variable: depths | None}`` selection into names and levels.

    The dict's keys are the variables requested, exactly as a list would name
    them; a non-``None`` value chooses that variable's depth levels for this
    request, over whatever config declares. Any other selection passes through
    with no levels.
    """
    if vars is None or isinstance(vars, (str, list)):
        return vars, {}

    override: dict[str, list[int]] = {}
    for name, levels in vars.items():
        if levels is None:
            continue
        if not isinstance(levels, (list, tuple)):
            raise TypeError(
                f"var_dict[{var_key!r}][{name!r}] must be a list of depths in "
                f"metres or None; got {levels!r}"
            )
        check_depth_levels(f"var_dict[{var_key!r}][{name!r}]", list(levels))
        override[name] = list(levels)
    return list(vars), override


def split_vars_by_source(
    requested: list[str] | None,
    stored: list[str],
    var_key: str,
    var_config,
    *,
    has_depth: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Split a request into what the native store holds and what the compiled owes.

    A var_key's ``compiled_vars`` are what it *publishes*. Whether its own store
    holds all of that depends on its ``time_step``: converting at **daily**
    cadence runs the reductions and the derived chain up front and writes them
    alongside the raw fields, so the store is complete; converting at **hourly**
    cadence keeps the raw source only and moves everything derived to compile
    time, where it is written to the compiled store instead. The same var_key is
    therefore complete in one deployment and thin in another — which is why the
    gap is routed rather than refused.

    ``requested=None`` means "everything this var_key publishes".

    The rule, uniformly: absent where the design puts it elsewhere is a routing
    decision; absent where the design says it should be present is a defect. A
    daily store missing what it publishes is the second kind, so it raises here
    rather than being quietly backfilled from the compiled store — that would
    mask a hole in the store.

    ``has_depth`` disables the reconciliation entirely, because for a 3-D
    variable ``compiled_vars`` and the store are not comparable: the store holds
    variables on a ``depth`` axis (``thetao``) while ``compiled_vars`` names
    the columns they become after slicing (``thetao_100``, …). Nor can the check
    simply be deferred until after the expansion — extraction slices at its own
    levels (``extract_depth_levels``, or levels chosen in the request) while
    ``compiled_vars`` follows ``depth_levels``, and the two may differ. There is
    nothing to reconcile; :meth:`Extractor._slice_depth` owns which levels
    appear.

    Returns:
        ``(from_native, from_compiled)``. For a daily store ``from_compiled`` is
        always empty. With ``has_depth`` the request passes through untouched,
        to be validated against the post-expansion names instead.

    Raises:
        ValueError: for names that belong to neither side, and for a daily store
            that cannot satisfy what its own config publishes.
    """
    if has_depth:
        return (list(requested) if requested is not None else []), []

    declared = _declared_vars(var_config)
    stored_set, declared_set = set(stored), set(declared)

    wanted = list(requested) if requested is not None else (declared or list(stored))

    unknown = sorted(v for v in wanted if v not in stored_set and v not in declared_set)
    if unknown:
        raise ValueError(
            f"[{var_key}] cannot extract {unknown}: not variables of "
            f"'{var_key}'. Store holds {sorted(stored)}"
            + (f"; config publishes {sorted(declared)}." if declared else ".")
        )

    from_native = [v for v in wanted if v in stored_set]
    from_compiled = [v for v in wanted if v not in stored_set]

    if from_compiled and step_freq(var_config) != "h":
        raise ValueError(
            f"[{var_key}] daily store holds {sorted(stored)} but this variable "
            f"publishes {sorted(declared)}. Absent: {sorted(from_compiled)}. A daily "
            f"store is written with everything it publishes, so this is a gap in "
            f"the store — re-run `uv run h2mare convert -v {var_key}`."
        )

    return from_native, from_compiled


def resolve_compiled_vars(
    available: list[str],
    requested: list[str] | None,
    var_key: str,
    var_config,
) -> list[str]:
    """
    The compiled-h2ds columns that belong to *var_key*.

    ``compiled_vars`` is already the var_key -> h2ds-column mapping (it is what
    ``h2mare parquet --add-var`` selects on), so it is reused here rather than
    re-derived.

    Raises:
        ValueError: if the var_key publishes nothing, or if h2ds does not yet
            hold a requested column — which means compile is behind convert, not
            that the request was wrong.
    """
    declared = _declared_vars(var_config)
    wanted = list(requested) if requested is not None else declared

    if not wanted:
        raise ValueError(
            f"[{var_key}] declares no compiled_vars, so there is no mapping onto "
            f"h2ds columns. Add compiled_vars to its config entry."
        )

    missing = sorted(v for v in wanted if v not in available)
    if missing:
        raise ValueError(
            f"[{var_key}] the compiled h2ds is missing {missing}. These are "
            f"derived at compile time from the hourly store, so compile is "
            f"behind convert — run `uv run h2mare compile`. h2ds holds: "
            f"{sorted(v for v in declared if v in available)}."
        )

    return wanted


@log_time
def load_dataset_to_memory(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    return ds.compute()


def ensure_row_id(
    data: pd.DataFrame | gpd.GeoDataFrame, col: str = "row_id"
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Guarantee a stable, unique key column on ``data`` for merging extraction
    results back onto the caller's dataframe.

    The merge key must exist on *both* sides of the eventual join, so it has to
    be established here — on the frame the caller keeps — rather than invented
    inside :class:`Extractor`, which never sees the caller's other columns. Use
    the returned frame for both the extraction and the later merge.

    Behaviour:
        - ``col`` present and unique     -> returned unchanged.
        - ``col`` present with duplicates -> ``ValueError`` (a duplicated key
          silently collapses rows on merge-back; the caller must fix it).
        - ``col`` absent                 -> a positional ``range(len(data))`` key
          is added on a copy.

    Parameters:
        data (pd.DataFrame | gpd.GeoDataFrame): input points or geometries.
        col (str): name of the key column. Defaults to ``"row_id"``.

    Returns:
        The frame (same type as ``data``) carrying a unique ``col``.
    """
    if col in data.columns:
        if data[col].duplicated().any():
            n = int(data[col].duplicated().sum())
            raise ValueError(
                f"'{col}' has {n} duplicate value(s); the key must be unique to "
                "merge extraction results back without collapsing rows."
            )
        return data

    data = data.copy()
    data[col] = range(len(data))
    logger.info(
        f"No '{col}' column found — added a positional one (0..{len(data) - 1})."
    )
    return data


class Extractor:
    """
    Extracts values at points or geometries from the stores, for analysis.

    A standalone tool, not part of the run/convert/compile flow. Takes a CSV,
    shapefile, GeoDataFrame or DataFrame of locations and returns one row per
    input row with a column per requested variable.

    Two independent arguments decide where each value comes from:
    ``time_cadence`` picks how the input's time column is read, and
    ``read_from`` picks the store. Under ``auto`` a daily store answers for
    itself, while an hourly one answers only sub-daily input — a date-only
    query against it goes to the compiled ``h2ds``, which therefore needs to be
    current. Units can differ between the two sources (``msl`` is Pa natively,
    hPa in h2ds). See ``docs/api/extractor.md#cadence``.

    A failed run leaves a checkpoint that a later run resumes from, keyed by a
    fingerprint of the input.
    """

    def __init__(
        self,
        file_path: Union[Path, gpd.GeoDataFrame, pd.DataFrame],
        *,
        index_col: str,
        time_col: Optional[str] = None,
        lon_col: Optional[str] = None,
        lat_col: Optional[str] = None,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Union[str, Path]] = None,
        crs: int | None = 4326,
        time_cadence: TimeCadence = "auto",
        read_from: ReadFrom = "auto",
        log_file: Optional[Union[str, Path]] = None,
    ):
        """
        Extract data from shp/csv file_path of open file

        Parameters:
            file_path (Union[Path, gpd.GeoDataFrame, pd.DataFrame]): Data for extraction
            index_col (str): Name of the unique key column used to merge results
                back onto the input. Required and must already exist in the data
                (establish it with :func:`ensure_row_id`); a missing or duplicated
                key raises ``ValueError``.
            time_col (str): Name of time column. Defaults to "time".
            lon_col (str, optional): Name of longitude column. Defaults to "lon".
            lat_col (str, optional): Name of latitude column. Defaults to "lat".
            app_config (AppConfig, optional): Dataclass with environmental data specifics. Defaults to cfg.
            store_root (Union[str, Path], optional): Path for environmental data main folder. Defaults to STORE_ROOT.
            crs (int | None, optional): Projection EPSG code for geometry extraction. Defaults to 4326.
            time_cadence ("auto" | "daily" | "hourly"): how ``time_col`` is read.
                ``"auto"`` (default) infers it: a time component that *varies*
                across rows means the caller wants hours; a date-only input, or
                one stamped identically on every row (an export default rather
                than a real hour), means days. ``"daily"`` truncates to midnight
                regardless; ``"hourly"`` keeps whatever precision is there. This
                only decides how the input is parsed — which store answers is
                ``read_from``.
            read_from ("auto" | "native" | "compiled"): which store each var_key
                is read from. ``"native"`` is its own per-variable Zarr,
                ``"compiled"`` is the h2ds every var_key is merged into, and
                ``"auto"`` (default) picks per var_key: a daily store answers for
                itself, while an hourly one answers only a sub-daily request and
                otherwise defers to the compiled store, which is where its daily
                values live. Note the two are not interchangeable — the compiled
                store is on the 0.25° base grid and carries the pipeline's units
                (ERA5 ``msl`` in hPa), while an hourly native store holds the raw
                source as published (``msl`` in Pa).
            log_file (str | Path, optional): Extraction log file for this session.
                Defaults to LOGS_DIR/extractor.log (first Extractor in the
                process decides; subsequent values are ignored).

        """
        configure_extraction_logging(log_path=log_file)

        self.time_col = time_col if time_col is not None else "time"
        self.index_col = index_col
        self.lon_col = lon_col if lon_col is not None else "lon"
        self.lat_col = lat_col if lat_col is not None else "lat"
        self.crs = crs
        self.time_cadence: TimeCadence = time_cadence
        self.read_from: ReadFrom = read_from

        self.app_config = app_config or get_settings().app_config

        self.store_root = (
            Path(store_root) if store_root is not None else get_settings().STORE_ROOT
        )

        data_orig = self._resolve_file_format(file_path)
        data_orig = self._resolve_index(data_orig)
        self.data = self._prepare_data(data_orig)
        self.data = self._resolve_time_col(self.data)

    def _store_dir(self, var_config) -> Path:
        """
        This variable's own store directory under the root that applies to it.

        ``self.store_root`` is a *root* holding one folder per variable, the
        shape ``STORE_ROOT`` has in ``.env``, so it cannot be handed to a
        catalog as-is — ``ZarrCatalog(store_root=...)`` means one store's exact
        directory. It is also only the *default*: a variable naming its own
        ``store_root`` in config.yaml is read from there, and ``--store-path``
        outranks both. Same rule as ``PipelineManager._store_dir`` and
        ``Compiler._catalog_for``.
        """
        return store_root_for(var_config, self.store_root) / var_config.local_folder

    # =================== DATA PREPARATION =====================

    def _resolve_time_col(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve time column to date or datetime based on time variance.

        Logic (``time_cadence="auto"``):
            - If time_col strings contain no time component → keep as date.
            - If time_col contains datetimes:
                - If time component is identical across all rows → truncate to date.
                - If time component varies → keep full datetime.

        ``"daily"`` always truncates; ``"hourly"`` never does, so a uniform
        stamp is honoured as a real hour rather than read as a nominal one.

        The verdict is kept on ``self.input_is_subdaily`` rather than thrown
        away: with ``read_from="auto"`` it is what decides whether an hourly
        var_key is served from its own store or the compiled one
        (:func:`resolve_read_from`). Nothing else needs a second time column —
        the branch below already leaves ``time`` at full precision exactly when
        the sub-daily route wants it, and at midnight when the daily route does.
        """
        data = data.rename(columns={self.time_col: "time"})

        # Check on raw strings BEFORE parsing — avoids 00:00:00 false negative
        raw = data["time"].astype(str)
        has_time_component = raw.str.contains(r"\d{2}:\d{2}:\d{2}", regex=True).any()

        data["time"] = pd.to_datetime(data["time"], utc=True).dt.tz_convert(None)

        if self.time_cadence == "daily":
            subdaily = False
        elif self.time_cadence == "hourly":
            subdaily = bool(has_time_component)
        elif has_time_component:
            # A stamp identical on every row reads as nominal (someone's export
            # default), not as a deliberate hour — hence uniform means daily.
            subdaily = data["time"].dt.time.nunique() > 1
        else:
            subdaily = False

        if subdaily:
            logger.debug("Sub-daily input detected. Keeping full datetime.")
        else:
            logger.debug("Daily input detected. Truncating to date.")
            data["time"] = data["time"].dt.normalize()

        self.input_is_subdaily = subdaily
        return data

    def _resolve_file_format(
        self, file_path: Union[Path, gpd.GeoDataFrame, pd.DataFrame]
    ):
        """determine input type and load accordingly"""

        if isinstance(file_path, gpd.GeoDataFrame):
            data_base = file_path.copy()
            self.input_type = "shp"
            self.input_label = "<in-memory GeoDataFrame>"

        elif isinstance(file_path, pd.DataFrame):
            data_base = file_path.copy()
            self.input_type = "csv"
            self.input_label = "<in-memory DataFrame>"

        else:
            file_path = Path(file_path)
            suffix = file_path.suffix.lower()
            self.input_label = file_path.name

            if suffix == ".shp":
                data_base = gpd.read_file(file_path)
                self.input_type = "shp"

            elif suffix == ".csv":
                data_base = pd.read_csv(file_path)
                self.input_type = "csv"
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

        return data_base

    def _resolve_index(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Set ``index_col`` as the frame index — the key used to merge results
        back onto the caller's data.

        The key is the caller's responsibility: it must already exist in the
        data and be unique (establish it up front with :func:`ensure_row_id`).
        The Extractor consumes the key, it never creates one.

        Raises:
            ValueError: if ``index_col`` is missing from the data, or has
                duplicate values.
        """
        if self.index_col not in data.columns:
            raise ValueError(
                f"index_col '{self.index_col}' not found in data — establish it "
                "first, e.g. with ensure_row_id(data)."
            )
        if data[self.index_col].duplicated().any():
            n = int(data[self.index_col].duplicated().sum())
            raise ValueError(
                f"index_col '{self.index_col}' has {n} duplicate value(s); it must "
                "be unique to merge extraction results back without collapsing rows."
            )
        return data.set_index(self.index_col)

    def _prepare_data(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Prepares data according to input_type and returns a subseted df/gdf with only essential cols for extraction.
            - csv: df['time', 'lon', 'lat']
            - shp: gdf['time', 'geometry']

        """
        cols = {self.time_col: "time", self.lon_col: "lon", self.lat_col: "lat"}

        if self.time_col not in data.columns:
            raise ValueError(
                f"Time column '{self.time_col}' not found in data attributes."
            )

        if self.input_type == "csv":
            if self.lon_col not in data.columns or self.lat_col not in data.columns:
                raise ValueError(
                    f"CSV must contain '{self.lon_col}' and '{self.lat_col}' columns."
                )

            data = data.rename(columns=cols)[["time", "lon", "lat"]].copy()

        elif self.input_type == "shp":
            if self.crs is None:
                raise ValueError("CRS must be provided for shapefile input.")

            if isinstance(data, gpd.GeoDataFrame):
                data = data.copy()
                if data.crs is None:
                    data.set_crs(self.crs, inplace=True)
                elif data.crs.to_epsg() != self.crs:
                    data = data.to_crs(self.crs)
            data = data.rename(columns=cols)[["time", "geometry"]].copy()

        else:
            raise ValueError(
                f"Invalid input_type: {self.input_type}. Must be 'csv' or 'shp'."
            )

        return data

    def _define_bbox(self, data: pd.DataFrame | gpd.GeoDataFrame) -> BBox:
        """
        Returns bbox from data according to input_type ('csv' or 'shp').

        An extent with no width — every row on one point, or strung along one
        parallel or meridian — is widened rather than refused. That shape is
        ordinary input: a single sample, a transect, or whatever is left after
        the coverage clip drops the rest. ``BBox`` goes on rejecting it, because
        for a *store's* own extent a single cell really is an error; here it is
        only a query, and the read pads it by a grid cell on each side anyway.
        """
        if self.input_type == "csv":
            bounds = (
                data["lon"].min(),
                data["lat"].min(),
                data["lon"].max(),
                data["lat"].max(),
            )

        elif self.input_type == "shp":
            bounds = tuple(data.total_bounds)

        else:
            raise ValueError(f"Unsupported input_type: {self.input_type!r}")

        return BBox.from_tuple(_widen_degenerate(bounds))

    def _resolve_coverage(
        self, catalog: ZarrCatalog, source_key: str | None = None
    ) -> list[pd.Timestamp]:
        """
        Resolve input/store data space/time coverage limits.

        ``source_key`` names the var_key whose columns are being read out of
        *catalog*, and narrows the time side to what that var_key actually has
        there. It matters only on the compiled store, where every source is
        padded out to the union axis by ``xr.merge``: the store ends where its
        furthest-ahead source ends, and clipping against that lets every date in
        a slower source's padding through, to be extracted as NaN, silently. A
        var_key's compiled columns all come from one source and so share one
        frontier, so the first one it publishes dates the group — the same
        convention the compiler's own watermarks use.

        Args:
            catalog: Store to clip against.
            source_key: var_key whose columns are being read. ``None`` uses the
                store's own coverage, which is right for a native store — its
                files end where its data does.

        Raises:
            ValueError: if the coverage lookup or ``get_bbox()`` returns None

        Returns:
            list[pd.Timestamp]: List of unique dates within store limits if out of range, else returns None.
        """
        published = (
            list(self.app_config.variables[source_key].compiled_vars or [])
            if source_key is not None
            else []
        )
        # get storage coverage
        store_dates = (
            catalog.get_var_coverage(published[0])
            if published
            else catalog.get_time_coverage()
        )
        store_bbox = catalog.get_bbox()

        if store_dates is None or store_bbox is None:
            raise ValueError(f"No coverage data for {catalog.var_key}")

        # `end` names a calendar day, so the window runs to the *end* of that
        # day. Compared bare against a full input timestamp it clips every
        # sample stamped after midnight on the final covered day — 23 hours'
        # worth against an hourly store, reported as "after store coverage".
        # Every other date-bounded read pairs the bound with end_of_day; this
        # one is the last that didn't.
        start_store = pd.Timestamp(store_dates.start)
        end_store = end_of_day(store_dates.end)

        # Input Data coverage
        dates = self._extract_unique_dates(self.data)
        start, end = dates.min(), dates.max()
        bounds = self._define_bbox(self.data)

        if not bounds.overlaps(store_bbox):
            logger.warning(
                f"Data input bbox does not overlap with store data for {catalog.var_key}"
            )

        start_date = pd.to_datetime(max(start, start_store))
        end_date = pd.to_datetime(min(end, end_store))

        # Name what was actually measured. Reporting these as the store's bounds
        # would have the compiled store "ending" on a different day for every
        # var_key routed through it.
        label = f"{source_key} in {catalog.var_key}" if published else catalog.var_key
        scope = "variable" if published else "store"

        if start < start_store:
            clipped = dates[dates < start_store]
            logger.warning(
                f"{label}: {len(clipped)} date(s) before {scope} coverage clipped "
                f"({clipped.min().date()} -> {clipped.max().date()} | {scope} starts {start_store.date()})"
            )
        if end > end_store:
            clipped = dates[dates > end_store]
            logger.warning(
                f"{label}: {len(clipped)} date(s) after {scope} coverage clipped "
                f"({clipped.min().date()} -> {clipped.max().date()} | {scope} ends {end_store.date()})"
            )

        return sorted(dates[(dates >= start_date) & (dates <= end_date)])

    def _extract_unique_dates(
        self, data: gpd.GeoDataFrame | pd.DataFrame
    ) -> pd.DatetimeIndex:
        """Extract unique dates from the GeoDataFrame's time column."""
        if "time" not in data.columns:
            raise ValueError("Time column 'time' not found in shapefile attributes.")
        return pd.DatetimeIndex(pd.to_datetime(data["time"])).drop_duplicates()

    # ===================  PROCESS DATA ===================

    def process_single_varkey(
        self, var_key: str, vars: VarSelection = None, n_workers: int = 8
    ) -> pd.DataFrame:
        """
        Run extraction process for a single var_key.

        Parameters:
            var_key : str
                Key to identify variable in config.
            vars : str, list[str], dict[str, list[int] | None], None
                Specific variables for extraction associated with the specified var_key. This avoids extracting all vars inside the var_key.
                A dict names the variables as its keys and, where a value is given,
                the depth levels (metres) to slice that 3-D variable at for this
                request, instead of the ones in config.
            n_workers : int, optional
                Number of parallel workers for geometries (shp) extraction, by default 8.

        Returns:
            pd.DataFrame with extracted values.
        """
        vars, depth_override = split_depth_request(var_key, vars)
        vars = [vars] if isinstance(vars, str) else vars

        # An empty list is the documented way to say "everything this var_key
        # publishes" — `run({"seapodym": [], "radiation": ["tisr"]})` — not an
        # explicit selection of nothing. Collapsed to None here so every
        # helper downstream sees one sentinel for "all" instead of each having
        # to remember there are two.
        if not vars:
            vars = None

        if depth_override and var_key in ("moon", "bathy"):
            raise ValueError(
                f"[{var_key}] has no depth axis; depth levels cannot be chosen "
                f"for it ({depth_override})."
            )

        # Moon and bathy first since they do not need data from ZarCatalog
        if var_key == "moon":
            return self._extract_moon_phase(self.data)

        if var_key == "bathy":
            return self._extract_bathy(self.data)

        var_cfg = self.app_config.variables[var_key]
        source = resolve_read_from(
            var_cfg,
            read_from=self.read_from,
            subdaily_input=self.input_is_subdaily,
        )

        if source == "compiled" and depth_override:
            raise ValueError(
                f"[{var_key}] depth levels {depth_override} can only be chosen "
                f"when reading its own store, and this request is answered from "
                f"the compiled store, which holds the fixed columns compile "
                f"published. Name those columns instead, or pass "
                f"read_from='native'."
            )

        if source == "compiled":
            # Date-only query against an hourly var_key: the daily numbers it
            # publishes live in the compiled store, not in its own.
            return self._extract_compiled(var_key, vars, var_cfg, n_workers)

        vr_catalog = ZarrCatalog(
            var_key, app_config=self.app_config, store_root=self._store_dir(var_cfg)
        )
        dates_resolved = self._resolve_coverage(vr_catalog)
        data_resolved = self._subset_to_coverage(dates_resolved)
        bounds = self._define_bbox(data_resolved)

        logger.info(f"Extracting {var_key} data from {vr_catalog.store_root}")
        logger.info(
            f"{data_resolved.shape[0]} samples | "
            f"{min(dates_resolved).date()} -> {max(dates_resolved).date()} | "
            f"{bounds}"
        )

        ds = vr_catalog.open_dataset(dates=dates_resolved, bbox=bounds)

        warn_on_subdaily_store(var_key, var_cfg, ds)
        has_depth = "depth" in ds.dims
        if depth_override and not has_depth:
            raise ValueError(
                f"[{var_key}] depth levels were given for "
                f"{sorted(depth_override)}, but its store has no depth axis."
            )
        from_native, from_compiled = split_vars_by_source(
            vars,
            [str(v) for v in ds.data_vars],
            var_key,
            var_cfg,
            has_depth=has_depth,
        )

        # Gated on the store's own dims rather than on the config key: a depth
        # axis left in place is not an error, it is silently averaged away by
        # the geometry engine's dimensionless .mean().
        if has_depth:
            ds = self._slice_depth(
                ds, var_key, var_cfg, from_native or None, depth_override
            )
        elif from_native:
            ds = ds[from_native]

        ds = ds.sortby("time")

        result = self._extract(data_resolved, ds, n_workers)

        if from_compiled:
            # Reached only when this var_key converts hourly, so its derived
            # features were never written to its own store (converting the same
            # variable daily computes them up front and this branch is dead).
            # They are daily by construction — a 7-day rolling mean against a
            # day-of-year climatology has no hourly value to give — so each
            # sample takes the value for the day it falls in.
            logger.warning(
                f"[{var_key}] {sorted(from_compiled)} are not in the native "
                f"store: this variable converts hourly, so they are derived at "
                f"compile time. Reading them from the compiled store and "
                f"broadcasting each day's value across that day's samples."
            )
            daily = self._extract_compiled(var_key, from_compiled, var_cfg, n_workers)
            # Both engines carry the coordinate columns out of to_dataframe();
            # keeping them on both sides would collide on join. _run_impl strips
            # them from the final frame anyway, so the store side's copy stands.
            result = result.join(
                daily.drop(columns=_COORD_COLS, errors="ignore"), how="left"
            )

        return result

    def _subset_to_coverage(self, dates_resolved: list[pd.Timestamp]):
        """Rows of the input that fall inside the resolved coverage window."""
        mask = self.data["time"].between(min(dates_resolved), max(dates_resolved))
        return self.data.loc[mask]

    def _extract(
        self,
        data_resolved,
        ds: xr.Dataset,
        n_workers: int,
    ) -> pd.DataFrame:
        """Run the point or geometry engine, whichever this input calls for."""
        if self.input_type == "shp":
            if not isinstance(data_resolved, gpd.GeoDataFrame):
                raise TypeError("Data must be a GeoDataFrame for shapefile extraction")

            # Unconditional, as extract_from_dataset already does and as
            # extract_from_shp documents it needs: rio.clip resolves dims by
            # name, and only falls back to lon/lat when they carry CF
            # attributes. CMEMS and AVISO stores inherit those from source, so
            # every var_key but fsle/eddies passed the precondition by luck;
            # CDS stores and the compiled h2ds carry no coordinate attributes
            # at all, and clipped to nothing but NaN.
            rename = {
                old: new for old, new in (("lon", "x"), ("lat", "y")) if old in ds.dims
            }
            if rename:
                ds = ds.rename(rename)

            # Strictly before ensure_crs, as _extract_bathy and
            # extract_from_dataset already are. write_crs names the grid-mapping
            # coordinate after the one the variables' `grid_mapping` attribute
            # already points at — but it can only find that attribute by walking
            # variables that have resolvable spatial dims. On a store whose
            # lon/lat carry no CF attributes (the compiled h2ds), that walk finds
            # nothing before the rename, so the CRS lands on the default
            # `spatial_ref` while CMEMS/AVISO variables keep pointing at the
            # `crs` coordinate they inherited from source and that h2ds never
            # carried. The dataset then reports a CRS while every variable in it
            # reports none, and rio.clip raises MissingCRS per geometry — an
            # all-NaN column for adt/sla/ugos/vgos/fsle_max under
            # read_from="compiled", while variables without the attribute
            # extracted fine.
            ds = self.ensure_crs(data_resolved, ds)

            return self.extract_from_shp(
                data_resolved, ds, self.index_col, n_workers=n_workers
            )

        elif self.input_type == "csv":
            return self.extract_from_csv(data_resolved, ds, self.index_col)

        raise ValueError(f"Unsupported input_type: {self.input_type}")

    @cached_property
    def _compiled_var_key(self) -> str:
        """
        The var_key holding the compiled dataset, found by its ``source``.

        Identified the same way :meth:`_normalize_var_dict` excludes it from
        default runs — by ``source: h2mare`` rather than by the name ``h2ds`` —
        so a deployment that names its compiled store differently still routes.
        Cached here because ``run()`` can route several var_keys through it.
        """
        return compiled_var_key(self.app_config)

    @cached_property
    def _compiled_catalog(self) -> ZarrCatalog:
        """
        The compiled daily store, opened once per Extractor.

        ``run()`` can route several var_keys here, so the catalog (and its index
        scan) is cached rather than rebuilt per var_key.
        """
        var_cfg = self.app_config.variables[self._compiled_var_key]
        return ZarrCatalog(
            self._compiled_var_key,
            app_config=self.app_config,
            store_root=self._store_dir(var_cfg),
        )

    def _extract_compiled(
        self,
        var_key: str,
        vars: list[str] | None,
        var_cfg,
        n_workers: int,
    ) -> pd.DataFrame:
        """
        Extract *var_key*'s columns from the compiled daily store.

        Coverage is resolved against the compiled store rather than the
        per-variable one: the two do not move together, since each source lags
        its provider differently and compile trails convert. Asking the native
        store what is available would over-promise.

        And asking the compiled store *as a whole* would over-promise the other
        way. It reports the union of everything merged into it, so a var_key
        whose source lags the furthest-ahead one is padded with NaN to that end
        and reads as covered there. Coverage is therefore asked per variable,
        keyed on the first column this var_key publishes — they all come from
        one source and share one frontier.

        Sample times are normalised to the day, because the compiled store is
        daily and the caller may arrive here holding sub-daily stamps — either
        via the broadcast path or by pinning ``read_from="compiled"``. Each
        sample then takes the value for the day it falls in.

        Depth slicing is skipped here: the compiled store
        is lon/lat on the base grid, and it already holds depth levels as
        separate variables (``o2_0``, ``o2_100``, …) rather than on a ``depth``
        axis.
        """
        catalog = self._compiled_catalog
        dates_resolved = self._resolve_coverage(catalog, var_key)
        data_resolved = self._subset_to_coverage(dates_resolved).copy()

        # Only for a pinned read_from: on the broadcast path the caller has
        # already been told, in terms of the specific variables involved.
        if self.input_is_subdaily and self.read_from == "compiled":
            logger.warning(
                f"[{var_key}] read_from='compiled' and the compiled store is "
                f"daily: sub-daily samples take the value for the day they fall "
                f"in, not the value for their hour."
            )
        data_resolved["time"] = data_resolved["time"].dt.normalize()
        bounds = self._define_bbox(data_resolved)

        logger.info(
            f"Extracting {var_key} data from the compiled "
            f"{self._compiled_var_key} at {catalog.store_root}"
        )
        logger.info(
            f"{data_resolved.shape[0]} samples | "
            f"{min(dates_resolved).date()} -> {max(dates_resolved).date()} | "
            f"{bounds}"
        )

        ds = catalog.open_dataset(
            dates=sorted({pd.Timestamp(d).normalize() for d in dates_resolved}),
            bbox=bounds,
        )
        wanted = resolve_compiled_vars(
            [str(v) for v in ds.data_vars], vars, var_key, var_cfg
        )

        ds = ds[wanted].sortby("time")
        return self._extract(data_resolved, ds, n_workers=n_workers)

    def extract_from_dataset(
        self,
        ds: xr.Dataset | xr.DataArray,
        *,
        vars: str | list[str] | None = None,
        n_workers: int = 8,
        clip_to_coverage: bool = False,
    ) -> pd.DataFrame:
        """
        Extract values from an arbitrary in-memory dataset, bypassing ZarrCatalog.

        This is the config-free counterpart to :meth:`process_single_varkey`: it runs
        the same extraction engine (:meth:`extract_from_csv` / :meth:`extract_from_shp`)
        against a ``ds`` the caller already holds in memory — useful for new data that
        is not yet ingested into the store. The prepared points/geometries in
        ``self.data`` are reused as-is.

        Only the config-free prep is applied here. Config-driven steps that
        :meth:`process_single_varkey` performs — depth-slice expansion, store
        selection (``read_from``) and store date/bbox coverage
        resolution — are the caller's responsibility: prepare ``ds`` beforehand.
        A ``ds`` handed over with a ``depth`` axis still on it is extracted as-is,
        which for geometry input means that axis is averaged away along with the
        spatial one; slice it yourself first.

        Parameters:
            ds (xr.Dataset | xr.DataArray): gridded data with coords ``lon``, ``lat``
                and optionally ``time``. For shapefile input it is assumed to be in
                ``self.crs`` (its CRS is overwritten, not reprojected, to match the
                geometries — see :meth:`ensure_crs`).
            vars (str | list[str] | None): subset of variables to extract. Only valid
                when ``ds`` is an ``xr.Dataset``; raises ``TypeError`` for a DataArray.
            n_workers (int): parallel workers for shapefile (geometry) extraction.
            clip_to_coverage (bool): when True, drop input rows whose location (and
                time, if ``ds`` has a time coord) falls outside the ``ds`` extent;
                dropped rows surface as NaN in the result. Defaults to False, since
                nearest-neighbour (csv) / clip-or-NaN (shp) already handle them.

        Returns:
            pd.DataFrame indexed by ``self.index_col`` (aligned to ``self.data.index``)
            with one column per variable (csv) or ``var`` / ``var_std`` columns (shp).
        """
        vars = [vars] if isinstance(vars, str) else vars

        if vars is not None:
            if not isinstance(ds, xr.Dataset):
                raise TypeError(
                    "`vars` can only be used when `ds` is an xr.Dataset, "
                    "not an xr.DataArray."
                )
            ds = ds[vars]

        if "time" in ds.coords:
            ds = ds.sortby("time")

        data = self._mask_to_ds_extent(ds) if clip_to_coverage else self.data

        if self.input_type == "shp":
            if not isinstance(data, gpd.GeoDataFrame):
                raise TypeError("Data must be a GeoDataFrame for shapefile extraction")

            # rioxarray's .rio.clip (used by extract_from_shp) resolves spatial dims
            # by name and expects x/y. Geographic datasets carry lon/lat, so rename —
            # the config-driven equivalent of var_cfg.rename_lonlat in the store path.
            if "lon" in ds.coords and "lat" in ds.coords:
                ds = ds.rename({"lon": "x", "lat": "y"})

            ds = self.ensure_crs(data, ds)
            result = self.extract_from_shp(
                data, ds, self.index_col, n_workers=n_workers
            )

        elif self.input_type == "csv":
            result = self.extract_from_csv(data, ds, self.index_col)

        else:
            raise ValueError(f"Unsupported input_type: {self.input_type}")

        # When rows were clipped out, realign to the full input so dropped rows
        # surface as NaN rather than silently vanishing from the result.
        if clip_to_coverage:
            result = result.reindex(self.data.index)

        return result

    def _mask_to_ds_extent(
        self, ds: xr.Dataset | xr.DataArray
    ) -> pd.DataFrame | gpd.GeoDataFrame:
        """
        Return a copy of ``self.data`` keeping only rows inside the ``ds`` extent.

        Spatial filtering uses the dataset bbox (point for csv, geometry centroid for
        shp); temporal filtering applies only when ``ds`` carries a ``time`` coord.
        ``self.data`` itself is never mutated.
        """
        bbox = BBox.from_dataset(ds)

        if self.input_type == "csv":
            mask = self.data["lon"].between(bbox.xmin, bbox.xmax) & self.data[
                "lat"
            ].between(bbox.ymin, bbox.ymax)
        else:
            centroids = self.data.geometry.centroid
            mask = centroids.x.between(bbox.xmin, bbox.xmax) & centroids.y.between(
                bbox.ymin, bbox.ymax
            )

        if "time" in ds.coords:
            dr = DateRange.from_dataset(ds)
            mask &= self.data["time"].between(dr.start, dr.end)

        return self.data.loc[mask]

    @overload
    def run(
        self,
        var_dict: Optional[Union[str, list[str], Mapping[str, VarSelection]]] = ...,
        output_path: None = ...,
        n_workers: int = ...,
    ) -> pd.DataFrame: ...

    @overload
    def run(
        self,
        var_dict: Optional[Union[str, list[str], Mapping[str, VarSelection]]] = ...,
        output_path: str | Path = ...,
        n_workers: int = ...,
    ) -> None: ...

    def run(
        self,
        var_dict: Optional[Union[str, list[str], Mapping[str, VarSelection]]] = None,
        output_path: Optional[str | Path] = None,
        n_workers: int = 8,
    ) -> pd.DataFrame | None:
        """
        Extract all or specified var_key and respective variables, and save dataframe with extracted data.

        Args:
            var_dict (str | list[str] | Mapping[str, VarSelection] | None, optional: Var_key str or list of strings or dict specifiying vars in var_key.
                Defaults to None, extracting all available var_keys and respective variables.
                A ``{variable: depths | None}`` value also chooses the depth levels
                (metres) of 3-D variables for this run, over those in config.
            output_path (str | Path | None): Path to save file. If None, it returns a dataframe with all results.
            n_workers (int, optional): Workers for shp parallel processing. Defaults to 8.

        Example:
            >>> var_dict = {
            >>>     'seapodym': [],
            >>>     'radiation': ['tisr', 'ssrd', 'slhf'],
            >>>     'dyn_rep': {'thetao': [0, 50], 'zos': None},
            >>>     }
            >>>
            >>> extractor = Extractor(file_path=input_path, time_col='date', index_col='id_row')
            >>> results = extractor.run(var_dict, output_path=output_path, n_workers=12)
        """
        t0 = time.perf_counter()
        # job="extract" routes every message in this scope (including from the
        # storage layer) to the extraction sink; see configure_extraction_logging.
        with logger.contextualize(job="extract"):
            out_label = str(output_path) if output_path is not None else "DataFrame"
            logger.info(
                f"Extraction started: input={self.input_label} "
                f"({self.data.shape[0]} rows, {self.input_type}) → output={out_label}"
            )

            df_processed, all_succeeded = self._run_impl(var_dict, n_workers)

            if output_path is not None:
                self._save_results(df_processed, Path(output_path))

            n_new = sum(1 for c in df_processed.columns if c not in self.data.columns)
            outcome = (
                f"input={self.input_label} → output={out_label}, "
                f"{len(df_processed)} rows × {n_new} new column(s) "
                f"in {time.perf_counter() - t0:.1f}s"
            )
            if all_succeeded:
                logger.success(f"Extraction complete: {outcome}")
            else:
                logger.warning(
                    f"Extraction finished with errors: {outcome} — "
                    "checkpoint preserved for resume."
                )

        if output_path is not None:
            return None
        return df_processed

    def _run_impl(
        self,
        var_dict: Optional[Union[str, list[str], Mapping[str, VarSelection]]],
        n_workers: int,
    ) -> tuple[pd.DataFrame, bool]:
        """Extraction loop body; returns (results, all_succeeded)."""
        # n_workers only drives the ThreadPoolExecutor in shp (geometry) extraction;
        # csv (point) extraction is vectorized and ignores it — so don't advertise it there.
        if self.input_type == "shp":
            logger.info(f"Starting extraction process with {n_workers} workers.")
        else:
            logger.info("Starting extraction process.")

        var_dict = self._normalize_var_dict(var_dict)

        tmp_path = get_settings().INTERIM_DIR / "extraction_checkpoint.feather"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        fingerprint = input_fingerprint(self.data, self.index_col)
        completed_keys = (
            _load_completed_keys(tmp_path, fingerprint) if tmp_path.exists() else None
        )

        if completed_keys is not None:
            logger.warning(
                f"Found checkpoint for this input: {tmp_path}, resuming. "
                f"Already done, and replayed rather than re-extracted: "
                f"{sorted(completed_keys) or 'nothing yet'}."
            )
            df_processed = pd.read_feather(tmp_path).set_index(self.index_col)
            if df_processed.index.duplicated().any():
                logger.warning(
                    "Duplicate index values found in checkpoint — keeping first occurrence."
                )
                df_processed = df_processed[
                    ~df_processed.index.duplicated(keep="first")
                ]
            # Feather can't round-trip live shapely geometries, so pd.read_feather
            # brings `geometry` back as WKB bytes. Restore it from the original
            # (index-aligned) input and re-wrap as a GeoDataFrame, matching the
            # fresh-run return type.
            if self.input_type == "shp":
                df_processed = gpd.GeoDataFrame(
                    df_processed.drop(columns="geometry", errors="ignore"),
                    geometry=self.data.geometry.reindex(df_processed.index),
                    crs=self.data.crs,
                )
        else:
            df_processed = self.data.copy()
            completed_keys = set()

        all_succeeded = True

        for var_key, vars_ in var_dict.items():
            if var_key in completed_keys:
                logger.info(f"Skipping {var_key}: already extracted.")
                continue

            t0 = time.perf_counter()
            try:
                with logger.contextualize(var=var_key):
                    result = self.process_single_varkey(
                        var_key=var_key, vars=vars_, n_workers=n_workers
                    )

                    result.drop(
                        columns=_COORD_COLS,
                        errors="ignore",
                        inplace=True,
                    )
                    if result.index.duplicated().any():
                        logger.warning(
                            f"Duplicate index values in '{var_key}' result — keeping first occurrence."
                        )
                        result = result[~result.index.duplicated(keep="first")]

                    # A kill between the two checkpoint writes below leaves the
                    # feather holding these columns while the sidecar still omits
                    # the key, so the resume re-extracts and joins onto columns
                    # already there — which pandas rejects, and the failure keeps
                    # the checkpoint from being cleared, so every later run fails
                    # identically. Dropping the stale copy makes re-extraction
                    # idempotent. The caller's own input columns are left to
                    # collide: overwriting those silently would be worse.
                    stale = [
                        c
                        for c in result.columns
                        if c in df_processed.columns and c not in self.data.columns
                    ]
                    if stale:
                        logger.warning(
                            f"'{var_key}' is already in the checkpoint but was not "
                            f"recorded as done — the previous run was interrupted "
                            f"mid-checkpoint. Replacing {stale} with the fresh values."
                        )
                        df_processed = df_processed.drop(columns=stale)

                    df_processed = df_processed.join(result)

                    # Data first, then the key claiming it is there. Reversed, a
                    # kill here would mark the var_key done with its columns
                    # missing and the resume would skip it. This way the worst
                    # case is repeated work, which the drop above absorbs.
                    completed_keys.add(var_key)
                    staging = tmp_path.with_suffix(".tmp")
                    df_processed.reset_index().to_feather(staging)
                    staging.replace(tmp_path)
                    _save_completed_keys(tmp_path, completed_keys, fingerprint)
                    logger.debug(f"Checkpoint saved to {tmp_path}")

                    logger.success(
                        f"{var_key}: {len(result)} row(s), "
                        f"{result.shape[1]} column(s) "
                        f"in {time.perf_counter() - t0:.1f}s"
                    )

            except Exception as e:
                logger.opt(exception=True).error(f"Error processing '{var_key}': {e}")
                all_succeeded = False
                continue

        logger.info("=" * 60)
        logger.info("  Number of null values per variable:")
        result_cols = [c for c in df_processed.columns if c not in self.data.columns]
        for line in null_summary_lines(df_processed, result_cols):
            logger.info(line)
        logger.info("=" * 60)

        if all_succeeded:
            tmp_path.unlink(missing_ok=True)
            _keys_path(tmp_path).unlink(missing_ok=True)

        return df_processed, all_succeeded

    @staticmethod
    def _nearest_grid_indices(
        ds: xr.Dataset | xr.DataArray,
        query_lons: np.ndarray,
        query_lats: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat_idx, lon_idx) arrays for each query point using a KDTree.

        The tree is cached at module level keyed on grid identity (shape + boundary
        values), so it is built only once per unique grid across all var_key calls.
        """
        lons = ds.lon.values  # (n_lon,)
        lats = ds.lat.values  # (n_lat,)

        cache_key = (
            lons.shape,
            float(lons[0]),
            float(lons[-1]),
            lats.shape,
            float(lats[0]),
            float(lats[-1]),
        )

        if cache_key not in _kdtree_cache:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            tree = KDTree(np.column_stack([lon_grid.ravel(), lat_grid.ravel()]))
            _kdtree_cache[cache_key] = (tree, len(lats), len(lons))

        tree, n_lats, n_lons = _kdtree_cache[cache_key]
        _, flat_idx = tree.query(np.column_stack([query_lons, query_lats]))
        lat_idx, lon_idx = np.unravel_index(flat_idx, (n_lats, n_lons))
        return np.asarray(lat_idx), np.asarray(lon_idx)

    @staticmethod
    def _nearest_time_indices(
        ds: xr.Dataset | xr.DataArray,
        query_times: np.ndarray,
    ) -> np.ndarray:
        """Return the nearest time index for each query timestamp via searchsorted.

        Picks whichever grid step (left or right of the insertion point) is
        closer, matching xarray's method='nearest' semantics exactly.

        Both sides are pinned to nanoseconds before the integer cast. The two
        arrive at different resolutions — a Zarr time axis decodes to
        ``datetime64[ns]`` while pandas parses input strings to ``[us]`` (or
        coarser, since pandas 2 stopped forcing nanoseconds) — and casting
        those to int64 compares counts of different units. A microsecond query
        reads as 1/1000th of its true instant, sorts before every stored step,
        and every row silently lands on index 0: one arbitrary time returned
        for the whole input, varying only by location.
        """
        grid_times = ds.time.values.astype("datetime64[ns]").astype("int64")
        q = (
            pd.to_datetime(query_times)
            .to_numpy()
            .astype("datetime64[ns]")
            .astype("int64")
        )

        right = np.searchsorted(grid_times, q).clip(0, len(grid_times) - 1)
        left = (right - 1).clip(0, len(grid_times) - 1)
        return np.where(
            np.abs(grid_times[right] - q) <= np.abs(grid_times[left] - q),
            right,
            left,
        )

    @staticmethod
    def extract_from_csv(
        data: pd.DataFrame, ds: xr.Dataset | xr.DataArray, index_col: str
    ) -> pd.DataFrame:
        """
        Point extraction from a dataframe. If run as staticmethod, time, lat and lon cols should be named 'time', 'lat' and 'lon', resp.

        Uses a KDTree for spatial nearest-neighbour lookup (works on regular and
        irregular grids) and numpy searchsorted for time, then selects with
        isel() (integer indexing) which is faster than coordinate-based sel().

        This is the low-level point-extraction engine and assumes its inputs are
        already prepared; :meth:`extract_from_dataset` (or :meth:`process_single_varkey`
        for the store) establishes these preconditions for you.

        Preconditions (the caller must guarantee these — they are not validated):
            - ``data`` has columns named exactly ``time``, ``lon`` and ``lat``.
            - ``ds`` has coordinates named exactly ``lon`` and ``lat`` (read directly
              when building the KDTree).
            - If ``ds`` has a ``time`` coordinate it must be **sorted ascending** —
              nearest-time lookup uses ``np.searchsorted`` and returns wrong indices
              on an unsorted axis. No CRS is required.

        Parameters:
            ds (xr.Dataset | xr.DataArray): dataset with coords lon, lat and optionally time.

        Returns:
            pd.DataFrame: extracted variables with previous index set
        """
        valid = data[data["lon"].notna() & data["lat"].notna()]
        coords = {index_col: valid.index}

        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, valid["lon"].to_numpy(), valid["lat"].to_numpy()
        )

        isel_kwargs: dict = {
            "lon": xr.DataArray(lon_idx, dims=index_col, coords=coords),
            "lat": xr.DataArray(lat_idx, dims=index_col, coords=coords),
        }

        if "time" in ds.coords:
            time_idx = Extractor._nearest_time_indices(ds, valid["time"].values)  # type: ignore
            isel_kwargs["time"] = xr.DataArray(time_idx, dims=index_col, coords=coords)

        ds = load_dataset_to_memory(ds.isel(**isel_kwargs))
        result = ds.to_dataframe()
        return result.reindex(data.index)

    @staticmethod
    def extract_from_shp(
        data: gpd.GeoDataFrame,
        ds: xr.Dataset | xr.DataArray,
        index_col: str,
        n_workers: int = 8,
    ) -> pd.DataFrame:
        """
        Extract data from shapefile using multiprocessing starmap.

        This is the low-level geometry-extraction engine and assumes its inputs are
        already prepared; :meth:`extract_from_dataset` (or :meth:`process_single_varkey`
        for the store) establishes these preconditions for you.

        Preconditions (the caller must guarantee these — they are not validated):
            - ``data`` is a GeoDataFrame with a ``geometry`` column (and a ``time``
              column when ``ds`` has a ``time`` coordinate).
            - ``ds`` has a CRS set (``ds.rio.crs``) and spatial dims named ``x``/``y``,
              because per-geometry ``ds.rio.clip`` resolves dims by name. Datasets
              with ``lon``/``lat`` must be renamed and given a CRS first (see
              :meth:`ensure_crs`); the ``lon``/``lat`` bbox pre-select still works,
              but the clip step will fail without ``x``/``y`` + CRS.

        Args:
            gdf (gpd.GeoDataFrame): geodataframe with geometries and time column.
            ds (xr.Dataset): xarray dataset with dask arrays.
            n_workers (int, optional): Number of workers for parallel processing of geometries. Defaults to 8.

        Returns:
            pd.DataFrame with extracted values.
        """

        # Clip to the combined spatial envelope of all geometries before pulling
        # data into memory — reduces what gets computed from the dask graph.
        # Padding (sel_padded_bbox) keeps a sub-cell bbox from yielding an empty
        # slice, where clip fails with "Unable to determine bounds".
        lat_coord = "lat" if "lat" in ds.coords else "y"
        lon_coord = "lon" if "lon" in ds.coords else "x"
        if lat_coord in ds.coords and lon_coord in ds.coords:
            ds = sel_padded_bbox(
                ds, tuple(data.total_bounds), lat_coord=lat_coord, lon_coord=lon_coord
            )

        ds_computed = load_dataset_to_memory(ds)

        has_time = "time" in ds.coords

        if has_time:
            tasks = [
                (id, date, geom, ds_computed, index_col)
                for id, date, geom in zip(data.index, data.time, data.geometry)
            ]
        else:
            tasks = [
                (id, None, geom, ds_computed, index_col)
                for id, geom in zip(data.index, data.geometry)
            ]

        # rasterio builds a scratch in-memory raster per clip and reads its
        # (not yet assigned) geotransform, warning about it inside its own
        # `catch_warnings(...ignore)`. That suppression is global filter state,
        # so with the thread pool below one worker's exit un-ignores it while
        # another is mid-clip and the warning leaks — a handful per run, on
        # correct data. A permanent filter has no such window: it is in the
        # list every worker saves and restores. A real missing-geotransform
        # would still surface, as the all-NaN columns _warn_if_wholly_failed
        # and the null summary report.
        warnings.filterwarnings(
            "ignore", category=NotGeoreferencedWarning, module="rasterio"
        )

        out = []
        errors: list[Exception] = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_extract_geometry, *task, errors) for task in tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    out.append(result)

        result_df = pd.DataFrame(out).set_index(index_col)
        _warn_if_wholly_failed(result_df, errors)
        return result_df

    def _extract_bathy(
        self, data: pd.DataFrame | gpd.GeoDataFrame, n_workers: int = 8
    ) -> pd.DataFrame:
        """
        Extract bathymetry data for geometries (shp - original 15s res, calculates mean and std where the geom touches)
        and points (csv - from coarser 0.25deg res with mean and std already calculated).
        """
        vkey = "bathy"
        var_cfg = self.app_config.variables[vkey]
        store_root = self._store_dir(var_cfg)

        if self.input_type == "shp":
            if var_cfg.data_file_hires is None:
                raise ValueError(
                    "bathy config entry is missing required 'data_file_hires' field"
                )
            data_path = store_root / var_cfg.data_file_hires
        elif self.input_type == "csv":
            if var_cfg.data_file is None:
                raise ValueError(
                    "bathy config entry is missing required 'data_file' field"
                )
            data_path = store_root / var_cfg.data_file

        else:
            raise ValueError(f"Unsupported input_type: {self.input_type!r}")

        bounds = self._define_bbox(data)

        # Same two lines the store-backed paths log, so a run reads the same
        # whichever var_key produced it. The path carries the file name rather
        # than stopping at the root: bathy is a single file picked by input type
        # — the hi-res tiled zarr for geometries, the 0.25 deg netCDF for points
        # — so the root alone would not say which of the two was read. There is
        # no date range to report in its place; the layer is static.
        logger.info(f"Extracting {vkey} data from {data_path}")
        logger.info(f"{data.shape[0]} samples | static, no time axis | {bounds}")

        # The hi-res layer (shp path) is a spatially-tiled Zarr store; the 0.25°
        # layer (csv path) stays netCDF. Open by suffix so the bbox .sel() below
        # reads only the overlapping tiles instead of the full grid.
        if data_path.suffix == ".zarr":
            ds = xr.open_zarr(data_path)
        else:
            ds = xr.open_dataset(data_path)
        ds_bbox = BBox.from_dataset(ds)

        if not bounds.overlaps(ds_bbox):
            logger.warning(
                f"Data input bbox does not overlap with store data for {vkey}"
            )

        if isinstance(data, gpd.GeoDataFrame):
            ds = (
                ds.sel(
                    lon=slice(bounds.xmin, bounds.xmax),
                    lat=slice(bounds.ymin, bounds.ymax),
                ).rename({"z": "bathy", "lon": "x", "lat": "y"})
            ).compute()

            ds = self.ensure_crs(data, ds)

            tasks = [
                (id, geom, ds, self.index_col)
                for id, geom in zip(data.index, data.geometry)
            ]

            out = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_extract_geometry_bathy, *task) for task in tasks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        out.append(result)

        else:
            ds = (
                ds.sel(
                    lon=slice(bounds.xmin, bounds.xmax),
                    lat=slice(bounds.ymin, bounds.ymax),
                )
            ).compute()

            out = ds.sel(
                lon=xr.DataArray(
                    data["lon"].values,
                    dims=self.index_col,
                    coords={self.index_col: data.index},
                ),
                lat=xr.DataArray(
                    data["lat"].values,
                    dims=self.index_col,
                    coords={self.index_col: data.index},
                ),
                method="nearest",
            ).to_dataframe()

        if isinstance(out, list):
            return pd.DataFrame(out).set_index(self.index_col)
        return out

    def _slice_depth(
        self,
        ds: xr.Dataset,
        var_key: str,
        var_config,
        requested: list[str] | None,
        override: dict[str, list[int]],
    ) -> xr.Dataset:
        """
        Slice a store with a depth axis down to the columns the caller asked for.

        Levels are the extraction ones from config (``extract_depth_levels``
        over ``depth_levels``, see ``models.depth_levels_for``) with *override*
        — levels chosen in the request — replacing them per variable. Left
        unsliced, a depth axis would be averaged away by the geometry engine's
        dimensionless ``.mean()`` into one value spanning the whole range.

        *requested* may name, besides 2-D variables:
          - a 3-D variable, meaning every level it is sliced at;
          - a single ``<variable>_<level>`` column;
          - the var_key itself, meaning everything (the older single-variable
            stores, where the variable and the var_key share a name).

        Only the variables requested are sliced, so a 3-D variable nobody asked
        for needs no levels.
        """
        levels = depth_levels_for(var_key, var_config, "extract")
        levels.update(override)
        if not levels:
            raise ValueError(
                f"[{var_key}] has a depth axis but declares no depth levels. Set "
                f"depth_levels (or extract_depth_levels) in its config entry, or "
                f"choose them in the request — var_dict={{'{var_key}': "
                f"{{'<variable>': [0, 100]}}}} — without them the depth axis "
                f"would be averaged away into one value spanning the whole range."
            )

        store_vars = [str(v) for v in ds.data_vars]
        unknown = sorted(set(levels) - set(store_vars))
        if unknown:
            raise ValueError(
                f"[{var_key}] depth levels name {unknown}, which the store does "
                f"not hold. Store variables: {sorted(store_vars)}."
            )

        if requested is None or (var_key in requested and var_key not in store_vars):
            logger.info(f"[{var_key}] slicing at depth levels {levels}")
            return select_depth_levels(ds, levels, var_key)

        needed: list[str] = []
        for name in requested:
            parent = name if name in store_vars else None
            if parent is None:
                parent = next((v for v in levels if name.startswith(f"{v}_")), None)
            if parent is None:
                raise ValueError(
                    f"[{var_key}] cannot extract '{name}': the store holds "
                    f"{sorted(store_vars)}, and depth columns are named "
                    f"<variable>_<level>."
                )
            if parent not in needed:
                needed.append(parent)

        needed_levels = {v: lv for v, lv in levels.items() if v in needed}
        logger.info(f"[{var_key}] slicing at depth levels {needed_levels}")
        sliced = select_depth_levels(ds[needed], needed_levels, var_key)

        available = [str(v) for v in sliced.data_vars]
        columns: list[str] = []
        missing: list[str] = []
        for name in requested:
            if name in available:
                columns.append(name)
            elif name in needed_levels:
                columns.extend(f"{name}_{level}" for level in needed_levels[name])
            else:
                missing.append(name)
        if missing:
            raise ValueError(
                f"[{var_key}] cannot extract {missing}: at the levels in use it "
                f"yields {available}. Pass one of those, a variable name for all "
                f"its levels, choose levels in the request (var_dict="
                f"{{'{var_key}': {{'<variable>': [levels]}}}}), or change "
                f"extract_depth_levels (extract_depth_slices)."
            )
        return sliced[list(dict.fromkeys(columns))]

    def _extract_moon_phase(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """
        Extract moon ilumination from ephem library. Lat/lon values are averaged.

        Returns:
            pd.DataFrame: _description_
        """
        bounds = self._define_bbox(data)

        logger.info(f"Extracting 'MOON' data | {data.shape[0]} rows | {bounds}")

        lat = (bounds.ymin + bounds.ymax) / 2
        lon = (bounds.xmin + bounds.xmax) / 2

        observer = ephem.Observer()
        observer.lat = str(lat)
        observer.lon = str(lon)

        result = []
        for id, date in zip(data.index, data.time):
            observer.date = date
            moon = ephem.Moon(observer)
            result.append({self.index_col: id, "moon_phase": moon.phase})
        return pd.DataFrame(result).set_index(self.index_col)

    # ======================= HELPERS =========================
    def _normalize_var_dict(
        self,
        var_dict: Optional[Union[str, list[str], Mapping[str, VarSelection]]] = None,
    ) -> dict[str, VarSelection]:
        """
        Helper function to resolves var_dict arg from ``run()``

        Args:
            var_dict (Optional[Union[str, list[str], Mapping[str, VarSelection]]], optional): _description_. Defaults to None.

        Raises:
            TypeError: if type list[str] but elements not str
            TypeError: No valid var_dict

        Returns:
            dict[str, VarSelection]: _description_
        """
        if var_dict is None:
            # Exclude compiled-output variables (source: h2mare) from default extraction
            # runs — they are derived from source variables, not standalone stores.
            all_var_keys = [
                k
                for k in self.app_config.variables
                if self.app_config.variables[k].source != "h2mare"
            ]
            logger.info(
                f"No variables provided. Using all key variables from config: "
                f"{all_var_keys}"
            )
            return {k: None for k in all_var_keys}

        elif isinstance(var_dict, Mapping):
            return dict(var_dict)

        # single var_key
        elif isinstance(var_dict, str):
            return {var_dict: None}

        # list of var_keys
        elif isinstance(var_dict, list):
            if not all(isinstance(v, str) for v in var_dict):
                raise TypeError("All elements in var_dict list must be strings")
            return {vd: None for vd in var_dict}
        else:
            raise TypeError("Provide a valid var_dict")

    def ensure_crs(
        self, data: gpd.GeoDataFrame, ds: xr.Dataset | xr.DataArray
    ) -> xr.Dataset | xr.DataArray:
        """Ensure the CRS of the dataset is the same as the prepared GeoDataFrame's."""
        if ds.rio.crs != data.crs:
            return ds.rio.write_crs(data.crs, inplace=True)
        return ds

    def remove_duplicated_cols(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compares column names from two dataframes and removes duplicated columns from df1.

        Parameters:
            df1, df2 (pd.DataFrame): Older/existing data (df1) from which columns will be removed if present in newer data (df2)

        Returns:
            (pd.DataFrame) with removed duplicated cols
        """
        overlapping_cols = df1.columns.intersection(df2.columns)

        if len(overlapping_cols) > 0:
            logger.warning(
                f"Removing overlapping columns from existing dataframe: {list(overlapping_cols)}"
            )
            return df1.drop(columns=overlapping_cols)
        else:
            return df1

    # ========================  I/O =========================
    def _save_results(self, result: pd.DataFrame, output_path: Path) -> None:
        """
        Save result dataframe to output_path. Checks if exists, and if so, remove duplicated columns.

        Parameters:
            result (pd.DataFrame): Dataframe with extracted data
            output_path (Path): Path to save csv file.
        """
        logger.info(f"Saving results to {output_path}")

        if output_path.exists():
            existing_df = pd.read_csv(output_path, index_col=self.index_col)
            logger.warning(
                f"Output_path already exists. Loading {output_path} with {len(existing_df)} observations."
            )
            existing_df = self.remove_duplicated_cols(existing_df, result)
            result = existing_df.join(result, how="left")

        result = result.reset_index(drop=False)

        # shp inputs carry a `geometry` column through to the result. Shapely
        # geometries only serialize to WKT strings in a CSV — unusable as
        # geometries on read-back — so drop it from the file. The in-memory
        # return value from run() keeps it.
        result = result.drop(columns="geometry", errors="ignore")

        result.to_csv(output_path, index=False)
        logger.success("Results saved")
