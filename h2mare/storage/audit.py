"""
Store integrity checks: does what is on disk match what it claims to cover?

Every coverage mechanism in the pipeline is a *frontier*, not a *density*.
``get_store_coverage`` returns a ``DateRange``; ``_h2ds_nonnull_ends`` tracks the
last non-null date per variable. Both answer "how far have we got", never "is
what we have solid". A store holding 1999-01-01 and 1999-12-31 with 237 days
missing in between reports itself healthy to all of them.

This module asks the other question. It is deliberately cheap and deliberately
narrow:

* **Cheap** — the Zarr side reads coordinates only, never data. Scanning all
  435 files of the production store takes ~55s, which is why there is no cache
  and no incremental mode to get wrong. The Parquet side reads footer
  statistics, so it reads no data either.
* **Narrow** — it flags days missing from a *time axis*, not days whose values
  are null. Those are two different failures that look identical downstream
  (both become all-null days in ``h2ds``) but are cleanly separable here:

  =============  ==========================================  ==================
  fsle_max 1999  128/365 steps — days absent from the axis    pipeline defect
  chl 1999       365/365 steps, 3 days null-valued            genuine source gap
  =============  ==========================================  ==================

  A check on axis length catches the defect and is structurally silent on chl.
  A check on values fires on both — and a check that flags chl every year is
  one people learn to ignore, at which point it protects nothing.

``check_slice_health`` is the value-based check, for the cases where a store is
present and complete but degenerate. It is the second line of defence and is
run only on request (``--values``), because it is the expensive one: it reads
every cell, and the stores are far past what page cache holds — chl alone is
97 GB across 29 files. Callers auditing more than one variable should bound it
with a window.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.models import step_freq
from h2mare.types import DateRange
from h2mare.utils.datetime_utils import end_of_day
from h2mare.utils.paths import resolve_store_path
from h2mare.validators import validate_var_key


class AxisGap(NamedTuple):
    """Days absent from a store's time axis, strictly inside its own span."""

    path: Path
    span: tuple[pd.Timestamp, pd.Timestamp]
    missing: pd.DatetimeIndex


class SliceIssue(NamedTuple):
    """A time slice that is present but carries no usable data."""

    path: Path
    variable: str
    date: pd.Timestamp
    kind: str  # "empty" | "degenerate"
    detail: str


class VarAudit(NamedTuple):
    """Everything found for one ``var_key``."""

    var_key: str
    store_root: Path
    n_files: int
    gaps: list[AxisGap]
    slices: list[SliceIssue]
    errors: list[str]
    # Days excluded because config records the provider never published them.
    # Carried as the dates rather than a count so a caller can show which days
    # were suppressed — a suppression list nothing can print is one that grows
    # unnoticed into a way to bury real defects.
    known_gaps: pd.DatetimeIndex = pd.DatetimeIndex([])
    # False when the variable has no store directory at all — `moon` is
    # computed at compile time and never gets one. Distinguishing this from an
    # empty-but-present store keeps "nothing was checked" from rendering as a
    # pass.
    store_exists: bool = True
    # Whether a store *should* be there. Read together with store_exists: the
    # pair separates "correctly has none" from "should have one and doesn't".
    store_expected: bool = True

    @property
    def n_known_gaps(self) -> int:
        return len(self.known_gaps)

    @property
    def ok(self) -> bool:
        return not (self.gaps or self.slices or self.errors)


def expects_store(var_config) -> bool:
    """
    Whether a missing store directory is news for this variable.

    True when something downloads it. A variable with no registered downloader
    is produced inside the pipeline — ``moon`` is computed in the compiler and
    written straight into ``h2ds``, so it never gets a directory of its own —
    and its absence is not evidence of anything. For a downloaded variable it
    is: that store holds the only copy of what was fetched, and an absent one
    means either nothing has ever run or ``STORE_ROOT`` points at the wrong,
    or unmounted, drive.

    This also treats the other two derived keys, ``bathy`` and ``h2ds``, as not
    expected — they do have stores, but rebuildable ones, so a missing one says
    a step has not been run rather than that data was lost. The case that must
    not pass quietly is a bad ``STORE_ROOT``, and that still trips every
    downloaded variable at once.

    Imported inside the function because the downloader package reads from
    ``storage`` (``cds_downloader`` calls ``get_store_coverage``), so a
    module-level import here would close the cycle.
    """
    from h2mare.downloader.registry import DOWNLOADER_REGISTRY

    return getattr(var_config, "source", None) in DOWNLOADER_REGISTRY


def known_gap_days(var_config) -> pd.DatetimeIndex:
    """
    Expand a variable's ``known_gaps`` config entries into individual days.

    Accepts ``"YYYY-MM-DD"`` and the closed interval ``"YYYY-MM-DD/YYYY-MM-DD"``.
    Malformed entries are warned about and skipped rather than raising: a typo
    in a suppression list must not be able to stop a pipeline run.

    These are days the provider never published. A source shipping one file per
    day produces an *axis* hole when it skips one, which is otherwise
    indistinguishable from data the pipeline lost, so without this the checks
    would report the same unfixable day on every run — and a check that cries
    wolf stops being read.
    """
    entries = getattr(var_config, "known_gaps", None) or []
    days: list[pd.DatetimeIndex] = []
    for entry in entries:
        try:
            if "/" in str(entry):
                start, end = str(entry).split("/", 1)
                days.append(pd.date_range(start.strip(), end.strip(), freq="D"))
            else:
                days.append(pd.DatetimeIndex([pd.to_datetime(str(entry))]))
        except Exception as e:
            logger.warning(f"Ignoring malformed known_gaps entry {entry!r}: {e}")

    if not days:
        return pd.DatetimeIndex([])
    return (
        pd.DatetimeIndex(np.concatenate([d.values for d in days]))
        .normalize()
        .unique()
        .sort_values()
    )


def contiguous_blocks(
    dates: pd.DatetimeIndex,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Compress a date index into ``(first, last)`` runs of consecutive days."""
    if len(dates) == 0:
        return []
    dates = pd.DatetimeIndex(dates).sort_values()
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = dates[0]
    for day in dates[1:]:
        if (day - prev).days > 1:
            runs.append((start, prev))
            start = day
        prev = day
    runs.append((start, prev))
    return runs


def format_date_blocks(dates: pd.DatetimeIndex, max_blocks: int = 8) -> str:
    """Render a date index inline: ``'2025-06-02, 2025-07-10→2025-07-14'``."""
    runs = contiguous_blocks(dates)
    if not runs:
        return "none"
    shown = [
        str(a.date()) if a == b else f"{a.date()}→{b.date()}"
        for a, b in runs[:max_blocks]
    ]
    if len(runs) > max_blocks:
        shown.append(f"… and {len(runs) - max_blocks} more block(s)")
    return ", ".join(shown)


def interior_gaps(dates: pd.DatetimeIndex, freq: str = "D") -> pd.DatetimeIndex:
    """
    Steps absent from *dates*, strictly between its own first and last entry.

    Restricting to the interior is what makes this usable without an allowlist.
    A store whose tail stops short of today is ordinary provider lag; a day the
    provider published, surrounded on both sides by days that are present, is
    not. Across all 435 files of the production store this rule produced two
    findings and no false positives — chl's null days included, since they are
    on the axis.

    *freq* is the store's own cadence (see :func:`h2mare.models.step_freq`).
    Daily stamps are normalized first, so a product publishing at 12:00 counts
    as its calendar day; an hourly axis is compared as-is, because normalizing
    it would hide every missing hour behind the day that survives.
    """
    if len(dates) < 2:
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(dates)
    if freq == "D":
        dates = dates.normalize()
    dates = dates.unique().sort_values()
    return pd.date_range(dates[0], dates[-1], freq=freq).difference(dates)


def check_slice_health(
    ds: xr.Dataset, *, window: Optional[DateRange] = None
) -> list[tuple[str, pd.Timestamp, str, str]]:
    """
    Per-(variable, time) reduction returning every empty or degenerate slice.

    Every slice is inspected, not just the last: an interior hole is invisible
    from ``isel(time=-1)``, the one position that cannot reveal one. The
    reduction is ``min``/``max`` rather than ``np.unique``, which would sort and
    fully materialise a dask slice to answer what one pass answers, and which
    cannot separate "all missing" from "constant" because NaN collapses to a
    single unique value.

    One lazy pass yields ``(n_finite, min, max)`` per slice, from which both
    signals fall out separately:

    * ``n_finite == 0``            → ``"empty"``      (no usable data at all)
    * ``min == max``, n_finite > 0 → ``"degenerate"`` (a single repeated value)

    Every reduction for every variable goes into a single ``compute()``. That
    is not a tidiness point: these stores are far too large to sit in page
    cache — chl alone is 97 GB — so each separate ``.values`` call is a real
    re-read from disk. Computing them one at a time made this three times
    slower than the disk requires.

    Args:
        ds: Dataset to inspect.
        window: Restrict the scan to this range. Reading values is disk-bound
            and unbounded scans cover the whole store, so callers auditing more
            than one variable should pass one.

    Returns:
        ``(variable, date, kind, detail)`` tuples. Empty when everything is fine.
    """
    if window is not None:
        # end_of_day, not window.end: the bound is a date, so a store whose
        # stamps are not at midnight would have its last day judged on one step
        # (an hourly store) or dropped from the scan entirely (a noon-stamped
        # daily one) — a value check that silently skips a day is worse than
        # none, since it reports the day clean.
        ds = ds.sel(time=slice(window.start, end_of_day(window.end)))
        # A file wholly outside the window keeps its variables but loses every
        # timestep, and min/max over a zero-length axis raises rather than
        # returning empty. Nothing to check here.
        if "time" not in ds.coords or ds.sizes.get("time", 0) == 0:
            return []

    # Keys are generated positionally so they cannot collide with a real
    # variable name; `columns` maps them back.
    lazy: dict[str, xr.DataArray] = {}
    columns: list[tuple[str, str, str, str]] = []

    for i, (name, da) in enumerate(ds.data_vars.items()):
        if "time" not in da.dims:
            continue
        spatial = [d for d in da.dims if d != "time"]
        if not spatial:
            continue
        k_n, k_lo, k_hi = f"_{i}_n", f"_{i}_lo", f"_{i}_hi"
        lazy[k_n] = np.isfinite(da).sum(dim=spatial)
        lazy[k_lo] = da.min(dim=spatial, skipna=True)
        lazy[k_hi] = da.max(dim=spatial, skipna=True)
        columns.append((str(name), k_n, k_lo, k_hi))

    if not lazy:
        return []

    stats = xr.Dataset(lazy).compute()
    times = pd.DatetimeIndex(ds["time"].values).normalize()

    issues: list[tuple[str, pd.Timestamp, str, str]] = []
    for name, k_n, k_lo, k_hi in columns:
        n_vals = np.asarray(stats[k_n].values)
        lo_vals = np.asarray(stats[k_lo].values)
        hi_vals = np.asarray(stats[k_hi].values)
        for i, when in enumerate(times):
            if n_vals[i] == 0:
                issues.append((name, when, "empty", "no finite values"))
            elif lo_vals[i] == hi_vals[i]:
                issues.append(
                    (name, when, "degenerate", f"single value {lo_vals[i]:.6g}")
                )

    return issues


def audit_zarr_file(
    path: Path,
    *,
    check_values: bool = False,
    known_gaps: Optional[pd.DatetimeIndex] = None,
    value_window: Optional[DateRange] = None,
    freq: str = "D",
) -> tuple[
    Optional[AxisGap],
    list[SliceIssue],
    Optional[str],
]:
    """
    Audit one Zarr store. Coordinates only unless *check_values* is set.

    Days listed in *known_gaps* are dropped from **both** results. They are
    days the provider never published, which is exactly what an all-null slice
    looks like — suppressing them from the axis check but not the value check
    would leave the value check reporting the same unfixable days forever.
    """
    try:
        ds = xr.open_zarr(path, consolidated=False)
    except Exception as e:
        return None, [], f"{path.name}: could not open ({e})"

    try:
        if "time" not in ds.coords:
            return None, [], None

        dates = pd.DatetimeIndex(ds.time.values)
        if freq == "D":
            dates = dates.normalize()
        dates = dates.unique().sort_values()
        missing = interior_gaps(dates, freq=freq)
        if known_gaps is not None and len(missing):
            # known_gaps are whole days, so drop a missing step whose *day* is
            # listed — on an hourly axis a day-level difference() matches only
            # the 00:00 step and would leave the other 23 reported forever.
            missing = missing[~missing.normalize().isin(known_gaps)]
        gap = (
            AxisGap(path=path, span=(dates[0], dates[-1]), missing=missing)
            if len(missing)
            else None
        )

        slices: list[SliceIssue] = []
        if check_values:
            # `known_gaps or []` would call bool() on the Index, which raises
            # "truth value is ambiguous" — caught by the handler below and
            # silently turned into zero findings.
            suppress = set(known_gaps) if known_gaps is not None else set()
            slices = [
                SliceIssue(path=path, variable=v, date=d, kind=k, detail=detail)
                for v, d, k, detail in check_slice_health(ds, window=value_window)
                if d.normalize() not in suppress
            ]
        return gap, slices, None
    except Exception as e:
        return None, [], f"{path.name}: audit failed ({e})"
    finally:
        ds.close()


def audit_var_key(
    var_key: str,
    *,
    app_config: Optional[AppConfig] = None,
    store_root: Optional[Path] = None,
    check_values: bool = False,
    value_window: Optional[DateRange] = None,
    progress: bool = False,
) -> VarAudit:
    """
    Audit every Zarr store belonging to *var_key*.

    Reads the stores directly rather than the catalog index: the index records
    ``num_timesteps`` and a start/end per file, which is exactly the frontier
    view that cannot see an interior hole. It is also what a hard kill leaves
    stale, and a hard kill is the most likely origin of a ragged store.
    """
    config = app_config or get_settings().app_config
    var_key = validate_var_key(var_key, config)
    # warn_if_missing is for callers about to write. This one only reads, and
    # its "will be created when data is added" advice is simply false here —
    # `moon` is computed at compile time and never gets a store at all. The
    # absence is reported through store_exists instead, where it is attributable
    # to a variable rather than being a log line that scrolls past.
    root = resolve_store_path(
        config.variables[var_key], store_root, warn_if_missing=False
    )

    gaps: list[AxisGap] = []
    slices: list[SliceIssue] = []
    errors: list[str] = []

    suppressed = known_gap_days(config.variables[var_key])
    # Compare each store against a calendar at its own cadence: a daily grid
    # cannot see a missing hour in an hourly store.
    freq = step_freq(config.variables[var_key])

    files = sorted(root.glob("*.zarr")) if root.exists() else []

    # A bar only earns its place on the value scan. The axis check runs at
    # ~126 ms/file, where progress reporting is pure noise; the value scan is
    # disk-bound over the whole store and has nothing to show until it
    # finishes, which for chl is 25 minutes of silence.
    stream = files
    if progress and files:
        from tqdm import tqdm

        # disable=None: tqdm auto-disables off a tty, so a scheduled run does
        # not persist one log line per refresh tick.
        stream = tqdm(files, desc=f"{var_key:<14}", unit="file", disable=None)

    for path in stream:
        gap, found, error = audit_zarr_file(
            path,
            check_values=check_values,
            known_gaps=suppressed,
            value_window=value_window,
            freq=freq,
        )
        if gap:
            gaps.append(gap)
        slices.extend(found)
        if error:
            errors.append(error)

    return VarAudit(
        var_key=var_key,
        store_root=root,
        n_files=len(files),
        gaps=gaps,
        slices=slices,
        errors=errors,
        known_gaps=suppressed,
        store_exists=root.exists(),
        store_expected=expects_store(config.variables[var_key]),
    )


def audit_parquet_nulls(
    parquet_root: Path, columns: Optional[list[str]] = None
) -> list[tuple[Path, str]]:
    """
    Columns that are entirely null in a Parquet file, from footer metadata.

    Costs no data read: every Parquet footer carries a per-column
    ``null_count`` per row group, so "is this column all null in this file"
    is a metadata comparison. With ``year=/month=`` partitioning that gives
    month granularity for free.

    Worth having as a parity check even though the Zarr side is authoritative,
    because the Parquet representation is the more dangerous of the two: a
    missing time step is detectable by counting an axis, but a null column
    inside complete-looking rows is not.

    Returns:
        ``(file, column)`` pairs. Empty when nothing is wholly null.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from h2mare.storage.parquet_store import iter_store_parquet_files

    findings: list[tuple[Path, str]] = []
    skip = {"time", "lon", "lat", "year", "month", "day"}

    for path in iter_store_parquet_files(parquet_root):
        try:
            pf = pq.ParquetFile(path)
            meta, arrow_schema = pf.metadata, pf.schema_arrow
        except Exception as e:
            logger.warning(f"Could not read parquet footer for {path.name}: {e}")
            continue
        if meta.num_rows == 0:
            continue

        names = meta.schema.names
        for col, name in enumerate(names):
            if name in skip or (columns is not None and name not in columns):
                continue

            # A null-typed column holds nothing by construction and carries no
            # statistics to consult.
            field = arrow_schema.field(name) if name in arrow_schema.names else None
            if field is not None and pa.types.is_null(field.type):
                findings.append((path, name))
                continue

            nulls = 0
            for rg in range(meta.num_row_groups):
                stats = meta.row_group(rg).column(col).statistics
                if stats is None or stats.null_count is None:
                    nulls = -1
                    break
                nulls += stats.null_count
            if nulls == meta.num_rows:
                findings.append((path, name))

    return findings
