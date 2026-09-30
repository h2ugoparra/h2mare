"""
Convert h2ds (or any variable) Zarr store to a Hive-partitioned Parquet store.
"""

from __future__ import annotations

import gc
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from h2mare.config import get_settings
from h2mare.format_converters.base import BaseConverter
from h2mare.models import SYSTEM_VAR_KEYS
from h2mare.storage import ZarrCatalog
from h2mare.storage.coverage import get_store_coverage, split_time_range
from h2mare.storage.parquet_indexer import ParquetIndexer
from h2mare.types import DateLike, DateRange, FilePeriod
from h2mare.utils.datetime_utils import end_of_day
from h2mare.validators import validate_file_period

# How far behind the parquet end the incremental backfill looks for "holes"
# (days whose rows were appended while a variable's compile lagged, leaving the
# column NaN). Lag holes are a recent phenomenon by construction; all-null days
# older than this are legitimate source gaps and are not rescanned every run.
_BACKFILL_HOLE_LOOKBACK_DAYS = 400


def convert_zarr_to_parquet(
    zarr_path: Path | str | Iterable[Path | str],
    parquet_root: Path | str,
    *,
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
    file_period: FilePeriod | str = FilePeriod.MONTH,
    depth: float | None = None,
    variables: list[str] | None = None,
    indexer_kwargs: Optional[dict] = None,
    open_kwargs: Optional[dict] = None,
) -> Path:
    """
    Convert an arbitrary Zarr store to a Hive-partitioned Parquet store, without
    a configured ``var_key``.

    This is the config-free counterpart to :class:`Zarr2Parquet`. It opens the
    store directly (instead of locating it through a ``ZarrCatalog`` keyed by a
    registered variable), splits the requested window into memory-sized chunks,
    and writes each chunk via :meth:`ParquetIndexer.add_data` — the same
    overlap-resolving write path the class uses. The incremental backfill mode
    (which is inherently config-driven) is intentionally not replicated.

    Args:
        zarr_path: One Zarr store path, or an iterable of them (opened together
            via ``xr.open_mfdataset(engine="zarr")``).
        parquet_root: Destination directory for the Parquet store. Unlike the
            class, no dataset sub-folder is derived — data is written here
            directly. If the store already exists, partitions are appended or
            JOINed via the indexer's standard overlap semantics.
        start_date: Start of the conversion window. Defaults to the store's
            first time step.
        end_date: End of the conversion window. Defaults to the store's last
            time step.
        file_period: Granularity of each write batch. Defaults to
            ``FilePeriod.MONTH`` so each chunk fits comfortably in memory.
        depth: Depth level (in metres) to select for stores with a ``depth``
            dimension; the nearest level is chosen. Required when the store has
            a ``depth`` dim (otherwise the time/lon/lat Parquet schema would get
            a depth cross-product).
        variables: Subset of data variables to read. ``None`` reads all.
        indexer_kwargs: Extra keyword arguments forwarded to
            :class:`ParquetIndexer` (e.g. ``time_col``/``lon_col``/``lat_col``
            for non-canonical coordinate names, or ``partition_by``).
        open_kwargs: Extra keyword arguments forwarded to the xarray open call.

    Returns:
        The ``parquet_root`` that was written.

    Raises:
        ValueError: If the store has a ``depth`` dim but ``depth`` is not given,
            or if ``start_date`` is after ``end_date``.
    """
    file_period = validate_file_period(file_period)

    if isinstance(zarr_path, (str, Path)):
        ds = xr.open_zarr(zarr_path, **(open_kwargs or {}))
    else:
        stores = [str(p) for p in zarr_path]
        # data_vars pinned for the reason netcdf2zarr pins it: xarray's default
        # flips from "all" to "minimal" here. Moot for the stores this reads —
        # every variable in them carries `time` — but these rows become Parquet
        # columns, so the set of variables the flip could change is exactly the
        # set of columns written.
        ds = xr.open_mfdataset(
            stores, engine="zarr", **{"data_vars": "all", **(open_kwargs or {})}
        )

    indexer = ParquetIndexer(Path(parquet_root), **(indexer_kwargs or {}))

    try:
        if variables is not None:
            ds = ds[variables]

        if "depth" in ds.dims and depth is None:
            raise ValueError(
                "Zarr store has a 'depth' dimension. Pass depth=<metres> to "
                "select a level before writing to the time/lon/lat Parquet store."
            )

        times = pd.to_datetime(ds.time.values)
        window = DateRange(
            start=pd.Timestamp(start_date) if start_date is not None else times.min(),
            end=pd.Timestamp(end_date) if end_date is not None else times.max(),
        )

        periods = split_time_range(window, file_period)
        logger.info(
            f"Zarr → Parquet conversion: {window.start.date()} → {window.end.date()} "
            f"({len(periods)} chunk(s)) → {Path(parquet_root)}"
        )

        for period in periods:
            df: pl.DataFrame | None = None
            try:
                # end_of_day, not period.end: periods tile the window at
                # date granularity, so a midnight bound leaves every step
                # after 00:00 on the period's last day in no period at all —
                # 23 hours per period silently absent from the Parquet store.
                sub = ds.sel(time=slice(period.start, end_of_day(period.end)))
                if depth is not None and "depth" in sub.dims:
                    sub = sub.sel(depth=depth, method="nearest")
                df = pl.from_pandas(sub.to_dataframe().reset_index())
                indexer.add_data(df)
            finally:
                del df
                gc.collect()
    finally:
        ds.close()

    return Path(parquet_root)


class Zarr2Parquet(BaseConverter):
    """
    Convert a compiled Zarr store to a Hive-partitioned Parquet store.

    The output directory is ``parquet_root / <dataset_base_name>`` where the
    base name is derived from the zarr filename by stripping the trailing date
    component.  For example, a zarr named
    ``h2mare_compiled-data-0.25deg-P1D_79W-9E-0N-69N_1998.zarr`` produces the
    folder ``h2mare_compiled-data-0.25deg-P1D_79W-9E-0N-69N``, which remains
    stable across all years and makes the dataset identity explicit.

    Date-range inference (when no explicit dates are given to :meth:`run`):

    - If the parquet store already has data: start = ``parquet_end + 1 day``,
      end = ``zarr_end``.
    - If the parquet store is empty (first run): start = ``zarr_start``,
      end = ``zarr_end``.

    Explicit dates always take priority over the inferred range.

    Args:
        var_key: Variable key that must exist in app_config.variables.
        parquet_root: Parent directory under which the dataset sub-folder is
            created.  The actual write path is
            ``parquet_root / <dataset_base_name>``.
        store_root: Override for the Zarr store root. Defaults to
            ``get_settings().STORE_ROOT``.
    """

    def __init__(
        self,
        var_key: str,
        parquet_root: Path | str,
        store_root: Optional[Path] = None,
    ) -> None:
        self.var_key = var_key
        self.app_config = get_settings().app_config

        self.zarr_repo = ZarrCatalog(self.var_key, store_root=store_root)
        repo_dates = self.zarr_repo.get_time_coverage()
        if not repo_dates:
            raise ValueError(
                f"No zarr data found for '{var_key}'. "
                "Run the compile step before converting to Parquet."
            )
        self.repo_start: pd.Timestamp = repo_dates.start
        self.repo_end: pd.Timestamp = repo_dates.end

        # Derive a stable dataset folder name from the zarr filename by stripping
        # the trailing date label (_YYYY, _YYYY-MM, or _YYYY-MM-DD).
        self.parquet_root = Path(parquet_root) / self._derive_folder_name()
        self.indexer = ParquetIndexer(
            self.parquet_root, column_groups=self._column_groups()
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        file_period: FilePeriod = FilePeriod.MONTH,
        depth: float | None = None,
        variables: list[str] | None = None,
    ) -> bool:
        """
        Convert Zarr data to Parquet, mirroring the compiler's incremental mode.

        Every conversion window is split by *file_period* (default: monthly)
        so each chunk fits comfortably in memory.

        Three modes, in priority order:

        1. **add-var** (*variables* given, no explicit dates) — merge those
           columns into every existing partition over the full Zarr range.
        2. **explicit dates** — convert exactly ``[start_date, end_date]`` with
           all variables (or *variables* if given).
        3. **incremental** (no dates, no *variables*, the default) — two regimes:

           * *append*: convert genuinely new trailing dates
             (``parquet_end + 1 day → zarr_end``) with **all** variables.
           * *backfill*: for each source var_key whose representative column lags
             behind its source coverage inside the already-written date range,
             re-read just that var_key's columns and JOIN them into the affected
             partitions. This lets a lagging variable (written as NaN while a
             faster one advanced) catch up on its own — exactly as the compiler
             resolves per-variable gaps into the h2ds Zarr.

        Args:
            start_date: Start of the conversion window. Inferred when omitted.
            end_date: End of the conversion window. Inferred when omitted.
            file_period: Granularity of each write batch. Defaults to
                ``FilePeriod.MONTH``.
            depth: Depth level to select (in metres) for variables that have a
                depth dimension. The nearest available level is chosen. Required
                for depth-aware variables (e.g. thetao, o2); ignored otherwise.
            variables: Subset of variable names to read from the Zarr and merge
                into the existing Parquet store (add-var mode).
        """
        # Each window logs exactly one line, and _convert_window's *label* names
        # the regime that produced it — the windows of the incremental mode are
        # otherwise indistinguishable from one another in the log.
        #
        # This header names the step once, before any of them. Without it the
        # first thing a run prints after the compile is a bare "Converting
        # requested range: ...", which says nothing about which conversion is
        # running or where it writes. Deliberately not repeated on the closing
        # line, which reports the outcome instead.
        t0 = time.perf_counter()
        logger.info(
            f"Zarr → Parquet conversion for '{self.var_key.upper()}' "
            f"→ {self.parquet_root}"
        )

        # Mode 1 — add-var: reprocess the full Zarr range so the overlap resolver
        # can JOIN the new columns into every partition.
        if variables is not None and start_date is None and end_date is None:
            ok = self._convert_window(
                DateRange(self.repo_start, self.repo_end),
                file_period,
                depth,
                variables,
                label=f"Merging {variables} into all existing partitions",
            )

        # Mode 2 — explicit dates (or partial override).
        elif start_date is not None or end_date is not None:
            window = self._resolve_date_range(start_date, end_date)
            # A partial override can resolve to nothing left to convert; that is
            # a no-op, not a failure (_resolve_date_range logs the reason).
            ok = (
                True
                if window is None
                else self._convert_window(
                    window,
                    file_period,
                    depth,
                    variables,
                    label="Converting requested range",
                )
            )

        # Mode 3 — incremental: append new dates, then backfill lagging columns.
        # Backfill groups are resolved up-front from the pre-append store metadata;
        # the append and backfill windows are disjoint, so execution order is free.
        else:
            ok = True

            # The pre-append store end is the pivot between the two regimes:
            # dates after it are appended, dates at or before it are backfilled.
            # Every window below is derived from it, so log it once — without it
            # the ranges that follow cannot be interpreted from the log alone.
            pivot = (
                self.indexer.get_time_coverage()
                if self.indexer._dataset_meta_initialized
                else None
            )
            if pivot is not None:
                logger.info(
                    f"Parquet store covers {pivot.start.date()} → {pivot.end.date()}"
                    f"; appending after {pivot.end.date()}, backfilling within."
                )

            backfill_groups = self._resolve_backfill_groups()

            # A ``None`` window means there is nothing new to append; backfill
            # still runs below. Errors from _convert_window now propagate instead
            # of being swallowed as "nothing to append".
            window = self._resolve_date_range(None, None)
            if window is not None:
                ok &= self._convert_window(
                    window,
                    file_period,
                    depth,
                    None,
                    label="Appending new dates",
                )

            for window, cols in backfill_groups:
                ok &= self._convert_window(
                    window,
                    file_period,
                    depth,
                    sorted(cols),
                    label=f"Backfilling {sorted(cols)} into existing partitions",
                )

        elapsed = time.perf_counter() - t0
        if ok:
            logger.success(
                f"Conversion complete for '{self.var_key.upper()}' in {elapsed:.1f}s."
            )
        else:
            logger.warning(
                f"Conversion for '{self.var_key.upper()}' finished with errors in "
                f"{elapsed:.1f}s — see messages above."
            )
        return ok

    def _convert_window(
        self,
        window: DateRange,
        file_period: FilePeriod,
        depth: float | None,
        variables: list[str] | None,
        *,
        label: str = "Converting",
    ) -> bool:
        """
        Convert a single date window to Parquet, one monthly chunk at a time.

        Reads *variables* (or all data variables when ``None``) from the Zarr for
        each chunk and writes them via ``ParquetIndexer.add_data``, which appends
        non-overlapping partitions or JOINs overlapping ones automatically.

        Args:
            label: Names the regime this window belongs to (append, backfill,
                add-var, explicit range) in the one line logged for it. The
                caller owns the wording because only it knows why the window
                exists; ``var_key`` is not repeated here — the log's ``var``
                column already carries it on every line.

        Returns ``True`` when every chunk converted without error.
        """
        periods = split_time_range(window, file_period)
        # The chunk count is only informative when the window actually splits;
        # the per-chunk lines below cover that case in full.
        chunks = f" ({len(periods)} chunks)" if len(periods) > 1 else ""
        logger.info(f"{label}: {window.start.date()} → {window.end.date()}{chunks}")

        _failed = False
        for i, period in enumerate(periods, 1):
            dt_ini, dt_end = period.start, period.end
            # The window header above already states the range; a per-chunk
            # line only adds information when the window has several chunks.
            if len(periods) > 1:
                logger.info(
                    f"  chunk {i}/{len(periods)}: {dt_ini.date()} → {dt_end.date()}"
                )
            ddf_new: pl.DataFrame | None = None
            try:
                ds = self.zarr_repo.open_dataset(
                    start_date=dt_ini, end_date=dt_end, variables=variables
                )
                if depth is not None and "depth" in ds.dims:
                    ds = ds.sel(depth=depth, method="nearest")
                elif "depth" in ds.dims:
                    raise ValueError(
                        f"Variable '{self.var_key}' has a depth dimension. "
                        "Pass --depth <metres> to select a level."
                    )
                ddf_new = pl.from_pandas(ds.to_dataframe().reset_index())
                ds.close()
                self.indexer.add_data(ddf_new)
            except Exception as e:
                logger.opt(exception=True).error(
                    f"Failed to convert '{self.var_key}' "
                    f"for {dt_ini.date()} → {dt_end.date()}: {e}"
                )
                _failed = True
            finally:
                del ddf_new
                gc.collect()

        return not _failed

    def _resolve_backfill_groups(self) -> list[tuple[DateRange, set[str]]]:
        """
        Find lagging variable columns and group them by the window to backfill.

        For every non-system source var_key whose columns appear in this Zarr
        store, the gap between its representative column's last non-null date in
        Parquet and its source coverage end is computed. Because all columns of a
        var_key share the same dates (``compiled_vars`` in config), one
        representative column is enough to date the whole group — no need to scan
        every column.

        The last non-null date alone misses *holes*: when an append runs while a
        variable's compile lags, rows land with the column NaN-padded; once a
        later append happens to carry the column (compile caught up between
        runs), the last non-null date jumps past the NaN stretch and an
        end-based window strands it forever. Holes are therefore detected
        explicitly — bounded by a lookback (older all-null days are legitimate
        source gaps, not lag holes) — and only count when the Zarr actually has
        data for them (a gap the source itself has cannot be filled, and
        re-merging it every run would never converge).

        Only the portion of the gap *inside* the already-written date range is
        returned here; genuinely new trailing dates are handled by the append
        regime in :meth:`run`. var_keys sharing an identical window are merged
        so each window is read once.

        Returns:
            List of ``(DateRange, columns)`` pairs to re-read and merge.
            Empty when the Parquet store has no data or nothing lags.
        """
        if not self.indexer._dataset_meta_initialized:
            return []
        parquet_cov = self.indexer.get_time_coverage()
        if parquet_cov is None:
            return []
        parquet_end = pd.Timestamp(parquet_cov.end)

        # Representative column per source var_key that is actually present in
        # this Zarr store (skip system keys; they track the global range).
        zarr_vars = self.zarr_repo.get_variables()
        reps: dict[str, str] = {}
        for vkey, vc in self.app_config.variables.items():
            if vkey in SYSTEM_VAR_KEYS or not vc.compiled_vars:
                continue
            rep = vc.compiled_vars[0]
            if rep in zarr_vars:
                reps[vkey] = rep

        if not reps:
            return []

        # Only the last non-null date of each representative column is needed to
        # date its group, so use the newest-first scan: it short-circuits after
        # the latest partition when nothing lags, instead of reading the whole
        # store on every incremental run.
        parquet_var_end = self.indexer.get_var_coverage_end(list(reps.values()))

        source_covs: dict[str, DateRange] = {}
        for vkey in reps:
            cov = get_store_coverage(vkey)
            if cov is not None:
                source_covs[vkey] = cov

        # Hole detection floor: never before the var's source start, and never
        # deeper than the lookback. Older all-null days are legitimate source
        # gaps (e.g. days the raw product never published), and scanning them
        # would walk the whole store on every incremental run.
        lookback_floor = parquet_end - pd.Timedelta(days=_BACKFILL_HOLE_LOOKBACK_DAYS)
        not_before = {
            reps[vkey]: max(pd.Timestamp(cov.start), lookback_floor).to_pydatetime()
            for vkey, cov in source_covs.items()
        }
        hole_starts = self.indexer.get_var_backfill_start(
            [reps[vkey] for vkey in source_covs], not_before=not_before
        )

        groups: dict[tuple[pd.Timestamp, pd.Timestamp], set[str]] = defaultdict(set)
        for vkey, rep in reps.items():
            source_cov = source_covs.get(vkey)
            if source_cov is None:
                continue
            # Backfill only within already-written dates; beyond parquet_end is
            # the append regime's responsibility.
            window_end = min(pd.Timestamp(source_cov.end), parquet_end)

            rep_end = parquet_var_end.get(rep)
            window_start = (
                pd.Timestamp(rep_end) + pd.Timedelta(days=1)
                if rep_end is not None
                else pd.Timestamp(source_cov.start)
            )

            hole = hole_starts.get(rep)
            if hole is not None and pd.Timestamp(hole) < window_start:
                fillable = self._fillable_hole_dates(
                    rep, DateRange(pd.Timestamp(hole), window_end)
                )
                if fillable:
                    window_start = min(window_start, fillable[0])
                else:
                    logger.debug(
                        f"{vkey}: null days from {pd.Timestamp(hole).date()} match "
                        "source gaps in the Zarr — nothing to backfill there."
                    )

            if window_start > window_end:
                logger.debug(f"{vkey}: parquet up to date, no backfill.")
                continue

            cols = self.app_config.variables[vkey].compiled_vars or []
            groups[(window_start, window_end)].update(cols)
            logger.debug(
                f"{vkey}: backfill {window_start.date()} → {window_end.date()} ({cols})"
            )

        return [(DateRange(s, e), cols) for (s, e), cols in groups.items()]

    def _fillable_hole_dates(
        self, column: str, window: DateRange
    ) -> list[pd.Timestamp]:
        """
        Dates in *window* where the Zarr has non-null data for *column* but the
        Parquet store does not (rows missing entirely, or the column all-null).

        Separates strandable lag holes (Zarr has the data → backfillable) from
        legitimate source gaps (Zarr is null too → nothing to gain, and
        re-merging the window every incremental run would never converge).
        """
        try:
            ds = self.zarr_repo.open_dataset(
                start_date=window.start, end_date=window.end, variables=[column]
            )
        except FileNotFoundError:
            return []
        try:
            da = ds[column]
            mask = da.notnull().any(dim=[d for d in da.dims if d != "time"]).compute()
            times = pd.to_datetime(ds.time.values).normalize()
            zarr_dates = set(times[mask.values])
        finally:
            ds.close()

        if not zarr_dates:
            return []

        pq = (
            self.indexer.scan(dates=(window.start, window.end), columns=[column])
            .group_by(self.indexer.time_col)
            .agg(pl.col(column).is_not_null().any().alias("has"))
            .collect(engine="streaming")
        )
        pq_dates = {
            pd.Timestamp(d)
            for d, has in zip(pq[self.indexer.time_col], pq["has"])
            if has
        }
        return sorted(zarr_dates - pq_dates)

    def sync_data(self, remote_root: Optional[Path] = None) -> None:
        """
        Copy the local Parquet store to a remote location.

        If *remote_root* is not provided, defaults to
        ``get_settings().STORE_ROOT / "parquet" / var_key``.  The backup is silently
        skipped when ``STORE_ROOT`` is not configured.

        Args:
            remote_root: Explicit destination root. The variable sub-directory
                is appended automatically when omitted.
        """
        if remote_root is None:
            store_root = get_settings().STORE_ROOT
            if store_root is None:
                logger.warning(
                    "STORE_ROOT is not set — skipping Parquet backup. "
                    "Set STORE_ROOT in .env or pass remote_root explicitly."
                )
                return
            remote_root = store_root / "parquet"

        dest = remote_root / self.parquet_root.name
        logger.info(f"Backing up Parquet: {self.parquet_root} → {dest}")
        try:
            shutil.copytree(str(self.parquet_root), str(dest), dirs_exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.exception(f"Parquet backup failed: {e}")
            return
        logger.success("Parquet backup complete.")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _column_groups(self) -> dict[str, str]:
        """
        Map every compiled column to the var_key that produces it.

        Lets the store report a lag as "seapodym is behind" rather than listing
        the nine columns seapodym happens to compile into — the columns of a
        var_key always move together (they share ``compiled_vars`` dates), so
        they carry no information the var_key does not.
        """
        return {
            col: vkey
            for vkey, vc in self.app_config.variables.items()
            for col in (vc.compiled_vars or [])
        }

    def _derive_folder_name(self) -> str:
        """
        Derive a stable dataset folder name from the zarr catalog filenames.

        Takes the first filename in the catalog, strips the extension and the
        trailing date label (``_YYYY``, ``_YYYY-MM``, or ``_YYYY-MM-DD``), and
        returns what remains.  Falls back to ``var_key`` if the catalog is empty
        or the filename does not match the expected pattern.

        Example::

            "h2mare_compiled-data-0.25deg-P1D_79W-9E-0N-69N_1998.zarr"
            → "h2mare_compiled-data-0.25deg-P1D_79W-9E-0N-69N"
        """
        df = self.zarr_repo.df
        if df.empty or "filename" not in df.columns:
            return self.var_key
        stem = Path(df["filename"].iloc[0]).stem
        base = re.sub(r"_\d{4}(-\d{2}(-\d{2})?)?$", "", stem)
        return base or self.var_key

    def _resolve_date_range(
        self,
        start_date: str | pd.Timestamp | None,
        end_date: str | pd.Timestamp | None,
    ) -> DateRange | None:
        """
        Resolve the conversion window.

        Priority:
        1. Explicit arguments (both must be provided together).
        2. Incremental gap: ``parquet_end + 1 day`` → ``zarr_end``.
        3. First run: ``zarr_start`` → ``zarr_end`` (parquet store empty).

        Mirrors :func:`h2mare.storage.coverage.resolve_date_range`: an inferred
        range that has nothing left to convert is a clean no-op (``None``), while
        explicitly supplied dates in the wrong order stay a caller error.

        Returns:
            The window to convert, or ``None`` when the range was inferred and
            the Parquet store is already up to date — callers should skip rather
            than treat it as an error.

        Raises:
            ValueError: If both dates are supplied explicitly and start > end.
        """
        if start_date is not None and end_date is not None:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            # Checked before constructing DateRange, which would otherwise raise
            # with a generic message that does not name the offending arguments.
            if start > end:
                raise ValueError(
                    f"start_date ({start.date()}) must be before end_date ({end.date()})"
                )
            return DateRange(start, end)

        # Infer from the gap between the parquet store and the zarr store
        parquet_coverage = (
            self.indexer.get_time_coverage()
            if self.indexer._dataset_meta_initialized
            else None
        )

        inferred_start = (
            parquet_coverage.end + pd.Timedelta(days=1)
            if parquet_coverage is not None
            else self.repo_start
        )
        # Allow a partial override: honour whichever side was explicitly given
        start = pd.Timestamp(start_date) if start_date is not None else inferred_start
        end = pd.Timestamp(end_date) if end_date is not None else self.repo_end

        if start > end:
            logger.info(
                f"Parquet store for '{self.var_key}' is already up to date "
                f"(inferred start {start.date()} > zarr end {end.date()})."
            )
            return None

        logger.debug(
            f"Inferred Parquet range for '{self.var_key}': "
            f"{start.date()} → {end.date()}"
        )
        return DateRange(start, end)
