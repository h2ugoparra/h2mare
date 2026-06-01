"""
Convert h2ds (or any variable) Zarr store to a Hive-partitioned Parquet store.
"""

from __future__ import annotations

import gc
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
from loguru import logger

from h2mare.config import get_settings
from h2mare.format_converters.base import BaseConverter
from h2mare.models import SYSTEM_VAR_KEYS
from h2mare.storage import ZarrCatalog
from h2mare.storage.coverage import get_store_coverage, split_time_range
from h2mare.storage.parquet_indexer import ParquetIndexer
from h2mare.types import DateRange, TimeResolution


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
        self.indexer = ParquetIndexer(self.parquet_root)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        time_resolution: TimeResolution = TimeResolution.MONTH,
        depth: float | None = None,
        variables: list[str] | None = None,
    ) -> bool:
        """
        Convert Zarr data to Parquet, mirroring the compiler's incremental mode.

        Every conversion window is split by *time_resolution* (default: monthly)
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
            time_resolution: Granularity of each write batch. Defaults to
                ``TimeResolution.MONTH``.
            depth: Depth level to select (in metres) for variables that have a
                depth dimension. The nearest available level is chosen. Required
                for depth-aware variables (e.g. thetao, o2); ignored otherwise.
            variables: Subset of variable names to read from the Zarr and merge
                into the existing Parquet store (add-var mode).
        """
        logger.info(
            f"Initializing Zarr->Parquet conversion for variable key: {self.var_key.upper()}"
        )
        # Mode 1 — add-var: reprocess the full Zarr range so the overlap resolver
        # can JOIN the new columns into every partition.
        if variables is not None and start_date is None and end_date is None:
            logger.info(
                f"add-var mode: merging {variables} into all existing partitions "
                f"({self.repo_start.date()} → {self.repo_end.date()})"
            )
            return self._convert_window(
                DateRange(self.repo_start, self.repo_end),
                time_resolution,
                depth,
                variables,
            )

        # Mode 2 — explicit dates (or partial override).
        if start_date is not None or end_date is not None:
            start, end = self._resolve_date_range(start_date, end_date)
            return self._convert_window(
                DateRange(start, end), time_resolution, depth, variables
            )

        # Mode 3 — incremental: append new dates, then backfill lagging columns.
        # Backfill groups are resolved up-front from the pre-append store metadata;
        # the append and backfill windows are disjoint, so execution order is free.
        ok = True
        backfill_groups = self._resolve_backfill_groups()

        try:
            start, end = self._resolve_date_range(None, None)
            logger.info(
                f"Appending new dates for '{self.var_key.upper()}': "
                f"{start.date()} → {end.date()}"
            )
            ok &= self._convert_window(
                DateRange(start, end), time_resolution, depth, None
            )
        except ValueError as e:
            logger.info(f"No new dates to append: {e}")

        for window, cols in backfill_groups:
            logger.info(
                f"Backfilling {sorted(cols)} into existing partitions: "
                f"{window.start.date()} → {window.end.date()}"
            )
            ok &= self._convert_window(window, time_resolution, depth, sorted(cols))

        return ok

    def _convert_window(
        self,
        window: DateRange,
        time_resolution: TimeResolution,
        depth: float | None,
        variables: list[str] | None,
    ) -> bool:
        """
        Convert a single date window to Parquet, one monthly chunk at a time.

        Reads *variables* (or all data variables when ``None``) from the Zarr for
        each chunk and writes them via ``ParquetIndexer.add_data``, which appends
        non-overlapping partitions or JOINs overlapping ones automatically.

        Returns ``True`` when every chunk converted without error.
        """
        periods = split_time_range(window, time_resolution)
        logger.info(
            f"Zarr → Parquet conversion for '{self.var_key.upper()}': "
            f"{window.start.date()} → {window.end.date()} ({len(periods)} chunk(s))"
        )

        _failed = False
        for period in periods:
            dt_ini, dt_end = period.start, period.end
            logger.debug(f"  chunk {dt_ini.date()} → {dt_end.date()}")
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
        var_key share the same dates (``variables_to_compile`` in config), one
        representative column is enough to date the whole group — no need to scan
        every column.

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
            if vkey in SYSTEM_VAR_KEYS or not vc.variables_to_compile:
                continue
            rep = vc.variables_to_compile[0]
            if rep in zarr_vars:
                reps[vkey] = rep

        if not reps:
            return []

        # Only the last non-null date of each representative column is needed to
        # date its group, so use the newest-first scan: it short-circuits after
        # the latest partition when nothing lags, instead of reading the whole
        # store on every incremental run.
        parquet_var_end = self.indexer.get_var_coverage_end(list(reps.values()))

        groups: dict[tuple[pd.Timestamp, pd.Timestamp], set[str]] = defaultdict(set)
        for vkey, rep in reps.items():
            source_cov = get_store_coverage(vkey)
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

            if window_start > window_end:
                logger.debug(f"{vkey}: parquet up to date, no backfill.")
                continue

            cols = self.app_config.variables[vkey].variables_to_compile or []
            groups[(window_start, window_end)].update(cols)
            logger.debug(
                f"{vkey}: backfill {window_start.date()} → {window_end.date()} ({cols})"
            )

        return [(DateRange(s, e), cols) for (s, e), cols in groups.items()]

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
            if get_settings().STORE_ROOT is None:
                logger.warning(
                    "STORE_ROOT is not set — skipping Parquet backup. "
                    "Set STORE_ROOT in .env or pass remote_root explicitly."
                )
                return
            remote_root = get_settings().STORE_ROOT / "parquet"

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
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Resolve the conversion window.

        Priority:
        1. Explicit arguments (both must be provided together).
        2. Incremental gap: ``parquet_end + 1 day`` → ``zarr_end``.
        3. First run: ``zarr_start`` → ``zarr_end`` (parquet store empty).

        Raises:
            ValueError: If explicit start > end, or the inferred start
                is already past the zarr end (nothing new to convert).
        """
        if start_date is not None and end_date is not None:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            if start > end:
                raise ValueError(
                    f"start_date ({start.date()}) must be before end_date ({end.date()})"
                )
            return start, end

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
            raise ValueError(
                f"Parquet store is already up to date "
                f"(inferred start {start.date()} > zarr end {end.date()})."
            )

        logger.info(
            f"Inferred Parquet range for '{self.var_key}': "
            f"{start.date()} → {end.date()}"
        )
        return start, end
