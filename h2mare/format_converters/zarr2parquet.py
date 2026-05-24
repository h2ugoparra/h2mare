"""
Convert h2ds (or any variable) Zarr store to a Hive-partitioned Parquet store.
"""

from __future__ import annotations

import gc
import re
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
from loguru import logger

from h2mare.config import settings
from h2mare.storage import ZarrCatalog
from h2mare.storage.coverage import split_time_range
from h2mare.storage.parquet_indexer import ParquetIndexer
from h2mare.types import DateRange, TimeResolution


class Zarr2Parquet:
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
            ``settings.STORE_ROOT``.
    """

    def __init__(
        self,
        var_key: str,
        parquet_root: Path | str,
        store_root: Optional[Path] = None,
    ) -> None:
        self.var_key = var_key

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
    ) -> None:
        """
        Convert Zarr data to Parquet for the resolved date range.

        The range is always split by *time_resolution* (default: monthly) so
        that each chunk fits comfortably in memory regardless of how wide the
        requested window is.

        Args:
            start_date: Start of the conversion window. Inferred from the
                parquet gap when omitted.
            end_date: End of the conversion window. Defaults to the zarr
                end date when omitted.
            time_resolution: Granularity of each write batch. Defaults to
                ``TimeResolution.MONTH``.
            depth: Depth level to select (in metres) for variables that have a
                depth dimension. The nearest available level is chosen. Required
                for depth-aware variables (e.g. thetao, o2); ignored otherwise.
        """
        start, end = self._resolve_date_range(start_date, end_date)
        periods = split_time_range(DateRange(start, end), time_resolution)

        logger.info(
            f"Starting Zarr → Parquet for '{self.var_key}': "
            f"{start.date()} → {end.date()} ({len(periods)} chunk(s))"
        )

        for period in periods:
            dt_ini, dt_end = period.start, period.end
            logger.info(f"  chunk {dt_ini.date()} → {dt_end.date()}")
            ddf_new: pl.DataFrame | None = None
            try:
                ds = self.zarr_repo.open_dataset(start_date=dt_ini, end_date=dt_end)
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
                logger.error(
                    f"Failed to convert '{self.var_key}' "
                    f"for {dt_ini.date()} → {dt_end.date()}: {e}"
                )
            finally:
                del ddf_new
                gc.collect()

    def sync_data(self, remote_root: Optional[Path] = None) -> None:
        """
        Copy the local Parquet store to a remote location.

        If *remote_root* is not provided, defaults to
        ``settings.STORE_ROOT / "parquet" / var_key``.  The backup is silently
        skipped when ``STORE_ROOT`` is not configured.

        Args:
            remote_root: Explicit destination root. The variable sub-directory
                is appended automatically when omitted.
        """
        if remote_root is None:
            if settings.STORE_ROOT is None:
                logger.warning(
                    "STORE_ROOT is not set — skipping Parquet backup. "
                    "Set STORE_ROOT in .env or pass remote_root explicitly."
                )
                return
            remote_root = settings.STORE_ROOT / "parquet"

        logger.info(f"Backing up Parquet: {self.parquet_root} → {remote_root}")
        try:
            shutil.copytree(str(self.parquet_root), str(remote_root), dirs_exist_ok=True)
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
