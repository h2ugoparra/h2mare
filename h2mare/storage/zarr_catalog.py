"""
Zarr catalog management for tracking processed zarr datasets.

`ZarrCatalog` is the entry point for a variable's Zarr store. It builds store
paths and composes the two halves it delegates to: a
:class:`~h2mare.storage.zarr_index.ZarrIndex` for catalog bookkeeping and a
:class:`~h2mare.storage.zarr_reader.ZarrReader` for opening the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.models import TimeStep
from h2mare.storage.zarr_index import ZarrIndex, _variables_list
from h2mare.storage.zarr_reader import ZarrReader
from h2mare.types import BBox, DateLike, DateRange, DatesLike, FilePeriod
from h2mare.utils.datetime_utils import end_of_day
from h2mare.utils.labels import create_label_from_dataset
from h2mare.utils.paths import resolve_store_path
from h2mare.validators import validate_file_period, validate_var_key


# ================ Convenience functions for quick access ==========================
def get_zarr_time_coverage(var_key: str) -> DateRange | None:
    """
    Time span of *var_key*'s Zarr store, or None if it holds nothing.

    Convenience wrapper that builds a default :class:`ZarrCatalog` — so it
    resolves the store root the usual way and takes no root argument. Callers
    needing a specific root should construct the catalog themselves.
    """
    catalog = ZarrCatalog(var_key)
    return catalog.get_time_coverage()


class ZarrCatalog:
    """
    Facade over one variable's Zarr store: what it holds and how to read it.

    Combines a ``ZarrIndex`` (the Parquet-backed catalog of files, their
    extents, variables and dates — which is what makes a run resumable) with a
    ``ZarrReader`` (opening those files as one dataset, snapping drifted axes
    and padding ragged variable sets).

    Construct one per var_key. It is the read-side counterpart to
    ``write_append_zarr``, and :meth:`build_file_path` is what decides the
    filename a write lands in.
    """

    def __init__(
        self,
        var_key: str,
        *,
        file_period: FilePeriod = FilePeriod.YEAR,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        metadata_root: Optional[Path] = None,
        auto_refresh: bool = True,
        verbose: bool = False,
        warn_if_missing: bool = True,
    ) -> None:
        """
        Manages a Parquet catalog of processed zarr datasets.
        The catalog tracks metadata about zarr files including spatial extent,
        temporal coverage, variables, and file locations.

        Args:
            var_key (str): Variable key that must exist in app_config.variables
            file_period: Temporal granularity for file storage ('year' or 'month'). Defaults to 'year'.
            app_config (Optional[AppConfig], optional): Application configuration. If None, loads from get_settings().
            store_root (Optional[Path]): Root directory for zarr files. If None, uses get_settings().STORE_ROOT or get_settings().FALLBACK_STORE_ROOT.
            metadata_root (Optional[Path]): Root directory for catalog parquet files. If None, uses get_settings().METADATA_DIR.
            auto_refresh (bool): Automatically check for changes on access. Defaults to True.

        Raises:
            ValueError: If var_key not found in configuration or invalid period
            ValueError: If store_root doesn't exist

        Example:
            >>> catalog = ZarrCatalog("ssh")
        """
        # Load config
        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[var_key]

        self.file_period = validate_file_period(file_period)

        # Setup directories
        self.store_root = resolve_store_path(
            self.var_config, store_root, warn_if_missing=warn_if_missing
        )
        self.metadata_root = metadata_root or get_settings().METADATA_DIR

        # The catalog DataFrame and its refresh lifecycle live in ZarrIndex;
        # everything below either opens Zarr data or delegates to it.
        # auto_refresh/verbose are exposed as properties over the index rather
        # than copied here, so setting them on the facade cannot silently
        # diverge from the flags the catalog actually reads.
        self._index = ZarrIndex(
            self.var_key,
            self.var_config,
            store_root=self.store_root,
            metadata_root=self.metadata_root,
            file_period=self.file_period,
            auto_refresh=auto_refresh,
            verbose=verbose,
        )
        self._reader = ZarrReader(self._index)

    def __repr__(self) -> str:
        """String representation with useful info."""
        df = self.df
        time_cov = self.get_time_coverage()

        time_str = (
            f"{time_cov.start.date()} to {time_cov.end.date()}"
            if time_cov
            else "No data"
        )

        bbox = self.get_bbox()

        # Two different facts, and either one alone gets read as the other:
        # how far apart the data's steps are, versus how the store is cut into
        # files. An hourly store written one file per year is legitimately
        # "hourly" and "year" at once, so both are shown.
        step = getattr(self.var_config, "time_step", TimeStep.DAILY)

        return (
            f"ZarrCatalog(\n"
            f"  var_key={self.var_key},\n"
            f"  time_step={step.value},\n"
            f"  file_period={self.file_period.value},\n"
            f"  files={df['path'].nunique() if not df.empty else 0},\n"
            f"  coverage={time_str},\n"
            f"  bbox={bbox.to_label() if bbox is not None else None},\n"
            f")"
        )

    # ==================== Index delegation ====================
    # The catalog DataFrame, its refresh lifecycle and every query answered
    # from it live in ZarrIndex. These forwards keep the public surface
    # unchanged for the call sites that construct a ZarrCatalog.

    @property
    def auto_refresh(self) -> bool:
        """Whether each ``.df`` access re-checks the store for changes."""
        return self._index.auto_refresh

    @auto_refresh.setter
    def auto_refresh(self, value: bool) -> None:
        self._index.auto_refresh = value

    @property
    def verbose(self) -> bool:
        """Whether catalog bookkeeping logs at all."""
        return self._index.verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._index.verbose = value

    def _log(self, level: str, msg: str) -> None:
        """Log at *level* when verbose, silent otherwise."""
        self._index._log(level, msg)

    @property
    def catalog_path(self) -> Path:
        """Path to the catalog parquet file."""
        return self._index.catalog_path

    def exists(self) -> bool:
        """Check if catalog file exists."""
        return self._index.exists()

    def has_changes(self) -> bool:
        """Check if the store directory has changed since the last scan."""
        return self._index.has_changes()

    def refresh(self, force: bool = False) -> pd.DataFrame:
        """Refresh the catalog if needed. See :meth:`ZarrIndex.refresh`."""
        return self._index.refresh(force=force)

    @property
    def df(self) -> pd.DataFrame:
        """Catalog DataFrame. See :attr:`ZarrIndex.df`."""
        return self._index.df

    def reload(self) -> pd.DataFrame:
        """Force reload from disk or rescan."""
        return self._index.reload()

    def get_change_summary(self) -> dict[str, Any]:
        """Return a summary of added / removed / modified zarr files."""
        return self._index.get_change_summary()

    def map_dates_to_paths(self, dates: DatesLike) -> dict[str, list[pd.Timestamp]]:
        """Map zarr file paths to their corresponding dates."""
        return self._index.map_dates_to_paths(dates)

    def get_paths_in_range(
        self,
        start_date: DateLike,
        end_date: DateLike,
    ) -> list[str]:
        """Get all zarr file paths that overlap with a date range."""
        return self._index.get_paths_in_range(start_date, end_date)

    def open_dataset(
        self,
        dates: DatesLike | None = None,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
        bbox: BBox | Sequence[float] | None = None,
        variables: str | Sequence[str] | None = None,
        chunks: dict | str | None = "auto",
    ) -> xr.Dataset:
        """Open zarr dataset(s). See :meth:`ZarrReader.open_dataset`."""
        return self._reader.open_dataset(
            dates=dates,
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
            variables=variables,
            chunks=chunks,
        )

    def build_file_path(
        self,
        ds: xr.Dataset,
        date_format: Literal["year", "date", "yearmonth"],
        store_root: Optional[Path] = None,
        name_key: Optional[str] = None,
    ) -> Path:
        """
        Build zarr file path as: store_root / {source}_{name_key}_{geo_extent}_{dates}.zarr

        name_key defaults to var_key; pass dataset_id_rep (or any string) to override.
        If store_root not provided, uses self.store_root.
        """
        label = create_label_from_dataset(ds, date_format=date_format)
        store_root = store_root or self.store_root
        key = name_key if name_key is not None else self.var_key
        return store_root / f"{self.var_config.source}_{key}_{label}.zarr"

    # ==================== Helper Methods ====================
    # ==================== Metadata Queries ====================
    def get_var_time_coverage(self, var_name: str) -> DateRange | None:
        """Time coverage for zarr files that contain *var_name* as a data variable."""
        return self._index.get_var_time_coverage(var_name)

    def _scan_nonnull_edge(
        self, var_names: Sequence[str], *, newest_first: bool
    ) -> dict[str, pd.Timestamp]:
        """
        Outermost date each variable holds a non-null value, from one end.

        Shared by :meth:`get_vars_nonnull_start` and :meth:`get_vars_nonnull_end`.
        ``newest_first`` picks which end to walk in from, and files are visited
        in that order so the common case — the answer lies in the outermost
        file — opens exactly one of them.
        """
        df = self.df
        if df.empty or "variables" not in df.columns or "path" not in df.columns:
            return {}

        sort_col = "end_date" if newest_first else "start_date"
        files = df.sort_values(sort_col, ascending=not newest_first).drop_duplicates(
            subset="path"
        )
        remaining = set(var_names)
        result: dict[str, pd.Timestamp] = {}

        for _, row in files.iterrows():
            if not remaining:
                break
            file_vars = _variables_list(row["variables"])
            cols_here = [c for c in remaining if c in file_vars]
            if not cols_here:
                continue
            try:
                ds = xr.open_zarr(row["path"], consolidated=False)
            except Exception as e:
                logger.warning(f"Could not open {row['path']} for non-null scan: {e}")
                continue
            try:
                if "time" not in ds.coords:
                    continue
                # A column carried without a time axis (bathy, which is lon/lat
                # alone) has no edge to report: the reduction below would
                # collapse it to a scalar, leaving no date to attach. Dropped
                # from the scan rather than skipped over, so older files are not
                # reopened looking for an answer that cannot exist.
                for c in [c for c in cols_here if "time" not in ds[c].dims]:
                    remaining.discard(c)
                cols_here = [c for c in cols_here if "time" in ds[c].dims]
                if not cols_here:
                    continue

                times = pd.to_datetime(ds["time"].values)
                # Reduce each variable to a per-time "has any data" mask, batched
                # into one lazy compute so the file is read in a single pass.
                lazy = {
                    c: ds[c].notnull().any(dim=[d for d in ds[c].dims if d != "time"])
                    for c in cols_here
                }
                mask_ds = xr.Dataset(lazy).compute()
                for c in cols_here:
                    m = mask_ds[c].values
                    if m.any():
                        if newest_first:
                            edge = pd.Timestamp(times[m].max())
                        else:
                            edge = pd.Timestamp(times[m].min())
                        result[c] = edge.normalize()
                        remaining.discard(c)
            finally:
                ds.close()

        return result

    def get_vars_nonnull_end(self, var_names: Sequence[str]) -> dict[str, pd.Timestamp]:
        """
        Last date at which each variable holds a non-null value in the store.

        Unlike :meth:`get_var_time_coverage` — which reports the *file's* date
        span for any file listing the variable — this reflects where the variable
        actually has data. That distinction matters for compiled stores where a
        lagging variable is NaN-padded out to the global end by ``xr.merge``: the
        file end overstates its real coverage, whereas this method returns the
        last date with genuine data.

        Files are opened newest-first and scanning stops once every requested
        variable has been found, so the common case (all variables current)
        touches only the most recent file. Variables never found non-null —
        absent from the store, or carried without a time axis — are omitted from
        the result.

        Args:
            var_names: Variables to look up (typically one representative column
                per source var_key).

        Returns:
            Mapping of variable name to its last non-null date (normalised to
            midnight). Empty when the store has no data.
        """
        return self._scan_nonnull_edge(var_names, newest_first=True)

    def get_vars_nonnull_start(
        self, var_names: Sequence[str]
    ) -> dict[str, pd.Timestamp]:
        """
        First date at which each variable holds a non-null value in the store.

        The leading-edge mirror of :meth:`get_vars_nonnull_end`, and needed for
        the same reason: ``xr.merge`` pads a variable to the union axis at both
        ends, so a source whose archive starts later than the store's is
        present-but-null over the stretch before it begins.

        Files are opened oldest-first, otherwise identical. Same omissions.

        Args:
            var_names: Variables to look up.

        Returns:
            Mapping of variable name to its first non-null date (normalised to
            midnight). Empty when the store has no data.
        """
        return self._scan_nonnull_edge(var_names, newest_first=False)

    def get_var_coverage(self, var_name: str) -> DateRange | None:
        """
        Dates *var_name* really holds data for, not the span of the files carrying it.

        The two differ on a compiled store. ``xr.merge(join="outer")`` pads every
        variable out to the union time axis, so a source lagging the furthest-ahead
        one by two months is present-but-null across that stretch and
        :meth:`get_time_coverage` — which reports the union — calls it covered.
        Reading that as coverage turns "the compile has not reached here yet"
        into "the value is missing at this point", and those have different
        fixes.

        The file span comes first (which files list the variable at all), then
        both ends are narrowed to where the values are real. A variable with no
        time axis, or one whose scan finds nothing non-null, keeps the file
        span — the honest answer when there is no frontier to measure.

        Costs two edge scans, each opening one file in the common case. Callers
        needing this for many variables at once should use
        :meth:`get_vars_nonnull_start` / :meth:`get_vars_nonnull_end` directly,
        which answer for a whole batch in the same pass.

        Args:
            var_name: Variable to report on.

        Returns:
            The variable's own DateRange, or None when no file lists it.
        """
        span = self.get_var_time_coverage(var_name)
        if span is None:
            return None

        start = self.get_vars_nonnull_start([var_name]).get(var_name, span.start)
        end = self.get_vars_nonnull_end([var_name]).get(var_name, span.end)
        return DateRange(start=start, end=end)

    def get_nonnull_days(
        self,
        window: DateRange,
        var_names: Optional[Sequence[str]] = None,
    ) -> dict[str, pd.DatetimeIndex]:
        """
        Dates within *window* where each variable holds non-null data.

        The set-valued counterpart to :meth:`get_vars_nonnull_end`. That method
        answers "how far has this variable got", which cannot see a null day
        *inside* an otherwise-compiled span — the frontier moves past it and
        nothing asks again. This answers "which days does it actually have".

        Bounded by *window* on purpose: it reads values, so an unbounded call
        would walk the whole store. Callers pass a lookback.

        Args:
            window: Date range to inspect.
            var_names: Variables to look up. ``None`` reduces across every data
                variable in each file and returns the result under the single
                key ``"__any__"`` — "does this store hold anything at all on
                this date", which is what distinguishes a fillable lag hole
                from a gap the source itself has.

        Returns:
            Mapping of variable name to the dates it has data for. Variables
            never found are omitted.
        """
        df = self.df
        if df.empty or "path" not in df.columns:
            return {}

        rows = df.drop_duplicates(subset="path")
        if {"start_date", "end_date"} <= set(rows.columns):
            rows = rows[
                (rows["end_date"] >= window.start) & (rows["start_date"] <= window.end)
            ]

        found: dict[str, list[pd.DatetimeIndex]] = {}
        for _, row in rows.iterrows():
            if var_names is not None:
                file_vars = _variables_list(row.get("variables"))
                if not [c for c in var_names if c in file_vars]:
                    continue
            try:
                ds = xr.open_zarr(row["path"], consolidated=False)
            except Exception as e:
                logger.warning(f"Could not open {row['path']} for non-null scan: {e}")
                continue
            try:
                if "time" not in ds.coords:
                    continue
                # end_of_day, not window.end: the bound is a date, and a store
                # whose stamps are not at midnight would lose its last day —
                # every sub-daily step after 00:00 on an hourly store, and the
                # whole day on a noon-stamped daily one.
                ds = ds.sel(time=slice(window.start, end_of_day(window.end)))
                times = pd.DatetimeIndex(ds["time"].values).normalize()
                if len(times) == 0:
                    continue

                cols = (
                    [c for c in var_names if c in ds.data_vars]
                    if var_names is not None
                    else list(ds.data_vars)
                )
                lazy = {
                    str(c): ds[c]
                    .notnull()
                    .any(dim=[d for d in ds[c].dims if d != "time"])
                    for c in cols
                    if "time" in ds[c].dims
                }
                if not lazy:
                    continue
                mask_ds = xr.Dataset(lazy).compute()

                if var_names is None:
                    any_mask = np.zeros(len(times), dtype=bool)
                    for c in lazy:
                        any_mask |= np.asarray(mask_ds[c].values, dtype=bool)
                    found.setdefault("__any__", []).append(times[any_mask])
                else:
                    for c in lazy:
                        m = np.asarray(mask_ds[c].values, dtype=bool)
                        if m.any():
                            found.setdefault(c, []).append(times[m])
            finally:
                ds.close()

        return {
            name: pd.DatetimeIndex(np.concatenate([p.values for p in parts]))
            .unique()
            .sort_values()
            for name, parts in found.items()
            if parts
        }

    def get_variables(self) -> set[str]:
        """Get all unique variables across all zarr files."""
        return self._index.get_variables()

    def get_time_coverage(self) -> DateRange | None:
        """Get overall time coverage across all files."""
        return self._index.get_time_coverage()

    def get_store_bbox(self) -> BBox | None:
        """
        Geographic extent the store actually holds, unioned across its files.

        Unlike :meth:`get_bbox` — which reports the bbox *requested* in
        config.yaml — this comes from the lon/lat bounds scanned out of each
        Zarr file, so it shows what was really written (a source delivering a
        wider or coarser grid than requested shows up here).
        """
        return self._index.get_store_bbox()

    def get_bbox(self) -> BBox | None:
        """
        Get the configured geographic extent (bbox) for this variable.

        This is the requested extent from config.yaml, not what the files
        contain — see :meth:`get_store_bbox` for that.
        """
        bbox = self.var_config.bbox
        if bbox is None:
            return None
        if not isinstance(bbox, BBox):
            return BBox.from_tuple(bbox)
        return bbox

    def summary(self) -> dict:
        """
        Get summary statistics about the catalog.

        Returns:
            Dictionary with catalog statistics
        """
        df = self.df

        if df.empty:
            return {
                "num_files": 0,
                "time_coverage": None,
                "bbox": "No data",
                "store_bbox": None,
                "period": self.file_period,
                "variables": set(),
                "total_timesteps": 0,
                "store_root": str(self.store_root),
                "catalog_path": str(self.catalog_path),
                "last_scanned": None,
            }

        time_cov = self.get_time_coverage()
        bbox = self.get_bbox()

        return {
            "num_files": df["path"].nunique(),
            "time_coverage": time_cov if time_cov is not None else "No data",
            "bbox": bbox if bbox is not None else "No data",
            "store_bbox": self.get_store_bbox(),
            "period": self.file_period,
            "variables": self.get_variables(),
            "total_timesteps": (
                df["num_timesteps"].sum().item()
                if "num_timesteps" in df.columns
                else None
            ),
            "store_root": str(self.store_root),
            "catalog_path": str(self.catalog_path),
            "last_scanned": (
                df["scanned_at"].max() if "scanned_at" in df.columns else None
            ),
        }

    # ==================== Migration Helpers ====================

    def refresh_provenance(self) -> int:
        """
        Rebuild every stored file's ``source_datasets`` from its own time axis,
        repairing records left narrower than the file they describe. Returns the
        number of files rewritten. See
        :func:`h2mare.storage.provenance.refresh_provenance`.
        """
        from h2mare.storage.provenance import refresh_provenance

        return refresh_provenance(self)

    def backfill_provenance(self, rep_end_date: DateLike) -> int:
        """
        One-time migration: retroactively write provenance for Zarr files
        that pre-date automatic tracking by Netcdf2Zarr. Returns the number
        of files updated. See :func:`h2mare.storage.provenance.backfill_provenance`.
        """
        from h2mare.storage.provenance import backfill_provenance

        return backfill_provenance(self, rep_end_date)
