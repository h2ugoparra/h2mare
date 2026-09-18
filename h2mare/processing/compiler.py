"""
Create h2ds zarr files
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Literal, Optional

import ephem
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.models import SYSTEM_VAR_KEYS, depth_levels_for
from h2mare.storage.coverage import (
    resolve_date_range,
    split_time_range,
)
from h2mare.storage.provenance import (
    collect_source_datasets,
    refresh_root_attrs,
    write_compiled_provenance,
)
from h2mare.storage.recovery import recover_zarr_store
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import (
    apply_cf_attrs,
    check_grid_compatible,
    chunk_dataset,
)
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox, DateLike, DateRange, FilePeriod
from h2mare.utils.datetime_utils import normalize_date
from h2mare.utils.paths import store_root_for
from h2mare.utils.spatial import GridBuilder
from h2mare.validators import validate_file_period, validate_var_key

#: Grid used when the compiled var_key's config entry declares none: 4 cells
#: per degree (0.25°), cell-centred. What every existing h2ds store is on.
DEFAULT_CELLS_PER_DEGREE = 4

# How far back an incremental compile looks for a null day it could refill.
# The check reads values, so it has to be bounded or every compile would walk
# the whole store; 24 months covers damage from a run that failed recently,
# which is the case worth automating. Older holes surface through
# ``h2mare audit`` and are repaired with explicit dates, which bypass every
# watermark. The Parquet layer runs the same idea at 400 days
# (``zarr2parquet._BACKFILL_HOLE_LOOKBACK_DAYS``) over a cheaper store.
_HOLE_LOOKBACK_DAYS = 730


def calculate_moon_phase(
    lat: float, lon: float, dates: pd.DatetimeIndex
) -> list[float]:
    """
    Calculate moon ilumination using ephem library

    Args:
        lat (float): latitude of observation
        lon (float): longitude of observation
        dates (pd.DatetimeIndex): time index values for extraction

    Returns:
        list[float]: list of lunar ilumination values for each date values
    """
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)

    phases = []
    for date in dates:
        observer.date = date
        moon = ephem.Moon(observer)
        phases.append(moon.phase)
    return phases


def postprocess_sst_fdist(ds: xr.Dataset, var_name: str = "sst_fdist") -> xr.Dataset:
    """
    Clip sst_fdist data because interp_like gives negative values
    """
    if var_name in ds:
        ds[var_name] = ds[var_name].clip(min=0)
    return ds


class Compiler:
    """
    Merges the per-variable Zarr stores into the compiled product (``h2ds``).

    Each variable is read from its own native store, interpolated onto the
    common 0.25° daily grid, and merged into one dataset written per period
    (a file per year by default). What a given var_key contributes is decided
    by ``compiler_registry.COMPILE_PROCESSORS``; anything unregistered goes
    through ``compile_default``.

    Compiling a subset (``run(var_keys=[...])``) writes only those variables'
    columns. The rest are preserved rather than nulled — see
    ``storage._append_data`` — and catch up on the next full compile.
    """

    def __init__(
        self,
        var_key: str = "h2ds",
        app_config: Optional[AppConfig] = None,
        remote_store_root: Optional[Path] = None,
        local_store_root: Optional[Path] = None,
        file_period: FilePeriod = FilePeriod.YEAR,
        date_format: Literal["year", "date", "yearmonth"] = "year",
    ):
        """
        Class function to compile zarr files from each var_key to a pre defined spatial res (set at 0.25) grid with daily interpolated data.

        Args:
            var_key (str, optional): Var key name of compiled data. Defaults to 'h2ds'.
            app_config (AppConfig, optional): Configuration data for var keys. Defaults to AppConfig.
            remote_store_root (Path, optional): Store directory where all environmental data lives (currently D:).
            local_store_root (Path], optional): Local data directory where compiled data lives (currently C:)
            file_period: Temporal granularity ('year' or 'month') for file storage. Defaults to 'year'.
            date_format: string date format for output file name.
        """
        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[self.var_key]

        self.local_store_root = (
            local_store_root or get_settings().ZARR_DIR / self.var_config.local_folder
        )
        resolved_remote = remote_store_root or get_settings().STORE_ROOT
        if resolved_remote is None:
            raise ValueError(
                "remote_store_root must be provided or STORE_ROOT must be set in the environment"
            )
        self.remote_store_root: Path = resolved_remote

        if self.var_config.bbox is None:
            raise ValueError(
                f"var_key '{self.var_key}' config must have a bbox for compilation"
            )
        self.bbox = BBox.from_tuple(self.var_config.bbox)

        self.file_period = validate_file_period(file_period)
        self.date_format: Literal["year", "date", "yearmonth"] = date_format

        self.catalog = self._catalog_for(self.var_key)
        # Cache of per-variable non-null end dates in h2ds, filled lazily on the
        # first incremental range resolution (one store scan per run).
        self._nonnull_ends_cache: Optional[dict[str, pd.Timestamp]] = None
        # Per-variable source catalogs, built once and reused across all time
        # chunks in a run. A source store's contents are stable during a
        # compile (only the h2ds output is written), so re-scanning it per
        # chunk is wasted I/O.
        self._catalog_cache: dict[str, ZarrCatalog] = {}

    def _catalog_for(self, var_key: str, **kwargs) -> ZarrCatalog:
        """
        Catalog for *var_key*, rooted under this compiler's ``remote_store_root``.

        Built explicitly rather than left to resolve from settings so that a
        relocated store root reaches the compiler's own reads. Left to default
        to ``STORE_ROOT``, a run pointed elsewhere would write h2ds to the
        override while reading its sources from the configured root.

        ``remote_store_root`` is the *default* root, not the final answer — a
        source variable may name its own in config.yaml, and a compile has to
        read each one where it actually lives. Variables naming none still
        resolve under ``remote_store_root``, exactly as before.
        """
        var_config = self.app_config.variables[var_key]
        root = store_root_for(var_config, self.remote_store_root)
        return ZarrCatalog(
            var_key,
            store_root=root / var_config.local_folder,
            **kwargs,
        )

    def run(
        self,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        var_keys: Optional[list[str]] = None,
        zarr_backup: bool = False,
        zarr_backup_dir: Optional[Path] = None,
    ) -> None:
        """
        Main entry point for the h2ds compilation process.

        Without explicit dates, uses incremental mode: reads the h2ds catalog
        end date and compiles ``[h2ds_end + 1 day → today]``.  To backfill a
        historical range (e.g. when adding a new variable), pass explicit
        *start_date* / *end_date*.

        Args:
            start_date: Start of the compilation window. If omitted, inferred
                from the store (see above).
            end_date: End of the compilation window. If omitted, inferred from
                the store (see above).
            var_keys: Variable keys to include. ``None`` compiles all configured
                variables (incremental mode).
            zarr_backup: Copy written zarr files to the local store. Defaults to False.
            zarr_backup_dir: Override destination for the zarr backup. Defaults to local_store_root.
        """
        t0 = time.perf_counter()
        logger.info(
            f"Initializing Zarr compilation for variable key: {self.var_key.upper()}"
        )

        # Reconcile any interrupted h2ds write before gap detection reads the
        # store (restore a stranded backup, discard a half-written temp).
        recover_zarr_store(self.catalog.store_root)

        start = normalize_date(start_date) if start_date else None
        end = normalize_date(end_date) if end_date else None

        self.var_keys = (
            [var_keys]
            if isinstance(var_keys, str)
            else var_keys or sorted(self.app_config.variables.keys())
        )

        # Source coverage for each variable — computed once
        self._source_coverage = self._compute_source_coverage()

        requested_range = self._resolve_compile_range(start, end)
        if requested_range is None:
            logger.info(
                "All configured variables are up to date — nothing to compile. "
                "Pass explicit --start-date / --end-date to force a range."
            )
            return

        self.base_grid = self._build_base_grid()
        self._check_store_grid()

        # time chunks
        chunks = split_time_range(requested_range, self.file_period)

        written_paths: list[Path] = []

        for i, chunk in enumerate(chunks, 1):
            logger.debug(
                f"Chunk {i}/{len(chunks)}: {chunk.start.date()} -> {chunk.end.date()}"
            )

            datasets = []
            # var_key -> the source datasets that delivered this chunk's dates.
            # Collected per chunk because a variable can switch from rep to nrt
            # part-way through the archive, so it is not a property of the run.
            provenance: dict[str, list[dict]] = {}

            for vkey in self.var_keys:
                if vkey == self.var_key:
                    continue

                with logger.contextualize(var=vkey):
                    ds = self._process_variable(vkey, chunk)
                if ds is not None:
                    datasets.append(ds)
                    # System variables (bathy, moon) have no store and no
                    # catalog, so they contribute no provenance.
                    if (catalog := self._catalog_cache.get(vkey)) is not None:
                        if records := collect_source_datasets(catalog, chunk):
                            provenance[vkey] = records

            if not datasets:
                logger.warning(
                    f"No datasets collected for chunk {i}: "
                    f"Period: {chunk.start} -> {chunk.end} — Skipping."
                )
                continue

            ds_final = xr.merge(datasets, join="outer")
            assert isinstance(ds_final, xr.Dataset)
            ds_final = chunk_dataset(ds_final)
            ds_final = self._set_attrs(ds_final)

            path = self.catalog.build_file_path(
                ds_final, self.date_format, name_key=self.var_config.dataset_id_rep
            )
            write_append_zarr(self.var_key, ds_final, path)
            written_paths.append(path)

            try:
                refresh_root_attrs(path, get_settings().global_attrs)
            except Exception as e:
                logger.warning(f"Could not refresh root attrs on {path.name}: {e}")

            if provenance:
                try:
                    write_compiled_provenance(path, provenance)
                except Exception as e:
                    # Bookkeeping about a compile must not fail the compile that
                    # already succeeded — the data is on disk by this point.
                    logger.warning(
                        f"Could not write compiled provenance for {path.name}: {e}"
                    )

            logger.success(
                f"Finished period {chunk.start.date()} -> {chunk.end.date()}"
            )

        self.catalog.refresh()

        if zarr_backup:
            # Backup all written files to local store in one pass — avoids repeated
            # large directory copies after each individual chunk
            for path in written_paths:
                self.sync_data(path, backup_dir=zarr_backup_dir)

        logger.success(
            f"Compile complete: {len(written_paths)}/{len(chunks)} chunk(s) written "
            f"({requested_range.start.date()} → {requested_range.end.date()}) "
            f"in {time.perf_counter() - t0:.1f}s"
        )

    def _build_base_grid(self) -> xr.Dataset:
        """
        The grid this compile writes on, from the compiled var_key's config.

        Declared as a whole number of cells per degree and a registration, so
        the resolution is exact and the phase is deliberate. An entry naming
        neither gets 0.25° cell-centred, which is what every existing h2ds
        store holds.
        """
        cells = self.var_config.cells_per_degree or DEFAULT_CELLS_PER_DEGREE
        registration = self.var_config.registration
        step = 1 / cells
        logger.info(
            f"Base grid: 1/{cells}° ({step:.6g}°), {registration}-registered, "
            f"over {self.bbox}"
        )
        return GridBuilder(
            self.bbox, step, step, registration=registration
        ).generate_grid()

    def _check_store_grid(self) -> None:
        """
        Refuse a run whose base grid is not the one the store already holds.

        The write path would merge a different grid into the store rather than
        reject it — every variable NaN at the other grid's cells.
        :func:`check_grid_compatible` is the check that catches it, and it runs
        there too; doing it here as well turns a changed ``cells_per_degree``
        or ``registration`` into a failure before the first chunk is read
        rather than after one has been computed.
        """
        existing = sorted(self.catalog.store_root.glob("*.zarr"))
        if not existing:
            return
        with xr.open_zarr(existing[-1], consolidated=False) as stored:
            try:
                check_grid_compatible(stored, self.base_grid)
            except ValueError as e:
                raise ValueError(f"{existing[-1].name}: {e}") from None

    # =========== DATE RANGE RESOLUTION ===========
    def _compute_source_coverage(self) -> dict[str, DateRange]:
        """
        Return source catalog coverage for every non-system source variable.

        Read through this compiler's own catalogs. The module-level
        ``get_store_coverage`` builds a rootless ``ZarrCatalog``, resolving from
        settings alone, so the compile *window* was computed from a different
        root than the data was *read* from whenever the two disagreed — a
        ``remote_store_root`` passed in, or a variable naming its own
        ``store_root``. The mismatch never raised: it surfaced as "No source
        coverage found — skipping", dropping a variable from the compile with a
        warning that reads like an empty store.
        """
        result: dict[str, DateRange] = {}
        for vkey in self.var_keys:
            if vkey == self.var_key or vkey in SYSTEM_VAR_KEYS:
                continue
            cov = self._read_source_coverage(vkey)
            if cov is not None:
                result[vkey] = cov
            else:
                logger.warning(f"No source coverage found for '{vkey}' — skipping.")
        return result

    def _read_source_coverage(self, var_key: str) -> Optional[DateRange]:
        """
        Coverage of *var_key*'s own store, or None when it cannot be read.

        Named apart from the ``_source_coverage`` dict this feeds — an instance
        attribute of that name would otherwise shadow the method.

        Shares ``_catalog_cache`` with the compile loop, so the scan this does
        is the one that loop would have done anyway.
        """
        try:
            catalog = self._catalog_cache.setdefault(
                var_key, self._catalog_for(var_key, auto_refresh=False)
            )
            return catalog.get_time_coverage()
        except Exception as e:
            logger.warning(f"Could not read store coverage for '{var_key}': {e}")
            return None

    def _h2ds_nonnull_ends(self) -> dict[str, pd.Timestamp]:
        """
        Last non-null date in h2ds for each source var_key's representative column.

        Computed once per run and cached. The representative is
        ``compiled_vars[0]``; by config convention all of a var_key's
        compiled columns share the same dates, so one column dates the group.
        Uses :meth:`ZarrCatalog.get_vars_nonnull_end` (a single newest-first
        scan) so the whole set costs one pass over the most recent h2ds file.
        """
        if self._nonnull_ends_cache is None:
            reps: list[str] = []
            for vk in self._source_coverage:
                cols = self.app_config.variables[vk].compiled_vars or []
                if cols:
                    reps.append(cols[0])
            self._nonnull_ends_cache = self.catalog.get_vars_nonnull_end(
                sorted(set(reps))
            )
        return self._nonnull_ends_cache

    def _get_h2ds_var_end(self, vkey: str) -> Optional[pd.Timestamp]:
        """
        Last compiled date of *vkey* in h2ds, measured by real (non-null) data.

        Keyed off the compiled column names (``compiled_vars``), not the
        raw source variable names, and uses non-null coverage rather than the
        h2ds file end. This is what lets a lagging variable (e.g. eddies, whose
        columns are NaN-padded out to the global h2ds end) be detected as behind
        and backfilled when its source advances — even within the already-written
        date range.

        Returns:
            The variable's last non-null date in h2ds; the file-level end as a
            conservative fallback if the column exists but the non-null scan
            found nothing; or ``None`` when the column is absent from h2ds
            (never compiled → caller backfills from the source start).
        """
        cols = self.app_config.variables[vkey].compiled_vars or []
        if not cols:
            return None
        rep = cols[0]

        end = self._h2ds_nonnull_ends().get(rep)
        if end is not None:
            return end

        # Column present in h2ds but no non-null data found: stay conservative
        # (file end) rather than forcing a full recompile from the source start.
        cov = self.catalog.get_var_time_coverage(rep)
        return cov.end if cov is not None else None

    def _fillable_hole_start(
        self, vkey: str, src_cov: DateRange
    ) -> Optional[pd.Timestamp]:
        """
        Earliest day inside the lookback where the source has data and h2ds does not.

        Two conditions, and both matter:

        * **h2ds is null there.** The compiled column has no usable value for
          that day, whether the row is missing or NaN-padded.
        * **the source has data there.** A day the source is null for cannot be
          filled by recompiling it, and treating it as a hole would re-merge the
          same window on every run without ever converging. chl's legitimate
          all-null days are precisely this case, and they must not trigger work.

        Bounded by :data:`_HOLE_LOOKBACK_DAYS` because it reads values. Older
        holes are reported by ``h2mare audit`` and repaired with explicit
        ``--start-date``/``--end-date``, which force a full recompile of the
        range regardless of what the watermarks say.

        Returns:
            The earliest fillable date, or ``None`` when there is nothing to
            backfill (the overwhelmingly common case).
        """
        var_config = self.app_config.variables.get(vkey)
        cols = (var_config.compiled_vars if var_config else None) or []
        if not cols:
            return None
        rep = cols[0]

        end = min(src_cov.end, pd.Timestamp.now().normalize())
        floor = max(src_cov.start, end - pd.Timedelta(days=_HOLE_LOOKBACK_DAYS))
        if floor > end:
            return None
        window = DateRange(start=floor, end=end)

        try:
            h2ds_have = self.catalog.get_nonnull_days(window, [rep]).get(
                rep, pd.DatetimeIndex([])
            )
            null_days = pd.date_range(floor, end, freq="D").difference(h2ds_have)
            if len(null_days) == 0:
                return None

            source_have = self._catalog_for(vkey, auto_refresh=False).get_nonnull_days(
                window
            )
            fillable = null_days.intersection(
                source_have.get("__any__", pd.DatetimeIndex([]))
            )
        except Exception as e:
            logger.warning(f"{vkey}: hole scan failed ({e}) — skipping backfill check.")
            return None

        if len(fillable) == 0:
            return None
        return pd.Timestamp(fillable.min())

    def _resolve_compile_range(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Optional[DateRange]:
        """
        Resolve the compilation date range.

        **Explicit dates** always win and are passed straight through to
        :func:`resolve_date_range`.

        **Per-variable incremental mode** (no explicit dates): for every source
        variable, the gap between its last compiled end date in h2ds (queried
        directly via :meth:`_get_h2ds_var_end`) and its current source catalog
        end is computed.  The union of all per-variable gaps becomes the
        compilation window, so a lagging variable does not hold back faster
        ones and catches up automatically when new source data arrives.

        Args:
            start: Normalised start timestamp, or ``None`` to infer.
            end: Normalised end timestamp, or ``None`` to infer.

        Returns:
            Resolved :class:`DateRange`, or ``None`` when every variable is
            already up to date in incremental mode (a clean no-op — the caller
            should skip compilation rather than treat it as an error).

        Raises:
            ValueError: If explicit dates are given but invalid (start > end).
        """
        if start is not None or end is not None:
            result = resolve_date_range(self.var_key, start, end)
            if result is None:
                raise ValueError(f"Invalid date range: start ({start}) > end ({end})")
            return result

        ranges: list[DateRange] = []
        compiling: list[str] = []
        up_to_date: list[str] = []
        for vkey, src_cov in self._source_coverage.items():
            h2ds_var_end = self._get_h2ds_var_end(vkey)
            var_start = (
                h2ds_var_end + pd.Timedelta(days=1)
                if h2ds_var_end is not None
                else src_cov.start
            )

            # The end alone cannot see a hole behind it. A variable compiled
            # while its source lagged lands NaN-padded; once a later compile
            # carries it, the non-null end jumps past the NaN stretch and an
            # end-based window strands those days forever. sst 2026-07-31 is
            # exactly this — h2ds has the day, the column is null, and the end
            # sits at 2026-08-06.
            hole = self._fillable_hole_start(vkey, src_cov)
            if hole is not None and hole < var_start:
                logger.info(
                    f"{vkey}: backfilling from {hole.date()} — source has data "
                    f"for days h2ds does not."
                )
                var_start = hole

            if var_start <= src_cov.end:
                ranges.append(DateRange(start=var_start, end=src_cov.end))
                compiling.append(f"{vkey} ({var_start.date()}→{src_cov.end.date()})")
            else:
                up_to_date.append(vkey)

        # One summary line each instead of a line per variable.
        if up_to_date:
            logger.debug(f"Up to date, skipping: {', '.join(sorted(up_to_date))}")
        if compiling:
            logger.debug(f"Compiling: {', '.join(compiling)}")

        if not ranges:
            # Benign no-op: nothing new to compile. Signal with None so the
            # caller can skip cleanly instead of raising (a scheduled run with
            # no new source data is success, not failure).
            return None

        inferred = DateRange(
            start=min(r.start for r in ranges),
            end=max(r.end for r in ranges),
        )
        logger.info(
            f"Per-variable incremental range: "
            f"{inferred.start.date()} → {inferred.end.date()}"
        )
        return inferred

    # =========== DATASET ATTRIBUTES ===========
    def _set_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Set global and variables attributes from yaml file.

        Variable and coordinate metadata comes from ``apply_cf_attrs``, shared
        with the convert path so a native store and h2ds cannot describe the
        same quantity differently. No ``native_var_key`` here: this is the
        compiled product, which takes the table as written.

        Args:
            ds: Dataset for atts assignment
        """
        ds.attrs = get_settings().global_attrs
        return apply_cf_attrs(ds)

    #  ============ PROCESSING ==================
    def _process_variable(
        self,
        var_key: str,
        date_range: DateRange,
    ) -> Optional[xr.Dataset]:
        """Dispatch var_key to its registered compile processor (or the default)."""
        # Lazy import breaks the compiler.py ↔ compiler_registry.py cycle.
        from h2mare.processing.compiler_registry import (
            COMPILE_PROCESSORS,
            _compile_depth_var,
            compile_default,
        )

        # System variables (bathy, moon) are generated rather than read from a
        # store, so they get no catalog and skip the overlap check.
        # auto_refresh=False: the source store is stable during a compile (see
        # _catalog_cache note in __init__), so skip the per-access change check
        # that would otherwise re-stat the store directory on every df access.
        catalog: Optional[ZarrCatalog] = None
        if var_key not in SYSTEM_VAR_KEYS:
            catalog = self._catalog_cache.setdefault(
                var_key, self._catalog_for(var_key, auto_refresh=False)
            )
            if not self._has_overlap(var_key, date_range, catalog):
                return None

        processor = COMPILE_PROCESSORS.get(var_key)
        if processor is None:
            # Depth handling follows the config, not the name, so a new 3-D
            # var_key needs no registry entry.
            has_levels = depth_levels_for(
                var_key, self.app_config.variables.get(var_key)
            )
            processor = _compile_depth_var if has_levels else compile_default
        return processor(self, catalog, date_range)

    # ============== UTILITIES ===================
    def _has_overlap(
        self, var_key: str, date_range: DateRange, catalog: ZarrCatalog
    ) -> bool:
        """Check for temporal overlap between requested date_range and catalog date_range"""
        env_daterange = catalog.get_time_coverage()

        if env_daterange:
            if date_range.overlaps(env_daterange):
                return True
            else:
                # Expected during incremental backfill: vars already up to date
                # don't overlap the union window. Not a warning.
                logger.debug(f"Skipping {var_key}: dates out of range.")
                return False
        return False

    def sync_data(self, remote_path: Path, backup_dir: Optional[Path] = None) -> None:
        """
        Copy a compiled zarr file to the local backup store.

        Args:
            remote_path: path built by the caller via ``ZarrCatalog.build_file_path()``
            backup_dir: destination directory; defaults to ``local_store_root``.
        """
        local_path = (backup_dir or self.local_store_root) / remote_path.name

        logger.info(f"Copying {remote_path} to {local_path}")

        try:
            shutil.copytree(remote_path, local_path, dirs_exist_ok=True)
        except (PermissionError, OSError) as e:
            # Return rather than fall through: the success line below sits
            # outside this handler, so falling through would log a failed backup
            # as an error and then announce "File copied!" on the next line.
            logger.exception(f"Failed to copy {remote_path} to {local_path}: {e}")
            return

        logger.success("File copied!")
