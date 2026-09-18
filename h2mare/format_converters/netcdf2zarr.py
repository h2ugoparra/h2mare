"""
Process downloaded netcdf/grib raw data to zarr files
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.format_converters.base import BaseConverter
from h2mare.models import StoreDtype, step_freq
from h2mare.processing.derived import apply_derived_vars
from h2mare.processing.registry import PROCESSORS
from h2mare.storage.audit import format_date_blocks, known_gap_days
from h2mare.storage.provenance import (
    annotate_covered,
    merge_records,
    read_store_dates,
    write_cf_root_attrs,
)
from h2mare.storage.recovery import recover_zarr_store
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import (
    apply_cf_attrs,
    chunk_dataset,
    int16_encoding,
    rename_dims,
    snap_grid_coords,
)
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import DateLike, DateRange, FilePeriod
from h2mare.utils.datetime_utils import normalize_date
from h2mare.utils.files_io import filter_raw_files, safe_move_files, safe_rmtree
from h2mare.utils.paths import resolve_download_path
from h2mare.validators import validate_file_period, validate_var_key


def _close_all(*datasets: Optional[xr.Dataset]) -> None:
    """Close every given dataset, ignoring ``None`` and already-closed ones."""
    for ds in datasets:
        if ds is None:
            continue
        try:
            ds.close()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Ignoring error while closing dataset: {e}")


def _dataset_dates(ds: xr.Dataset, freq: str = "D") -> pd.DatetimeIndex:
    """
    Unique time steps on a dataset's axis, sorted. Empty if time-less.

    A daily axis is normalized first, so a product publishing at 12:00 counts as
    its calendar day. An hourly axis is left as-is: normalizing it would collapse
    24 steps onto one and make the date checks blind to a missing hour.
    """
    if "time" not in ds.coords:
        return pd.DatetimeIndex([])
    times = pd.DatetimeIndex(ds.time.values)
    if freq == "D":
        times = times.normalize()
    return times.unique().sort_values()


def _span_end(end: DateLike, freq: str) -> pd.Timestamp:
    """
    Last step of *end*'s day at *freq*.

    Manifest spans are date-level, so on an hourly cadence a range ending at the
    end date's midnight would expect one step of the final day and report the
    other 23 as undelivered.

    Both sources of *end* (a DateRange bound and a manifest date string) are
    already at midnight, so the normalize is a no-op that only pins the type.
    """
    ts = normalize_date(end)
    return ts if freq == "D" else ts + pd.Timedelta(hours=23)


def convert_netcdf_to_zarr(
    paths: Path | str | Iterable[Path | str],
    out_path: Path | str,
    *,
    name: str = "data",
    processor: Callable[[xr.Dataset], xr.Dataset] | None = None,
    apply_rename: bool = True,
    open_kwargs: Optional[dict] = None,
) -> Path:
    """
    Convert one or more NetCDF/GRIB files to a single Zarr store, without a
    configured ``var_key``.

    This is the config-free counterpart to :class:`Netcdf2Zarr`. It applies the
    same generic Zarr-prep the pipeline uses — open → (optional ``rename_dims``)
    → (optional ``processor``) → ``snap_grid_coords`` → ``chunk_dataset`` →
    ``write_append_zarr`` — but driven entirely by arguments instead of
    ``config.yaml``. Use it to convert arbitrary files (a single path or a list)
    that are not registered as a variable.

    Args:
        paths: One path, or an iterable of paths, to ``.nc``/``.grib`` files.
            NetCDF and GRIB may be mixed; the engine is auto-detected from the
            first file.
        out_path: Destination ``.zarr`` path. If it already exists, the data is
            appended using the store's standard overlap semantics.
        name: Identity label used in the write/append logs and overlap
            resolution. It need **not** exist in config.
        processor: Optional callable applied after rename and before snap/chunk —
            the same slot a registry processor occupies in
            ``Netcdf2Zarr.process_dataset``. To reuse a registered processor,
            wrap it: ``processor=lambda ds: PROCESSORS["sst"](ds, cfg, "sst")``.
        apply_rename: Apply ``rename_dims`` (``longitude/latitude/valid_time`` →
            ``lon/lat/time``). Set ``False`` when the files already use canonical
            dim names.
        open_kwargs: Extra keyword arguments forwarded to ``xr.open_mfdataset``.

    Returns:
        The ``out_path`` that was written.

    Raises:
        FileNotFoundError: If no input paths are given.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = sorted(Path(p) for p in paths)
    if not files:
        raise FileNotFoundError("convert_netcdf_to_zarr: no input files given")

    out_path = Path(out_path)
    engine = "cfgrib" if files[0].suffix.lower() in {".grib", ".grb"} else "netcdf4"

    logger.info(
        f"Converting {len(files)} file(s) → {out_path.name} "
        f"(engine={engine}, name={name!r})"
    )

    # Explicit, for the reason ZarrReader pins it: xarray's default moves from
    # data_vars="all" to data_vars=None, which resolves to "minimal" whenever
    # the concat dim is present — as `time` always is here. The two differ only
    # for a data variable carrying no time dim (AVISO's raw files ship `crs`,
    # `lat_bnds`, `lon_bnds`): "all" broadcasts it along time, "minimal" leaves
    # it alone.
    #
    # This engine applies only the `processor` its caller passes, which is
    # often none, so here that difference reaches the store directly. "all" is
    # the current behaviour, pinned so the flip is a no-op. Switching to
    # "minimal" is a deliberate call, not a default's to make, and it cascades
    # a second FutureWarning for `compat`.
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        engine=engine,
        decode_timedelta=True,
        chunks={"time": 1, "depth": 1},
        **{"data_vars": "all", **(open_kwargs or {})},
    )
    try:
        if apply_rename:
            ds = rename_dims(ds)
        if processor is not None:
            ds = processor(ds)
        ds = snap_grid_coords(ds)
        ds = chunk_dataset(ds)
        write_append_zarr(name, ds, out_path)
    finally:
        ds.close()

    return out_path


class Netcdf2Zarr(BaseConverter):
    """
    Converts one variable's downloaded NetCDF/GRIB files into its Zarr store.

    The convert half of the pipeline: raw files are matched to dates by the
    variable's filename ``pattern``, read (GRIB via ``cfgrib``, NetCDF via the
    default engine), passed through that var_key's convert-time processor from
    ``processing.registry`` if it has one, and written per period.

    Writes go through ``write_append_zarr``, so an existing store is extended
    rather than replaced and keeps the encoding it was created with. Encoding
    from :meth:`_encoding` is therefore consulted only on a first write.
    """

    def __init__(
        self,
        var_key: str,
        *,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        download_root: Optional[Path] = None,
        file_period: FilePeriod = FilePeriod.YEAR,
        date_format: Literal["year", "date", "yearmonth"] = "year",
    ) -> None:
        """
        Initializes the setup to process downloaded raw data into Zarr files.

        Args:
            var_key: The key for the variable to be processed
            app_config: The application's configuration object.
            store_root: The root directory where the processed Zarr files will be stored.
            download_root: The root directory containing the downloaded raw data files to be processed.
            file_period: Temporal granularity ('year' or 'month') for file storage. Defaults to 'year'.
            date_format: string date format for output file name.
        """

        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[self.var_key]

        # Single source of truth for the archive-vs-delete decision, read by
        # both _archive_raw_files and _cleanup_period_files so they cannot drift.
        self.archive_raw = self.var_config.archive_raw

        self.download_root = resolve_download_path(self.var_config, download_root)

        self.file_period = validate_file_period(file_period)
        self.date_format: Literal["year", "date", "yearmonth"] = date_format

        self.catalog = ZarrCatalog(self.var_key, store_root=store_root)
        self.store_root = self.catalog.store_root

    def _encoding(self, ds: xr.Dataset) -> Optional[dict]:
        """
        Encoding for a store being created, or None to write as always.

        Only consulted on a first write: write_append_zarr ignores it when the
        path exists, because an append inherits the store's own encoding.
        """
        if self.var_config.store_dtype is not StoreDtype.INT16:
            return None
        return int16_encoding(ds)

    @property
    def _step_freq(self) -> str:
        """Cadence of this variable's store, as a pandas frequency alias."""
        return step_freq(self.var_config)

    def run(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> bool:
        """
        Convert downloaded raw files to Zarr.

        Args:
            start_date: Restrict the conversion to this window. Both dates must
                be given together; when omitted every downloaded file is
                converted, which is the default resume behaviour.
            end_date: End of the conversion window.

        Restricting the window is what makes it possible to re-convert one
        period from raw files already on disk, without re-downloading.
        """
        t0 = time.perf_counter()
        logger.info(
            f"Initializing Netcdf -> Zarr conversion for variable key: {self.var_key.upper()}"
        )

        window = (
            DateRange(pd.to_datetime(start_date), pd.to_datetime(end_date))
            if start_date is not None and end_date is not None
            else None
        )
        if window is not None:
            logger.info(f"Restricting conversion to {window}")

        # Reconcile any zarr write interrupted by a hard kill before gap
        # detection reads the store, so a half-written store is not trusted and
        # data stranded in a backup is restored.
        recover_zarr_store(self.store_root)

        # Trajectory-format variables (e.g. eddies) require spatial binning
        # before zarr storage — bypass the standard open_mfdataset pipeline.
        if self.var_config.trajectory_format:
            self._process_eddies(start_date, end_date)
            self.catalog.refresh(force=True)
            logger.success(
                f"Conversion complete (trajectory) in {time.perf_counter() - t0:.1f}s"
            )
            return True

        file_groups = self._group_map(groupby=self.file_period, window=window)

        if not file_groups:
            # Distinguish two ways of getting nothing. A window that matches no
            # downloaded file is a legitimate no-op (already warned about in
            # _group_map). Files that yield no date at all is a fault: every raw
            # file failed the pattern. Reported as a "0 period(s)" success it
            # would take the downloads with it: _cleanup_downloads deletes the
            # files, so a download that worked is discarded, the store never
            # changes, and the run exits 0.
            if self._get_file_date_series().empty:
                raise RuntimeError(
                    f"[{self.var_key}] {len(self._get_downloaded_files())} "
                    f"downloaded file(s) in {self.download_root}, but none "
                    f"yielded a date via pattern={self.var_config.pattern!r}. "
                    f"Raw files were left in place — check the pattern against "
                    f"the filenames on disk."
                )
            logger.warning(
                f"[{self.var_key}] Nothing to convert: no downloaded file falls "
                f"within {window}. Raw files left in place."
            )
            return False

        for period, paths in file_groups.items():
            self._process_period(period, paths)

        self.catalog.refresh(force=True)
        self._cleanup_downloads()
        logger.success(
            f"Conversion complete: {len(file_groups)} period(s) "
            f"in {time.perf_counter() - t0:.1f}s"
        )
        return True

    # ========= DATA PREPARATION FUNCTIONS =========

    def _group_map(
        self, groupby: FilePeriod, window: Optional[DateRange] = None
    ) -> dict[int | tuple[int, int], list[Path]]:
        """
        Group paths of nc files by period i.e groupby str.
        Returns a dictionary where the key is either an integer year or a
        (year, month) tuple and the value is a lis of paths to open in xr.open_mfdataset.

        Args:
            groupby: How to aggregate files before opening.

        Conventions:
            * GRIBs are read with ``engine="cfgrib"``.
            * NetCDFs are read with the default ``netcdf4`` engine.
            * The function automatically detects which engine to use per file,
              so you can mix formats in the same directory without trouble.

        Returns:
            Mapping of group key/period → files path for the selected period.
        """

        file_map = self._get_file_date_series()
        if file_map.empty:
            return {}

        if window is not None:
            file_map = file_map[
                (file_map.index >= window.start) & (file_map.index <= window.end)
            ]
            if file_map.empty:
                logger.warning(
                    f"[{self.var_key}] No downloaded files fall within {window}"
                )
                return {}

        if groupby == FilePeriod.YEAR:
            groups = file_map.groupby(file_map.index.year)  # type: ignore
        elif groupby == FilePeriod.MONTH:
            groups = file_map.groupby([file_map.index.year, file_map.index.month])  # type: ignore
        else:
            raise ValueError("groupby must be 'year' or 'month'")

        out = {}
        for key, group in groups:
            paths = sorted(set(group))
            out[key] = [Path(p) for p in paths]
        return out

    def _get_file_date_series(self) -> pd.Series:
        files = self._get_downloaded_files()
        records = []

        for f in files:
            for d in self._parse_file_dates(f):
                records.append((d, f))

        if not records:
            return pd.Series(dtype="object")

        dates, paths = zip(*records)
        return pd.Series(paths, index=pd.DatetimeIndex(dates)).sort_index()

    def _get_downloaded_files(self) -> list[Path]:
        """
        Return all downloaded files matching the pattern (nc and grib)

         Raises:
        FileNotFoundError: If no ``*.nc`` or ``*.grib`` files are present in
                           :pyattr:`self.down_dir`.
        """
        files = list(self.download_root.rglob("*.nc")) + list(
            self.download_root.rglob("*.grib")
        )
        if not files:
            raise FileNotFoundError(
                f"No downloaded NetCDF/GRIB files found in {self.download_root!s}"
            )

        files = filter_raw_files(files, self.var_config)
        if not files:
            raise FileNotFoundError(
                f"No downloaded files in {self.download_root!s} match "
                f"raw_include={self.var_config.raw_include!r}"
            )
        return sorted(files)

    def _match_file_dates(self, file: Path) -> Optional[tuple[str, ...]]:
        """
        Date groups captured from a filename, with unmatched optional ones dropped.

        A one-day subset is named with a single date — copernicusmarine writes
        ``…_2026-07-31.nc`` rather than ``…_2026-07-31-2026-07-31.nc`` — so a
        range pattern with an optional tail yields ``(date, None)``. Dropping
        the ``None`` lets the caller treat it as the single day it is, which is
        what makes a one-day repair download convertible at all.
        """
        if self.var_config.pattern is None:
            return None
        match = re.search(self.var_config.pattern, file.name)
        if not match:
            return None
        groups = tuple(g for g in match.groups() if g)
        return groups or None

    def _parse_file_dates(self, file: Path) -> list[pd.Timestamp]:
        groups = self._match_file_dates(file)
        if groups is None:
            return []

        if self.var_config.filename_date_range:
            if len(groups) < 2:
                return [pd.to_datetime(groups[0])]
            start, end = map(pd.to_datetime, groups[:2])
            return list(pd.date_range(start, end, freq="D"))

        return [pd.to_datetime("-".join(groups))]

    def _get_file_date_bounds(
        self, file: Path
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Return (start, end) date for a raw file without expanding the full date range."""
        groups = self._match_file_dates(file)
        if groups is None:
            return None
        if self.var_config.filename_date_range and len(groups) >= 2:
            start, end = map(pd.to_datetime, groups[:2])
            return start, end
        date = pd.to_datetime(
            groups[0] if self.var_config.filename_date_range else "-".join(groups)
        )
        return date, date

    def _read_manifest(self) -> list[dict]:
        """Read the download manifest written by CMEMSDownloader, or return [] if absent."""
        manifest_path = self.download_root / "h2mare_manifest.json"
        if not manifest_path.exists():
            return []
        try:
            return json.loads(manifest_path.read_text())
        except Exception as e:
            logger.warning(f"Could not read download manifest: {e}")
            return []

    def _write_provenance(
        self,
        zarr_path: Path,
        paths: list[Path],
        stored: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        """
        Write source provenance as a zarr root attribute (``source_datasets``).
        Skipped silently when no download manifest is available.

        Args:
            stored: The store's time axis, recorded against each requested span
                so the file states what it actually holds and not only what was
                asked for. Read back from *zarr_path* when omitted.
        """
        manifest = self._read_manifest()
        if not manifest:
            return

        tasks = [
            {
                "dataset_id": e["dataset_id"],
                "dataset_type": e["dataset_type"],
                "start": pd.to_datetime(e["start"]),
                "end": pd.to_datetime(e["end"]),
            }
            for e in manifest
        ]

        # For each raw file, find its date bounds and match to the correct task
        dataset_info: dict[str, dict] = {}
        for file_path in paths:
            bounds = self._get_file_date_bounds(file_path)
            if bounds is None:
                continue
            f_start, f_end = bounds

            matched = next(
                (t for t in tasks if t["start"] <= f_start <= t["end"]),
                None,
            )
            if matched is None:
                logger.warning(
                    f"Could not match {file_path.name} (start={f_start.date()}) "
                    "to any manifest task — skipping provenance for this file"
                )
                continue

            did = matched["dataset_id"]
            if did not in dataset_info:
                dataset_info[did] = {
                    "dataset_type": matched["dataset_type"],
                    "start_date": f_start,
                    "end_date": f_end,
                }
            else:
                dataset_info[did]["start_date"] = min(
                    dataset_info[did]["start_date"], f_start
                )
                dataset_info[did]["end_date"] = max(
                    dataset_info[did]["end_date"], f_end
                )

        if not dataset_info:
            return

        # The raw files say where each dataset's contribution begins; what the
        # file holds from it is read back off the axis by annotate_covered.
        records = sorted(
            [
                {
                    "dataset_id": did,
                    "dataset_type": info["dataset_type"],
                    "start_date": info["start_date"].strftime("%Y-%m-%d"),
                    "end_date": info["end_date"].strftime("%Y-%m-%d"),
                }
                for did, info in dataset_info.items()
            ],
            key=lambda r: r["start_date"],
        )

        import zarr

        root = zarr.open_group(str(zarr_path), mode="r+")

        # Merge rather than overwrite. A period file is appended to across runs,
        # so replacing the attribute with only this run's records drops the
        # earlier part of the same file — the eddies path has always merged.
        raw = root.attrs.get("source_datasets")
        existing = json.loads(raw) if isinstance(raw, str) and raw else []
        combined = merge_records(existing, records)
        combined = annotate_covered(
            combined, stored if stored is not None else read_store_dates(zarr_path)
        )

        root.attrs["source_datasets"] = json.dumps(combined)
        logger.debug(f"Wrote provenance to zarr attrs: {zarr_path.name}")

    # ========= PROCESSING FUNCTIONS =========
    def _process_eddies(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> None:
        import h2mare.processing.core.aviso as aviso

        try:
            ed_processor = aviso.EDDIESProcessor(
                store_root=self.store_root,
                download_root=self.download_root,
                file_period=self.file_period,
                date_format=self.date_format,
            )
            ed_processor.run(start_date, end_date)
            # Staging is download cleanup: it files freshly downloaded raw data
            # into the store. A windowed convert re-reads files that are
            # already wherever they belong, so there is nothing to file away.
            if start_date is None and end_date is None:
                self._stage_eddies_to_store(self.download_root)
            else:
                logger.debug(
                    f"[{self.var_key}] Windowed conversion — leaving raw files "
                    "where they are"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed processing data for var_key {self.var_key}"
            ) from e

    def _stage_eddies_to_store(self, download_root: Path) -> None:
        """
        Move raw eddies NetCDF files from download subfolders to the store.

        REP files (``download_root/rep/``) are moved additively — existing
        store files are preserved alongside new ones.

        NRT files (``download_root/nrt/``) replace the store entirely — all
        previous NRT files are deleted before the new ones are moved in.
        This ensures only the most recent NRT snapshot is kept.

        Falls back to a flat move to ``store_root`` when no rep/nrt subfolders
        are present (backward compatibility).

        The spent download manifest is deleted last, so the emptied folder can
        be pruned after the run.

        Does nothing when the raw files already live in the store, i.e. when
        ``download_root`` and ``store_root`` are the same directory — which is
        the case whenever ``convert --in-dir`` is pointed at an ``archive_raw``
        store. Staging there would move every file onto itself. The generic
        path guards this the same way in ``_archive_raw_files``.
        """
        if download_root.resolve() == self.store_root.resolve():
            logger.debug(
                f"[{self.var_key}] Raw files already in the store "
                f"({self.store_root}) — nothing to stage"
            )
            return

        rep_src = download_root / "rep"
        nrt_src = download_root / "nrt"

        if rep_src.exists() or nrt_src.exists():
            if rep_src.exists():
                rep_dst = self.store_root / "rep"
                rep_dst.mkdir(parents=True, exist_ok=True)
                safe_move_files(list(rep_src.glob("*.nc")), rep_dst)
                logger.debug(f"Moved REP eddies files to {rep_dst}")

            if nrt_src.exists():
                nrt_dst = self.store_root / "nrt"
                nrt_dst.mkdir(parents=True, exist_ok=True)

                # Move the new snapshot in first, then drop whatever it did not
                # replace. Clearing the destination up front means a move that
                # fails half way leaves neither copy.
                incoming = {p.name for p in nrt_src.glob("*.nc")}
                safe_move_files(list(nrt_src.glob("*.nc")), nrt_dst)
                logger.info(f"Moved new NRT eddies files to {nrt_dst}")

                stale = [p for p in nrt_dst.glob("*.nc") if p.name not in incoming]
                for old_file in stale:
                    old_file.unlink()
                if stale:
                    logger.info(
                        f"Cleared {len(stale)} stale NRT eddies file(s) from {nrt_dst}"
                    )
        else:
            # No subfolders — flat move (legacy layout)
            paths = list(download_root.rglob("*.nc"))
            if paths:
                safe_move_files(paths, self.store_root)

        # EDDIESProcessor has already read the manifest for provenance by the
        # time we stage, so it is spent. It is also the only non-".nc" file in
        # download_root, and nothing else would ever remove it: run() returns
        # early for a trajectory variable, so _cleanup_downloads — which drops
        # the manifest along with the folder for every other variable — never
        # runs, and the empty-dir prunes cannot fire while it sits there.
        manifest_path = download_root / "h2mare_manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
            logger.debug(f"Removed spent download manifest {manifest_path}")

    def _process_period(self, period, paths: list[Path]) -> None:
        logger.info(f"Processing period (year/year-month): {period}")

        # Keep the object open_mfdataset returned. process_dataset rebinds its
        # argument, so without this the only reference to it is lost — and
        # closing the *derived* dataset does not release the underlying *.nc
        # handles. On Windows those handles then block _archive_raw_files from
        # moving the raw files ([WinError 32]), which is why archive_raw: true
        # variables could not complete a conversion at all.
        ds_raw = self._open_dataset(paths)
        ds: Optional[xr.Dataset] = None

        try:
            ds = self.process_dataset(ds_raw)
            path = self.catalog.build_file_path(ds, self.date_format)
            # Read the axis before the write: write_append_zarr closes ds.
            written = _dataset_dates(ds, self._step_freq)
            write_append_zarr(self.var_key, ds, path, encoding=self._encoding(ds))
            # Deliberately before _archive_raw_files and _cleanup_period_files:
            # a failure here must leave the raw files where they are, so the
            # period can simply be re-converted once the gap is understood.
            self._verify_written_dates(
                path, written, self._expected_dates(period, paths), period
            )
            try:
                self._write_provenance(path, paths, self._stored_dates(path))
            except Exception as e:
                logger.warning(f"Could not write provenance for {path.name}: {e}")

            # After the write, so the extent describes what the file ended up
            # holding rather than what this run contributed to it. Warned about
            # rather than raised for the same reason provenance is: metadata is
            # not worth failing a conversion whose data landed correctly.
            try:
                write_cf_root_attrs(path)
            except Exception as e:
                logger.warning(f"Could not write CF root attrs for {path.name}: {e}")

            # Release every handle on the raw files before anything moves or
            # deletes them. Both closes are needed and neither is redundant.
            _close_all(ds, ds_raw)
            ds = None

            self._archive_raw_files(period, paths)
            self._cleanup_period_files(paths)

        except Exception as e:
            raise RuntimeError(
                f"Failed processing data for var_key {self.var_key}"
            ) from e
        finally:
            # Idempotent, so the success path closing early costs nothing; this
            # is here for the failure path, which otherwise leaks the handles
            # and leaves the store directory locked on Windows.
            _close_all(ds, ds_raw)

    # ========= WRITE VERIFICATION =========

    def _period_bounds(self, period: int | tuple[int, int]) -> DateRange:
        """Calendar span of a group key produced by :meth:`_group_map`."""
        if isinstance(period, tuple):
            year, month = period
            start = pd.Timestamp(year=year, month=month, day=1)
            return DateRange(start=start, end=start + pd.offsets.MonthEnd(0))
        start = pd.Timestamp(year=period, month=1, day=1)
        return DateRange(start=start, end=pd.Timestamp(year=period, month=12, day=31))

    def _expected_dates(
        self, period: int | tuple[int, int], paths: list[Path]
    ) -> pd.DatetimeIndex:
        """
        Daily calendar this period was *asked* to deliver, for the lag warning.

        Taken from the download manifest rather than the files on disk, and the
        distinction is the whole point: a file that failed to download is
        absent from disk, so an expectation derived from disk moves to match
        the loss and reports success. The manifest states what was requested,
        and is written before conversion runs.

        Falls back to the dates the raw filenames claim when no manifest exists
        (a re-convert from archived files). That fallback cannot see a missing
        file — which is why it only ever drives a warning, never the error.
        """
        bounds = self._period_bounds(period)
        spans = [
            (pd.to_datetime(e["start"]), pd.to_datetime(e["end"]))
            for e in self._read_manifest()
            if e.get("start") and e.get("end")
        ]
        if not spans:
            dates = pd.DatetimeIndex(
                sorted({d for p in paths for d in self._parse_file_dates(p)})
            )
            return dates[(dates >= bounds.start) & (dates <= bounds.end)]

        start = max(min(s for s, _ in spans), bounds.start)
        end = min(max(e for _, e in spans), bounds.end)
        if start > end:
            return pd.DatetimeIndex([])
        return pd.date_range(
            start, _span_end(end, self._step_freq), freq=self._step_freq
        )

    def _stored_dates(self, path: Path) -> pd.DatetimeIndex:
        """Time axis of the Zarr just written. Coordinate read only — no data."""
        try:
            with xr.open_zarr(path, consolidated=False) as stored:
                return _dataset_dates(stored, self._step_freq)
        except Exception as e:
            logger.warning(f"Could not read back the time axis of {path.name}: {e}")
            return pd.DatetimeIndex([])

    def _verify_written_dates(
        self,
        path: Path,
        written: pd.DatetimeIndex,
        expected: pd.DatetimeIndex,
        period: int | tuple[int, int],
    ) -> None:
        """
        Reconcile the dates now in the store against the dates asked for.

        Raises when a day is missing from the *middle* of the span this
        conversion just wrote: the provider published the days on either side,
        both are in the store, and the one between them is not. There is no
        benign reading of that, and it is the one shape of loss that nothing
        downstream can recover — store coverage is a min/max watermark that an
        interior hole cannot move, so the year goes on reporting itself full.

        A shortfall at either end is only warned about, because provider lag
        legitimately produces one every time the requested window runs to today.

        Deliberately silent on days that are present but entirely null. That is
        what a genuine source gap looks like (chl has three in 1999 alone), and
        a check that fires on those every year is one people learn to ignore —
        at which point it protects nothing.

        Raises:
            RuntimeError: If a day is missing from inside the written span.
        """
        if len(written) < 2:
            return

        stored = self._stored_dates(path)
        if len(stored) == 0:
            return

        span_start, span_end = written[0], written[-1]
        missing = pd.date_range(span_start, span_end, freq=self._step_freq).difference(
            stored
        )
        # Days the provider never published are not the pipeline's doing, and
        # failing on them would make a whole period permanently unconvertible.
        # known_gaps are whole days: match a missing step by its day, or an
        # hourly axis would keep reporting the 23 non-midnight steps forever.
        gap_days = known_gap_days(self.var_config)
        missing = missing[~missing.normalize().isin(gap_days)]
        if len(missing):
            raise RuntimeError(
                f"[{self.var_key}] period {period}: {len(missing)} day(s) missing "
                f"from {path.name} inside the range just written "
                f"({span_start.date()} → {span_end.date()}): "
                f"{format_date_blocks(missing)}. Raw files were left in place — "
                f"re-run the download for those dates, then re-convert."
            )

        shortfall = expected[
            (expected < span_start) | (expected > span_end)
        ].difference(stored)
        if len(shortfall):
            logger.warning(
                f"[{self.var_key}] period {period}: {len(shortfall)} requested day(s) "
                f"not delivered outside the written span: "
                f"{format_date_blocks(shortfall)}. Ordinary provider lag at the "
                f"tail; anything else is a gap `h2mare audit` will keep reporting."
            )

    def _open_dataset(self, paths: list[Path]) -> xr.Dataset:
        """Open a group of files as a single dataset."""
        first_ext = paths[0].suffix.lower()
        engine = "cfgrib" if first_ext in {".grib", ".grb"} else "netcdf4"

        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            import h2mare.processing.core.cds as cds

            ds = rename_dims(ds)
            ds = cds.merge_time_step(ds)
            return cds._get_ds_for_month(ds)

        # Explicit, for the reason ZarrReader pins it: xarray's default moves
        # from data_vars="all" to data_vars=None, which resolves to "minimal"
        # whenever the concat dim is present — as `time` always is here. The
        # two differ only for a data variable carrying no time dim (AVISO's
        # raw FSLE files ship `crs`, `lat_bnds`, `lon_bnds`): "all" broadcasts
        # it along time, "minimal" leaves it alone.
        #
        # That difference does not reach any store today, so the pin is not
        # protecting store contents — `process_dataset` runs after this and
        # drops such variables (`process_fsle` selects `source_vars`), while
        # the var_keys with no convert-time processor get raw files that
        # already hold only what CMEMS `subset()` was asked for. What the pin
        # protects is the dataset handed to `process_dataset`: a processor
        # that indexes or merges by variable would see a different shape if
        # xarray flipped the default under it. "all" is the current behaviour.
        # Switching to "minimal" is a deliberate call, not a default's to
        # make, and it cascades a second FutureWarning for `compat`.
        return xr.open_mfdataset(
            sorted(paths),
            combine="by_coords",
            engine=engine,
            decode_timedelta=True,
            chunks={"time": 1, "depth": 1},
            data_vars="all",
            preprocess=(preprocess if self.var_config.merge_time_step else None),
        )

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply dataset-specific processing depending on downloader and variable key."""
        if self.var_config.source != "cds":
            ds = rename_dims(ds)

        processor = PROCESSORS.get(self.var_key)
        if processor:
            ds = processor(ds, self.var_config, self.var_key)

        # After the processor, so derived_vars name variables as it leaves them.
        ds = apply_derived_vars(ds, self.var_config.derived_vars, self.var_key)

        # Snap lon/lat to a canonical grid so float-noise drift between a source's
        # reprocessed periods can't union into a doubled axis on read/append.
        ds = snap_grid_coords(ds)

        # After the processor, so a derived variable gets the same treatment as
        # a pass-through one: arithmetic and rolling reductions drop attrs, so
        # sst, gke and the _std layers arrived here carrying none at all.
        ds = apply_cf_attrs(ds, native_var_key=self.var_key)

        return chunk_dataset(ds)

    # ========= CLEANUP FUNCTIONS =========

    def _archive_raw_files(
        self, period: int | tuple[int, int], paths: list[Path], retries=10, delay=0.5
    ) -> None:
        """
        Move files for period folders (year or month as defined in period) if store_root is different from download root.
        Runs when ``archive_raw`` is set (by default aviso (fsle only) and cds data).
        """
        if not self.archive_raw:
            return None

        if self.download_root != self.store_root:
            logger.info(
                f"Archiving raw files from {self.download_root} to {self.store_root}"
            )
            dest_dir = self.store_root / self._resolve_string(period)
            dest_dir.mkdir(parents=True, exist_ok=True)

            safe_move_files(paths, dest_dir, retries=retries, delay=delay)

    def _cleanup_period_files(self, paths: list[Path]) -> None:
        """
        Delete a period's raw files once it has been converted successfully.

        Without this, a failure on a later period aborts ``run()`` with every
        raw file still on disk, so the next run re-globs and reprocesses *all*
        periods (correct, via the overlap resolver, but wasteful — and it
        rewrites already-good stores, widening the crash-exposure window).
        Removing each period's inputs as it completes makes the convert step
        incrementally resumable.

        Variables whose raw files are archived into the store (``archive_raw``;
        by default ``cds``/``aviso``) are handled by :meth:`_archive_raw_files`
        and skipped here; so is the in-place layout where downloads live in the
        store itself.
        """
        if self.archive_raw:
            return
        if self.download_root == self.store_root:
            return
        for p in paths:
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.debug(f"Could not remove raw file {p}: {e}")

    def _cleanup_downloads(self) -> None:
        """
        Remove the downloads folder, but only once this run has consumed it.

        Everything this run converted is already gone by the time we get here:
        :meth:`_archive_raw_files` moved it into the store, or
        :meth:`_cleanup_period_files` deleted it. So whatever is left was *not*
        converted — a period outside the requested window, or a file the date
        pattern skipped — and removing the tree wholesale would delete raw data
        this run never looked at. Re-converting a single year would take every
        other year's downloads with it.

        Left in place when downloads live in the store itself; there the raw
        files *are* the archive.
        """
        if self.download_root == self.store_root:
            return
        if not self.download_root.exists():
            return

        # The converter's own notion of a raw file, so leftover .idx sidecars
        # and the download manifest do not keep an otherwise-spent folder alive.
        try:
            remaining = self._get_downloaded_files()
        except FileNotFoundError:
            remaining = []

        if remaining:
            logger.debug(
                f"Leaving {self.download_root} in place: {len(remaining)} raw "
                "file(s) there were not converted by this run."
            )
            return

        try:
            logger.debug(f"Removing spent downloads folder {self.download_root}")
            safe_rmtree(self.download_root)
        except OSError:
            logger.exception(f"Could not remove {self.download_root}")

    def _resolve_string(self, period: int | tuple[int, int]) -> str:
        """
        Resolve year/yearmonth folders for file move.

        Args:
            period (int | tuple[int, int]): year (int) of tuple (year, month) derived from _group_map function.

        Raises:
            ValueError: If period is not int nor tuple

        Returns:
            str: with year or year/month

        The separator is a forward slash, which ``Path`` resolves to a nested
        directory on every platform. A literal backslash does not: on POSIX
        ``store_root / "2021\\3"`` names a *single* directory containing a
        backslash instead of ``2021/3``.
        """
        if isinstance(period, int):
            return str(period)
        elif isinstance(period, tuple) and len(period) == 2:
            return f"{period[0]}/{period[1]}"
        raise ValueError("Input must be a int or a 2-tuple of ints")
