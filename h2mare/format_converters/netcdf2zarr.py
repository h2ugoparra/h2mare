"""
Process downloaded netcdf/grib raw data to zarr files
"""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.format_converters.base import BaseConverter
from h2mare.processing.registry import PROCESSORS
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import chunk_dataset, rename_dims, snap_grid_coords
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import TimeResolution
from h2mare.utils.files_io import safe_move_files, safe_rmtree
from h2mare.utils.paths import resolve_download_path
from h2mare.validators import validate_time_resolution, validate_var_key

warnings.filterwarnings("ignore")


class Netcdf2Zarr(BaseConverter):
    def __init__(
        self,
        var_key: str,
        *,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        download_root: Optional[Path] = None,
        time_resolution: TimeResolution = TimeResolution.YEAR,
        date_format: Literal["year", "date", "yearmonth"] = "year",
    ) -> None:
        """
        Initializes the setup to process downloaded raw data into Zarr files.

        Args:
            var_key: The key for the variable to be processed
            app_config: The application's configuration object.
            store_root: The root directory where the processed Zarr files will be stored.
            download_root: The root directory containing the downloaded raw data files to be processed.
            time_resolution: Temporal granularity ('year' or 'month') for file storage. Defaults to 'year'.
            date_format: string date format for output file name.
        """

        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[self.var_key]

        self.download_root = resolve_download_path(self.var_config, download_root)

        self.time_resolution = validate_time_resolution(time_resolution)
        self.date_format: Literal["year", "date", "yearmonth"] = date_format

        self.catalog = ZarrCatalog(self.var_key, store_root=store_root)
        self.store_root = self.catalog.store_root

    def run(self) -> bool:
        t0 = time.perf_counter()
        logger.info(
            f"Initializing Netcdf -> Zarr conversion for variable key: {self.var_key.upper()}"
        )

        # Trajectory-format variables (e.g. eddies) require spatial binning
        # before zarr storage — bypass the standard open_mfdataset pipeline.
        if self.var_config.trajectory_format:
            self._process_eddies()
            self.catalog.refresh(force=True)
            logger.success(
                f"Conversion complete (trajectory) in {time.perf_counter() - t0:.1f}s"
            )
            return True

        file_groups = self._group_map(groupby=self.time_resolution)

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
        self, groupby: TimeResolution
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

        if groupby == TimeResolution.YEAR:
            groups = file_map.groupby(file_map.index.year)  # type: ignore
        elif groupby == TimeResolution.MONTH:
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
        return sorted(files)

    def _parse_file_dates(self, file: Path) -> list[pd.Timestamp]:
        match = re.search(self.var_config.pattern, file.name)
        if not match:
            return []

        if self.var_config.filename_date_range:
            start, end = map(pd.to_datetime, match.groups())
            return list(pd.date_range(start, end, freq="D"))

        return [pd.to_datetime("-".join(match.groups()))]

    def _get_file_date_bounds(
        self, file: Path
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Return (start, end) date for a raw file without expanding the full date range."""
        match = re.search(self.var_config.pattern, file.name)
        if not match:
            return None
        if self.var_config.filename_date_range:
            start, end = map(pd.to_datetime, match.groups())
            return start, end
        date = pd.to_datetime("-".join(match.groups()))
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

    def _write_provenance(self, zarr_path: Path, paths: list[Path]) -> None:
        """
        Write source provenance as a zarr root attribute (``source_datasets``).
        Skipped silently when no download manifest is available.
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
        root.attrs["source_datasets"] = json.dumps(records)
        logger.debug(f"Wrote provenance to zarr attrs: {zarr_path.name}")

    # ========= PROCESSING FUNCTIONS =========
    def _process_eddies(self):
        import h2mare.processing.core.aviso as aviso

        try:
            ed_processor = aviso.EDDIESProcessor(
                store_root=self.store_root,
                download_root=self.download_root,
                time_resolution=self.time_resolution,
                date_format=self.date_format,
            )
            ed_processor.run()
            self._stage_eddies_to_store(self.download_root)
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
        """
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
                if nrt_dst.exists():
                    for old_file in nrt_dst.glob("*.nc"):
                        old_file.unlink()
                    logger.info(f"Cleared stale NRT eddies files from {nrt_dst}")
                nrt_dst.mkdir(parents=True, exist_ok=True)
                safe_move_files(list(nrt_src.glob("*.nc")), nrt_dst)
                logger.info(f"Moved new NRT eddies files to {nrt_dst}")
        else:
            # No subfolders — flat move (legacy layout)
            paths = list(download_root.rglob("*.nc"))
            if paths:
                safe_move_files(paths, self.store_root)

    def _process_period(self, period, paths: list[Path]) -> None:
        logger.info(f"Processing period (year/year-month): {period}")

        ds = self._open_dataset(paths)

        try:
            ds = self.process_dataset(ds)
            path = self.catalog.build_file_path(ds, self.date_format)
            write_append_zarr(self.var_key, ds, path)
            try:
                self._write_provenance(path, paths)
            except Exception as e:
                logger.warning(f"Could not write provenance for {path.name}: {e}")
            ds.close()
            del ds
            self._archive_raw_files(period, paths)

        except Exception as e:
            raise RuntimeError(
                f"Failed processing data for var_key {self.var_key}"
            ) from e
        # finally:
        #    ds.close()

    def _open_dataset(self, paths: list[Path]) -> xr.Dataset:
        """Open a group of files as a single dataset."""
        first_ext = paths[0].suffix.lower()
        engine = "cfgrib" if first_ext in {".grib", ".grb"} else "netcdf4"

        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            import h2mare.processing.core.cds as cds

            ds = rename_dims(ds)
            ds = cds.merge_time_step(ds)
            return cds._get_ds_for_month(ds)

        return xr.open_mfdataset(
            sorted(paths),
            combine="by_coords",
            engine=engine,
            decode_timedelta=True,
            chunks={"time": 1, "depth": 1},
            preprocess=(preprocess if self.var_config.merge_time_step else None),
        )

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply dataset-specific processing depending on downloader and variable key."""
        if self.var_config.source != "cds":
            ds = rename_dims(ds)

        processor = PROCESSORS.get(self.var_key)
        if processor:
            ds = processor(ds, self.var_config, self.var_key)

        # Snap lon/lat to a canonical grid so float-noise drift between a source's
        # reprocessed periods can't union into a doubled axis on read/append.
        ds = snap_grid_coords(ds)

        return chunk_dataset(ds)

    # ========= CLEANUP FUNCTIONS =========

    def _archive_raw_files(
        self, period: int | tuple[int, int], paths: list[Path], retries=10, delay=0.5
    ) -> None:
        """
        Move files for period folders (year or month as defined in period) if store_root is different from download root.
        Currently only for aviso (fsle only) and cds data.
        """
        if self.var_config.source not in ["cds", "aviso"]:
            return None

        if self.download_root != self.store_root:
            logger.info(
                f"Archiving raw files from {self.download_root} to {self.store_root}"
            )
            dest_dir = self.store_root / self._resolve_string(period)
            dest_dir.mkdir(parents=True, exist_ok=True)

            safe_move_files(paths, dest_dir, retries=retries, delay=delay)

    def _cleanup_downloads(self) -> None:
        """Remove raw data from dowloads folder if download_root is different from store_root to avoid cluttering downloads with raw files."""
        if self.download_root != self.store_root:
            try:
                logger.debug(f"Removing raw files from {self.download_root}")
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
        """
        if isinstance(period, int):
            return str(period)
        elif isinstance(period, tuple) and len(period) == 2:
            return rf"{str(period[0])}\{str(period[1])}"
        raise ValueError("Input must be a int or a 2-tuple of ints")
