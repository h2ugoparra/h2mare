"""
PipelineManager class to orchestrate the download and processing of datasets based on the provided configuration and registry
"""

from pathlib import Path
from typing import List, Optional, Type, Union

import pandas as pd
from loguru import logger

from h2mare import SYSTEM_VAR_KEYS, AppConfig, get_settings
from h2mare.format_converters.netcdf2zarr import Netcdf2Zarr
from h2mare.utils.files_io import prune_empty_dirs


class PipelineManager:
    def __init__(
        self,
        app_config: AppConfig,
        registry: dict[str, Type],
        store_root: Union[str, Path],
        dry_run: bool = False,
        start_date: Union[pd.Timestamp, None] = None,
        end_date: Union[pd.Timestamp, None] = None,
        no_convert: bool = False,
        no_compile: bool = False,
        no_parquet: bool = False,
        h2ds_zarr_backup: bool = False,
        h2ds_parquet_backup: bool = False,
        zarr_backup_dir: Optional[Path] = None,
        parquet_backup_dir: Optional[Path] = None,
    ):

        self.app_config = app_config
        self.registry = registry
        self.store_root = Path(store_root)
        self.dry_run = dry_run
        self.start_date = start_date
        self.end_date = end_date
        self.no_convert = no_convert
        self.no_compile = no_compile
        self.no_parquet = no_parquet
        self.h2ds_zarr_backup = h2ds_zarr_backup
        self.h2ds_parquet_backup = h2ds_parquet_backup
        self.zarr_backup_dir = zarr_backup_dir
        self.parquet_backup_dir = parquet_backup_dir

    def run(self, variables: Optional[List[str]] = None) -> bool:
        """Run the full pipeline. Returns True if all steps succeeded, False if any failed."""
        if variables is None:
            variables = list(self.app_config.variables.keys())

        # Pre-flight: fail fast if any variable's source has no registered downloader
        unregistered = {
            var_key: cfg.source
            for var_key in variables
            if var_key not in SYSTEM_VAR_KEYS
            and (cfg := self.app_config.variables.get(var_key)) is not None
            and cfg.source not in self.registry
        }
        if unregistered:
            for var_key, source in unregistered.items():
                logger.error(
                    f"No downloader registered for source '{source}' (variable '{var_key}'). "
                    f"Registered sources: {sorted(self.registry)}."
                )
            return False

        _failed = False

        for var_key in variables:
            if var_key in SYSTEM_VAR_KEYS:
                continue

            var_config = self.app_config.variables.get(var_key)
            if not var_config:
                logger.warning(f"⚠️ Variable '{var_key}' not found in config. Skipping.")
                _failed = True
                continue

            DownloaderClass = self.registry.get(var_config.source)
            if DownloaderClass is None:
                logger.warning(
                    f"⚠️ No downloader registered for source '{var_config.source}' "
                    f"(variable '{var_key}'). Skipping."
                )
                _failed = True
                continue

            downloader = DownloaderClass(
                var_key=var_key,
                app_config=self.app_config,
                store_root=self.store_root,
            )

            try:
                downloaded = downloader.run(
                    dry_run=self.dry_run,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )
            except Exception as e:
                logger.opt(exception=True).error(
                    f"Download failed for '{var_key}': {e}"
                )
                _failed = True
                continue

            if self.no_convert or self.dry_run or not downloaded:
                continue

            try:
                Netcdf2Zarr(var_key).run()
            except Exception as e:
                logger.opt(exception=True).error(
                    f"Processing failed for '{var_key}': {e}"
                )
                _failed = True

        if not self.no_compile and not self.no_convert and not self.dry_run:
            from h2mare.processing.compiler import Compiler

            try:
                Compiler(remote_store_root=self.store_root).run(
                    start_date=self.start_date,
                    end_date=self.end_date,
                    var_keys=variables,
                    zarr_backup=self.h2ds_zarr_backup,
                    zarr_backup_dir=self.zarr_backup_dir,
                )
            except Exception as e:
                logger.opt(exception=True).error(f"Compile step failed: {e}")
                _failed = True

        _skip_parquet = (
            self.no_parquet or self.no_compile or self.no_convert or self.dry_run
        )
        if not _skip_parquet:
            from h2mare.format_converters.zarr2parquet import Zarr2Parquet

            try:
                h2ds_local_folder = self.app_config.variables["h2ds"].local_folder
                converter = Zarr2Parquet(
                    var_key="h2ds",
                    parquet_root=get_settings().PARQUET_DIR,
                    store_root=self.store_root / h2ds_local_folder,
                )
                converter.run(
                    start_date=self.start_date,
                    end_date=self.end_date,
                )
                if self.h2ds_parquet_backup:
                    converter.sync_data(remote_root=self.parquet_backup_dir)
            except Exception as e:
                logger.opt(exception=True).error(f"Parquet conversion step failed: {e}")
                _failed = True

        self._cleanup_empty_download_dirs()

        if _failed:
            logger.warning("Pipeline finished with errors — see messages above.")
        elif self.dry_run:
            logger.success("Dry run complete — no data written.")
        else:
            logger.success("Pipeline completed successfully.")
        return not _failed

    def _cleanup_empty_download_dirs(self) -> None:
        """
        Prune empty directories left under the downloads root after a run.

        A bottom-up prune handles what a per-variable rmdir cannot: nested
        empty subfolders (eddies' rep/nrt staging dirs) and multi-level
        local_folder paths (e.g. CMEMS_2nd_productivity/mnkc) whose parent
        survives when only the leaf is removed.
        """
        downloads_root = get_settings().DOWNLOADS_DIR
        removed = prune_empty_dirs(downloads_root)
        if removed:
            logger.debug(
                f"Removed {removed} empty download director{'y' if removed == 1 else 'ies'} "
                f"under {downloads_root}"
            )
