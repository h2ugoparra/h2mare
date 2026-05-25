from __future__ import annotations

import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from ftplib import FTP
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from h2mare.config import AppConfig
from h2mare.downloader.base import BaseDownloader
from h2mare.utils.date_range import resolve_date_range
from h2mare.types import DateLike, DateRange
from h2mare.utils.datetime_utils import normalize_date

warnings.filterwarnings("ignore")


class AVISODownloader(BaseDownloader):
    def __init__(
        self,
        var_key: str,
        *,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        download_root: Optional[Path] = None,
    ):
        """
        Initializes AVISO Downloader via FTP.

        Args:
            var_key: Variable key from app_config.variables.
            app_config: Application configuration. If None, loads from settings.
            store_root: Root directory for zarr files. If None, uses settings.STORE_ROOT.
            download_root: Root directory for downloads. If None, uses settings.DOWNLOADS_DIR.
        """
        super().__init__(
            var_key,
            app_config=app_config,
            store_root=store_root,
            download_root=download_root,
        )
        self.ftp = self.connect_ftp()
        self._rep_availability = None
        self._nrt_availability = None

    # ==================== FTP Connection ====================
    def get_all_files_recursively(self, path=""):
        """Recursively get all files using MLSD (more reliable if supported)"""
        all_files = []
        try:
            for item_name, item_facts in self.ftp.mlsd(path):
                # Skip . and .. directories
                if item_name in [".", ".."]:
                    continue

                # Build full path
                full_path = f"{path}/{item_name}" if path else item_name
                # Check if it's a directory
                if item_facts.get("type") == "dir":
                    # Recursively get files from subdirectory
                    all_files.extend(self.get_all_files_recursively(full_path))
                elif item_facts.get("type") == "file":
                    all_files.append(full_path)
        except Exception as e:
            logger.error(f"Error accessing {path}: {e}")

        return sorted(all_files)

    def connect_ftp(self):
        """Connect to the FTP server."""
        # FTP Connection - for AVISO data, dataset_id represents the ftp path.
        # server is the root of the path
        ftp_server = self.app_config.secrets.aviso_ftp_server
        # logger.debug(f"Connecting to {ftp_server}")

        if not ftp_server:
            raise EnvironmentError(
                "AVISO server not found in .env or environment variables."
            )

        username = self.app_config.secrets.aviso_username
        password = self.app_config.secrets.aviso_password

        if not username or not password:
            raise EnvironmentError(
                "AVISO credentials not found in .env or environment variables."
            )

        ftp = FTP(host=ftp_server, user=str(username), passwd=str(password))
        ftp.set_pasv(True)
        # logger.debug("Connected!")
        return ftp

    def adjust_ftp_path_to_dataset(self, dataset_id: str) -> FTP:
        self.ftp.cwd("/")
        self.ftp.cwd(dataset_id)  # adjust path
        return self.ftp

    # ==================== Dates and data coverage Resolution ====================
    def _resolve_date_range(
        self,
        start_date: DateLike | None,
        end_date: DateLike | None,
    ) -> DateRange:
        """
        Resolve storage date range for download.
        Priority:
            1. Explicit arguments
            2. Default dates from __init__
            3. Latest date from store + 1 day to today
        """
        # Use explicit args, fall back to defaults
        start = normalize_date(start_date) if start_date else None
        end = normalize_date(end_date) if end_date else None

        return resolve_date_range(self.var_key, start=start, end=end)

    def _get_dataset_files(self, dataset_id: str) -> list[str]:
        """Get list of files for a given dataset_id from FTP."""
        self.adjust_ftp_path_to_dataset(dataset_id)
        return self.get_all_files_recursively()

    def _get_dataset_availability(self, files: list[str]) -> DateRange:
        """
        Get date range from files.

        Args:
            files: list of files retrieved from ftp/dataset_id
        """
        dates = []
        for f in files:
            try:
                extracted = self._extract_date_from_filename(f)
                if isinstance(extracted, tuple):
                    dates.extend(extracted)
                else:
                    dates.append(extracted)
            except ValueError:
                logger.warning(f"Could not extract date from file: {f}")
        if dates:
            if isinstance(dates, tuple):
                return DateRange(dates[0], dates[-1])
            elif isinstance(dates, list):
                return DateRange(min(dates), max(dates))
        raise ValueError(
            f"No valid dates found in filenames for variable '{self.var_key}'"
        )

    def _extract_date_from_filename(
        self, files: str | list[str]
    ) -> pd.Timestamp | tuple[pd.Timestamp, pd.Timestamp]:
        """
        Extract date from FTP filename based on regex patterns.
        - fsle: returns single date (first date in filename)
        - eddies: returns (start_date, end_date) tuple
        """
        files = [files] if isinstance(files, str) else files

        # regex patterns for ftp file search
        fsle_pattern = re.compile(self.var_config.pattern)
        eddies_pattern = re.compile(self.var_config.pattern)

        for f in files:
            if self.var_key == "fsle":
                match = fsle_pattern.search(f)
                if match:
                    return pd.Timestamp(match.group(1))

            elif "eddies" in self.var_key:
                match = eddies_pattern.search(f)
                if match:
                    return pd.Timestamp(match.group(1)), pd.Timestamp(match.group(2))
        else:
            raise ValueError(
                f"No valid dates found in filenames for variable '{self.var_key}' with pattern '{self.var_config.pattern}'"
            )

    def _filter_files_by_range(
        self, files: list[str], date_range: DateRange
    ) -> list[str]:
        """Filter file list to those overlapping with the given DateRange."""
        result = []
        for filepath in files:
            dates = self._extract_date_from_filename(filepath)
            if dates is None:
                continue
            if isinstance(dates, tuple):
                file_start, file_end = dates
                if file_start <= date_range.end and file_end >= date_range.start:
                    result.append(filepath)
            else:
                if date_range.start <= dates <= date_range.end:
                    result.append(filepath)
        return result

    def get_rep_availability(self) -> DateRange:
        """Get REP dataset availability from FTP file listing (cached)."""
        if self._rep_availability is None:
            files = self._get_dataset_files(self.var_config.dataset_id_rep)
            self._rep_availability = self._get_dataset_availability(files)
        return self._rep_availability

    def get_nrt_availability(self) -> Optional[DateRange]:
        """Get NRT dataset availability from FTP file listing (cached), or None if not configured."""
        if self.var_config.dataset_id_nrt is None:
            return None
        if self._nrt_availability is None:
            files = self._get_dataset_files(self.var_config.dataset_id_nrt)
            self._nrt_availability = self._get_dataset_availability(files)
        return self._nrt_availability

    def _create_download_tasks(
        self, requested_range: DateRange
    ) -> list[tuple[str, str]]:
        """
        Split date range into download tasks based on dataset availability.

        Strategy:
            - Uses REP for historical data up to its end date.
            - NRT dataset for any remaining period (if available).

        Args:
            requested_range: Date range to download

        Returns:
            List of (filepath, source) tuples where source is 'rep' or 'nrt'.
        """
        tasks: list[tuple[str, str]] = []

        rep_files = self._get_dataset_files(self.var_config.dataset_id_rep)
        rep_avail = self._get_dataset_availability(rep_files)

        if self.var_config.dataset_id_nrt:
            nrt_files = self._get_dataset_files(self.var_config.dataset_id_nrt)
            nrt_avail = self._get_dataset_availability(nrt_files)

        # REP covers the requested range up to its end date
        rep_overlap = requested_range.intersection(rep_avail)
        if rep_overlap:
            rep_files = self._filter_files_by_range(rep_files, rep_overlap)
            tasks.extend((fp, "rep") for fp in rep_files)

        # NRT covers anything beyond REP's end date, if available
        if nrt_avail:
            nrt_start = (
                rep_avail.end + pd.Timedelta(days=1)
                if rep_overlap
                else requested_range.start
            )
            nrt_request = DateRange(start=nrt_start, end=requested_range.end)
            nrt_overlap = nrt_request.intersection(nrt_avail)
            if nrt_overlap:
                nrt_files = self._filter_files_by_range(nrt_files, nrt_overlap)
                tasks.extend((fp, "nrt") for fp in nrt_files)

        if not tasks:
            logger.warning(
                f"Requested range {requested_range} does not overlap with available datasets"
            )

        return tasks

    # ==================== Download Execution ====================
    def download_file(self, path: str, output_dir: Optional[Path] = None) -> None:
        """Download individual files from FTP."""
        local_path = (output_dir or self.download_dir) / path.split("/")[-1]
        logger.debug(f"📥 Downloading {path} to {local_path}")

        try:
            self.ftp.voidcmd("TYPE I")  # switch to binary mode for SIZE command
            file_size = self.ftp.size(path)
        except Exception as e:
            logger.warning(f"⚠️ Error getting file size for {path}: {e}")
            file_size = None

        with open(local_path, "wb") as f:
            if file_size:
                with tqdm(
                    total=file_size, unit="B", unit_scale=True, desc=path.split("/")[-1]
                ) as pbar:

                    def callback(data):
                        f.write(data)
                        pbar.update(len(data))

                    self.ftp.retrbinary(f"RETR {path}", callback)
            else:
                # Fallback if file_size is unknown
                self.ftp.retrbinary(f"RETR {path}", f.write)

        logger.success(f"Downloaded {path.split('/')[-1]} to {local_path}")

    def download_parallel(
        self,
        paths: list[str],
        dataset_id: str,
        output_dir: Optional[Path] = None,
        max_workers: int = 2,
    ):
        """Download multiple FSLE files in parallel using multiple FTP connections."""
        output_dir = output_dir or self.download_dir

        def download_single(
            path: str, dataset_id: str = dataset_id, output_dir: Path = output_dir
        ):
            # Each thread gets its own FTP connection
            ftp = self.connect_ftp()
            ftp.cwd(dataset_id)
            try:
                local_path = output_dir / path.split("/")[-1]
                ftp.voidcmd("TYPE I")
                file_size = ftp.size(path)
            except Exception as e:
                logger.warning(f"⚠️ Error getting file size for {path}: {e}")
                file_size = None

            with open(local_path, "wb") as f:
                if file_size:
                    with tqdm(
                        total=file_size,
                        unit="B",
                        unit_scale=True,
                        desc=path.split("/")[-1],
                    ) as pbar:

                        def callback(data):
                            f.write(data)
                            pbar.update(len(data))

                        ftp.retrbinary(f"RETR {path}", callback)
                else:
                    ftp.retrbinary(f"RETR {path}", f.write)

            ftp.quit()
            return path

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, path): path for path in paths}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"❌ Failed to download {futures[future]}: {e}")

    def run(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        parallel: bool = True,
        max_workers: int = 2,
    ) -> bool:
        """
        Run download for specified date range.

        Args:
            start_date: Start date (None = use default or infer)
            end_date: End date (None = use default or infer)
            output_dir: Optional directory to save downloads (defaults to self.download_dir)
            dry_run: If True, plan download but don't execute
            parallel: If True, download files in parallel
            max_workers: Number of parallel threads to use if parallel=True (only two workers recommended for FTP)

        Example:
            >>> downloader = AVISODownloader("fsle")
            >>>
            >>> # Download specific range
            >>> files = downloader.run("2023-01-01", "2023-12-31")
            >>>
            >>> # Dry run to see what would be downloaded
            >>> downloader.run("2023-01-01", "2023-12-31", dry_run=True)

        Returns:
            True if downloads were executed, False if skipped (no tasks or dry run).
        """
        requested_range = self._resolve_date_range(start_date, end_date)

        tasks = self._create_download_tasks(requested_range)

        if not tasks:
            logger.warning("No download tasks created - no data available")
            return False

        # Log tasks
        logger.info(f"Created {len(tasks)} download task(s):")
        for i, task in enumerate(tasks, 1):
            logger.info(f"  {i}. {task}")

        self._warn_if_rep_updated(pd.Timestamp(self.get_rep_availability().end))

        if dry_run:
            logger.info("DRY RUN - no downloads executed")
            self._cleanup_empty_download_dir()
            return False

        base_dir = output_dir or self.download_dir

        rep_paths = [path for path, source in tasks if source == "rep"]
        nrt_paths = [path for path, source in tasks if source == "nrt"]

        rep_dir = base_dir / "rep"
        nrt_dir = base_dir / "nrt"
        if rep_paths:
            rep_dir.mkdir(parents=True, exist_ok=True)
        if nrt_paths:
            nrt_dir.mkdir(parents=True, exist_ok=True)

        if parallel:
            if rep_paths:
                logger.info(
                    f"Starting parallel download of {len(rep_paths)} REP files..."
                )
                self.download_parallel(
                    rep_paths,
                    dataset_id=self.var_config.dataset_id_rep,
                    output_dir=rep_dir,
                    max_workers=max_workers,
                )
            if nrt_paths and self.var_config.dataset_id_nrt:
                logger.info(
                    f"Starting parallel download of {len(nrt_paths)} NRT files..."
                )
                self.download_parallel(
                    nrt_paths,
                    dataset_id=self.var_config.dataset_id_nrt,
                    output_dir=nrt_dir,
                    max_workers=max_workers,
                )
        else:
            for path, source in tasks:
                dest = rep_dir if source == "rep" else nrt_dir
                dataset_id = (
                    self.var_config.dataset_id_rep
                    if source == "rep"
                    else self.var_config.dataset_id_nrt
                )
                if dataset_id:
                    self.adjust_ftp_path_to_dataset(dataset_id)
                self.download_file(path, dest)

        # Disconnect FTP
        # self.ftp.quit()
        logger.success(f"Key-Variable: {self.var_key.upper()} processed!")
        self._cleanup_empty_download_dir()
        return True
