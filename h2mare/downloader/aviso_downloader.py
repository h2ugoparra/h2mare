from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ftplib import FTP
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from h2mare.config import AppConfig
from h2mare.downloader.base import BaseDownloader
from h2mare.storage.coverage import resolve_date_range
from h2mare.types import DateLike, DateRange, FTPDownloadTask


def _download_to_part(
    local_path: Path, ftp: FTP, remote_path: str, file_size: Optional[int]
) -> None:
    """
    Retrieve *remote_path* into *local_path*, via a sidecar renamed on success.

    Writing straight to ``local_path`` would leave a truncated ``*.nc`` behind
    when a connection drops mid-``RETR``. The convert step globs ``*.nc`` and
    has no way to tell a partial file from a complete one, so the damage
    surfaces much later as missing or corrupt days inside a year that every
    coverage check reports as healthy.
    """
    part_path = local_path.with_name(local_path.name + ".part")
    try:
        with open(part_path, "wb") as f:
            if file_size:
                # disable=None: tqdm auto-disables on non-tty, so scheduled
                # runs don't persist one log line per refresh tick.
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=remote_path.split("/")[-1],
                    disable=None,
                ) as pbar:

                    def callback(data):
                        f.write(data)
                        pbar.update(len(data))

                    ftp.retrbinary(f"RETR {remote_path}", callback)
            else:
                ftp.retrbinary(f"RETR {remote_path}", f.write)
        os.replace(part_path, local_path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise


class AVISODownloader(BaseDownloader):
    """
    Downloads AVISO products over FTP (FSLE, eddy trajectories).

    Registered for ``source: aviso``. Unlike the API-based downloaders, the
    variable's ``dataset_id`` is an FTP *path*: the server is the root and the
    id selects a directory under it. Files are listed, matched to dates by the
    variable's filename ``pattern``, and fetched with retry and reconnection —
    a dropped control connection is expected on long transfers.

    Where a variable configures both, delayed-time (REP) and near-real-time
    (NRT) trees are fetched into separate staging directories, so the convert
    step can prefer REP where it exists.

    Credentials come from ``AVISO_USERNAME``/``AVISO_PASSWORD``/
    ``AVISO_FTP_SERVER`` in ``.env``; a missing one raises rather than
    attempting an anonymous login.
    """

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

    def close(self) -> None:
        """Close the FTP control connection opened in __init__.

        Without this the connection survives until the interpreter collects it,
        which Python reports as ``ResourceWarning: unclosed <socket.socket ...>``
        — invisible by default, since ResourceWarning is ignored unless asked
        for. `quit()` sends QUIT and is the polite close, but it raises if the
        server already went away, so fall back to `close()`, which only drops
        the local handle. Idempotent: closing twice is not an error.
        """
        ftp = getattr(self, "ftp", None)
        if ftp is None:
            return
        try:
            ftp.quit()
        except Exception:
            try:
                ftp.close()
            except Exception:
                pass
        finally:
            self.ftp = None  # type: ignore[assignment]

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
        """
        Connect to the AVISO FTP server.

        Unencrypted, and not by choice. ``ftp-access.aviso.altimetry.fr`` offers
        plain FTP only: its ``FEAT`` reply advertises no ``AUTH``, and an
        explicit ``AUTH TLS`` is refused with *"500 AUTH not understood"*
        (checked 2026-09-01). So the credentials and the transfers both cross
        the network in cleartext, and there is no flag here that changes it —
        see the warning beside ``AVISO_USERNAME`` in ``.env.template``.

        If AVISO enables AUTH TLS later, this becomes ``FTP_TLS`` plus a
        ``prot_p()`` after login; nothing else in this class depends on which
        of the two it is.
        """
        # FTP Connection - for AVISO data, dataset_id represents the ftp path.
        # server is the root of the path
        ftp_server = self.app_config.secrets.aviso_ftp_server
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
        return ftp

    def adjust_ftp_path_to_dataset(self, dataset_id: str) -> FTP:
        self.ftp.cwd("/")
        self.ftp.cwd(dataset_id)  # adjust path
        self._current_dataset_id = dataset_id
        return self.ftp

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
        if self.var_config.pattern is None:
            raise ValueError(
                f"Variable '{self.var_key}' has no `pattern`; cannot extract dates "
                "from filenames"
            )
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
    ) -> list[FTPDownloadTask]:
        """
        Split date range into download tasks based on dataset availability.

        Strategy:
            - Uses REP for historical data up to its end date.
            - NRT dataset for any remaining period (if available).

        Args:
            requested_range: Date range to download

        Returns:
            List of FTPDownloadTask objects.
        """
        tasks: list[FTPDownloadTask] = []

        rep_files = self._get_dataset_files(self.var_config.dataset_id_rep)
        rep_avail = self._get_dataset_availability(rep_files)

        nrt_files = []
        nrt_avail = None
        if self.var_config.dataset_id_nrt:
            nrt_files = self._get_dataset_files(self.var_config.dataset_id_nrt)
            nrt_avail = self._get_dataset_availability(nrt_files)

        # REP covers the requested range up to its end date
        rep_overlap = requested_range.intersection(rep_avail)
        if rep_overlap:
            rep_files = self._filter_files_by_range(rep_files, rep_overlap)
            tasks.extend(FTPDownloadTask(filepath=fp, source="rep") for fp in rep_files)

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
                tasks.extend(
                    FTPDownloadTask(filepath=fp, source="nrt") for fp in nrt_files
                )

        return tasks

    def _task_date_range(self, paths: list[str]) -> Optional[DateRange]:
        """Span covered by *paths*, derived from their filenames.

        Handles both filename shapes: a single date (fsle) and a start/end pair
        (eddies trajectory files).
        """
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        for filepath in paths:
            dates = self._extract_date_from_filename(filepath)
            if isinstance(dates, tuple):
                starts.append(dates[0])
                ends.append(dates[1])
            else:
                starts.append(dates)
                ends.append(dates)

        if not starts:
            return None
        return DateRange(start=min(starts), end=max(ends))

    def _write_manifest(self, tasks: list[FTPDownloadTask], output_dir: Path) -> None:
        """Record which dataset covered each downloaded date range.

        ``Netcdf2Zarr`` reads this manifest to stamp ``source_datasets`` onto the
        Zarr it builds. Without it that step silently no-ops and the catalog
        scanner falls back to ``dataset_id_rep``, labelling near-real-time data
        as delayed-time.

        Ranges come from the filenames of the *planned* tasks rather than from
        the caller's requested range, so an incremental run records only the
        gap it set out to fill instead of claiming the whole history. Note this
        is the planned set, not the delivered one — a task whose download failed
        still widens the span, which is what lets the convert step notice that
        a date the manifest covers never made it into the store.
        """
        records = []
        for source, dataset_id in (
            ("rep", self.var_config.dataset_id_rep),
            ("nrt", self.var_config.dataset_id_nrt),
        ):
            if dataset_id is None:
                continue
            span = self._task_date_range(
                [t.filepath for t in tasks if t.source == source]
            )
            if span is None:
                continue
            records.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_type": source,
                    "start": span.start.strftime("%Y-%m-%d"),
                    "end": span.end.strftime("%Y-%m-%d"),
                }
            )

        if not records:
            return

        manifest_path = output_dir / "h2mare_manifest.json"
        manifest_path.write_text(json.dumps(records, indent=2))
        logger.debug(f"Wrote download manifest to {manifest_path}")

    # ==================== Download Execution ====================
    def download_file(self, path: str, output_dir: Optional[Path] = None) -> None:
        """Download individual files from FTP, with retry and automatic reconnection."""
        local_path = (output_dir or self.download_dir) / path.split("/")[-1]
        logger.debug(f"Downloading {path} to {local_path}")

        def _attempt() -> None:
            # Reconnect if the connection is no longer alive.
            try:
                self.ftp.voidcmd("NOOP")
            except Exception:
                logger.debug("FTP connection lost — reconnecting")
                # Drop the dead socket before replacing it. `quit()` would try
                # to QUIT over a connection that just failed NOOP, so close()
                # is what actually releases the handle; without this each
                # reconnect strands one more socket for the GC to complain
                # about (ResourceWarning, hidden by default).
                try:
                    self.ftp.close()
                except Exception:
                    pass
                self.ftp = self.connect_ftp()
                dataset_id = getattr(self, "_current_dataset_id", None)
                if dataset_id:
                    self.adjust_ftp_path_to_dataset(dataset_id)

            try:
                self.ftp.voidcmd("TYPE I")
                file_size = self.ftp.size(path)
            except Exception as e:
                logger.warning(f"Error getting file size for {path}: {e}")
                file_size = None

            _download_to_part(local_path, self.ftp, path, file_size)

        self._retry_call(_attempt, max_attempts=3, wait_min=10, wait_max=60)
        logger.success(f"Downloaded {path.split('/')[-1]} to {local_path}")

    def download_parallel(
        self,
        paths: list[str],
        dataset_id: str,
        output_dir: Optional[Path] = None,
        max_workers: int = 2,
    ) -> list[str]:
        """
        Download multiple FSLE files in parallel using multiple FTP connections.

        Returns:
            The remote paths that could not be downloaded after all retries.

        Callers must treat a non-empty result as a failed download. A file that
        never arrives leaves no trace anywhere downstream: store coverage is a
        min/max watermark that a one-day hole cannot move, and the convert step
        derives its expectations from the files that *did* arrive, so it writes
        a short year and reports success.
        """
        output_dir = output_dir or self.download_dir

        def download_single(
            path: str, dataset_id: str = dataset_id, output_dir: Path = output_dir
        ):
            # Each retry creates a fresh FTP connection — safe without reconnect logic.
            def _attempt() -> None:
                ftp = self.connect_ftp()
                ftp.cwd(dataset_id)
                try:
                    local_path = output_dir / path.split("/")[-1]
                    try:
                        ftp.voidcmd("TYPE I")
                        file_size = ftp.size(path)
                    except Exception as e:
                        logger.warning(f"Error getting file size for {path}: {e}")
                        file_size = None

                    _download_to_part(local_path, ftp, path, file_size)
                finally:
                    try:
                        ftp.quit()
                    except Exception:
                        pass

            self._retry_call(_attempt, max_attempts=3, wait_min=10, wait_max=60)
            return path

        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_single, path): path for path in paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"❌ Failed to download {path}: {e}")
                    failed.append(path)
        return failed

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
        requested_range = resolve_date_range(
            self.var_key, start=start_date, end=end_date
        )
        if requested_range is None:
            logger.info(f"'{self.var_key}' is already up to date — skipping.")
            return False

        tasks = self._create_download_tasks(requested_range)

        if not tasks:
            self._log_nothing_to_download(
                requested_range,
                self.get_rep_availability(),
                self.get_nrt_availability(),
            )
            return False

        logger.debug(f"Created {len(tasks)} download task(s)")

        self._warn_if_rep_updated(pd.Timestamp(self.get_rep_availability().end))

        if dry_run:
            logger.info("DRY RUN - no downloads executed")
            self._cleanup_empty_download_dir()
            return False

        base_dir = output_dir or self.download_dir
        t0 = time.perf_counter()

        rep_paths = [t.filepath for t in tasks if t.source == "rep"]
        nrt_paths = [t.filepath for t in tasks if t.source == "nrt"]

        rep_dir = base_dir / "rep"
        nrt_dir = base_dir / "nrt"
        if rep_paths:
            rep_dir.mkdir(parents=True, exist_ok=True)
        if nrt_paths:
            nrt_dir.mkdir(parents=True, exist_ok=True)

        failed: list[str] = []
        if parallel:
            if rep_paths:
                logger.info(
                    f"Starting parallel download of {len(rep_paths)} REP files..."
                )
                failed += self.download_parallel(
                    rep_paths,
                    dataset_id=self.var_config.dataset_id_rep,
                    output_dir=rep_dir,
                    max_workers=max_workers,
                )
            if nrt_paths and self.var_config.dataset_id_nrt:
                logger.info(
                    f"Starting parallel download of {len(nrt_paths)} NRT files..."
                )
                failed += self.download_parallel(
                    nrt_paths,
                    dataset_id=self.var_config.dataset_id_nrt,
                    output_dir=nrt_dir,
                    max_workers=max_workers,
                )
        else:
            for task in tasks:
                dest = rep_dir if task.source == "rep" else nrt_dir
                dataset_id = (
                    self.var_config.dataset_id_rep
                    if task.source == "rep"
                    else self.var_config.dataset_id_nrt
                )
                if dataset_id:
                    self.adjust_ftp_path_to_dataset(dataset_id)
                try:
                    self.download_file(task.filepath, dest)
                except Exception as e:
                    logger.error(f"❌ Failed to download {task.filepath}: {e}")
                    failed.append(task.filepath)

        # The FTP connection is *not* closed here: this object stays usable
        # after a download (availability lookups re-use self.ftp), so the
        # owner decides when it dies — see close(), which PipelineManager
        # calls in a finally.
        # Written even on failure, and deliberately: the manifest records the
        # range that was *requested*, which is the only surviving statement of
        # what the store should contain once the raw files are archived or
        # deleted. Reconciling it against what was delivered is what turns a
        # partial download into a detectable defect rather than a short year.
        self._write_manifest(tasks, base_dir)
        self._cleanup_empty_download_dir()

        requested = len(rep_paths) + len(nrt_paths)
        if failed:
            raise RuntimeError(
                f"[{self.var_key}] Download incomplete: {len(failed)} of {requested} "
                f"file(s) failed after retries — {', '.join(sorted(failed)[:10])}"
                f"{' …' if len(failed) > 10 else ''}. Re-run to fetch the missing "
                f"dates; converting now would write a store with silent gaps."
            )

        logger.success(
            f"Download complete: {requested}/{requested} file(s) "
            f"({len(rep_paths)} REP + {len(nrt_paths)} NRT) "
            f"in {time.perf_counter() - t0:.1f}s"
        )
        return True
