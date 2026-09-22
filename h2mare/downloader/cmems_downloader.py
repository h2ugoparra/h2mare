"""
CMEMS data downloader using Copernicus Marine Toolbox API.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import copernicusmarine
import pandas as pd
from loguru import logger

from h2mare import AppConfig
from h2mare.downloader.base import BaseDownloader
from h2mare.downloader.cmems_utils import CMEMSAPIError, get_dataset_time_range
from h2mare.models import step_freq
from h2mare.storage import split_time_range
from h2mare.storage.coverage import resolve_date_range
from h2mare.types import DateLike, DateRange, DownloadTask, FilePeriod
from h2mare.utils.datetime_utils import end_of_day


def download_subset(
    dataset_id: str,
    start: DateLike,
    end: DateLike,
    output_dir: str | Path,
    *,
    variables: str | list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    depth_range: tuple[float, float] | None = None,
) -> None:
    """
    Download a CMEMS dataset subset directly via copernicusmarine.subset.

    Does not require a CMEMSDownloader instance — all parameters are explicit.

    Args:
        dataset_id:  CMEMS dataset ID (e.g. 'METOFFICE-GLO-SST-L4-REP-OBS-SST').
        start:       Start datetime, sent to the API as given.
        end:         End datetime, sent to the API as given — it is *not*
                     widened to cover the day. On a sub-daily dataset a
                     midnight-stamped end therefore delivers only that day's
                     00:00 step; pass ``end_of_day(end)`` to get all 24.
        output_dir:  Directory to write the downloaded file.
        variables:   Variable name(s) to download. None downloads all.
        bbox:        (xmin, ymin, xmax, ymax). None = full geographic extent.
        depth_range: (min_depth, max_depth). None = no depth constraint.

    Example:
        >>> from h2mare.downloader import cmems_download_subset
        >>> cmems_download_subset(
        ...     dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        ...     start="2024-01-01",
        ...     end="2024-01-03",
        ...     output_dir="data/raw/downloads/sst",
        ...     variables=["analysed_sst"],
        ...     bbox=(-80, 0, -10, 70),
        ... )
    """
    if isinstance(variables, str):
        variables = [variables]

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=bbox[0] if bbox else None,
        maximum_longitude=bbox[2] if bbox else None,
        minimum_latitude=bbox[1] if bbox else None,
        maximum_latitude=bbox[3] if bbox else None,
        start_datetime=start_ts.isoformat(),
        end_datetime=end_ts.isoformat(),
        minimum_depth=depth_range[0] if depth_range else None,
        maximum_depth=depth_range[1] if depth_range else None,
        output_directory=output_dir,
        # Overwrite an existing target instead of the toolbox default, which
        # writes a new 'filename_(1).nc'. Gaps are recomputed from catalog
        # coverage, so a re-triggered download must cleanly replace any stale
        # or partially written file from a previous failed run rather than
        # leaving duplicates for the convert step to glob.
        overwrite=True,
        # The progress bar redraws via carriage returns; redirected to a file
        # (scheduled runs) every refresh persists as its own line.
        disable_progress_bar=not sys.stderr.isatty(),
    )
    logger.debug(f"Downloaded to {output_dir}")


def _generate_date_patterns(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> list[str]:
    """Generate copernicusmarine.get filter patterns for a date range within the same month."""
    patterns = []
    year, month = start_date.year, start_date.month
    start_day, end_day = start_date.day, end_date.day
    current_day = start_day

    while current_day <= end_day:
        tens_digit = current_day // 10
        ones_start = current_day % 10
        max_in_tens = min((tens_digit + 1) * 10 - 1, end_day, 31)
        ones_end = max_in_tens % 10

        if current_day == max_in_tens:
            patterns.append(f"*{year}{month:02d}{current_day:02d}*")
        elif tens_digit == max_in_tens // 10:
            if ones_start == 0 and ones_end == 9:
                patterns.append(f"*{year}{month:02d}{tens_digit}[0-9]*")
            else:
                patterns.append(
                    f"*{year}{month:02d}{tens_digit}[{ones_start}-{ones_end}]*"
                )

        current_day = max_in_tens + 1

    return patterns


def generate_copernicus_patterns(
    start: str | pd.Timestamp, end: str | pd.Timestamp
) -> list[str]:
    """
    Generate copernicusmarine.get filter patterns for a date range.

    Produces efficient year/month/day glob patterns, using full-year or full-month
    shortcuts when applicable.

    Examples:
        >>> generate_copernicus_patterns("2023-01-21", "2023-01-23")
        ['*2023012[1-3]*']
        >>> generate_copernicus_patterns("2023-01-01", "2023-01-31")
        ['*2023/01/*']
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    patterns = []

    if (
        start_ts.day == 1
        and end_ts.day == end_ts.days_in_month
        and start_ts.year == end_ts.year
        and start_ts.month == end_ts.month
    ):
        return [f"*{start_ts.year}/{start_ts.month:02d}/*"]

    if (
        start_ts.day == 1
        and start_ts.month == 1
        and end_ts.day == 31
        and end_ts.month == 12
        and start_ts.year == end_ts.year
    ):
        return [f"*{start_ts.year}/*"]

    current_year = start_ts.year
    while current_year <= end_ts.year:
        year_start = (
            start_ts
            if current_year == start_ts.year
            else pd.Timestamp(current_year, 1, 1)
        )
        year_end = (
            end_ts
            if current_year == end_ts.year
            else pd.Timestamp(current_year, 12, 31)
        )

        if (
            year_start.day == 1
            and year_start.month == 1
            and year_end.day == 31
            and year_end.month == 12
        ):
            patterns.append(f"*{current_year}/*")
        else:
            current = year_start
            while current <= year_end:
                month_start = current
                month_end = min(
                    year_end,
                    pd.Timestamp(current.year, current.month, current.days_in_month),
                )
                if month_start.day == 1 and month_end.day == month_start.days_in_month:
                    patterns.append(f"*{month_start.year}/{month_start.month:02d}/*")
                else:
                    patterns.extend(_generate_date_patterns(month_start, month_end))
                current = month_end + timedelta(days=1)

        current_year += 1

    return patterns


def download_original(
    dataset_id: str,
    start: DateLike,
    end: DateLike,
    output_dir: str | Path,
) -> None:
    """
    Download CMEMS original files (full geographic extent) via copernicusmarine.get.

    Args:
        dataset_id: CMEMS dataset ID.
        start:      Start date.
        end:        End date.
        output_dir: Directory to write downloaded files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pattern in generate_copernicus_patterns(pd.Timestamp(start), pd.Timestamp(end)):
        logger.info(f"Downloading pattern {pattern} to {output_dir}")
        copernicusmarine.get(
            dataset_id=dataset_id,
            filter=pattern,
            output_directory=output_dir,
            no_directories=True,
            # Overwrite stale/partial files from a previous failed run instead
            # of the toolbox default ('filename_(1).nc' duplicates). See note
            # in download_subset.
            overwrite=True,
            disable_progress_bar=not sys.stderr.isatty(),
        )
    logger.debug(f"Downloaded to {output_dir}")


class CMEMSDownloader(BaseDownloader):
    """
    Downloads Copernicus Marine products via the ``copernicusmarine`` client.

    Registered for ``source: cmems``, and the source behind most variables.
    Credentials are held by the ``copernicusmarine`` CLI (``copernicusmarine
    login``); nothing here reads them.

    A variable's ``subset`` flag picks the API: ``subset()`` honours the
    configured bbox and ``source_vars``, while ``get()`` fetches whole original
    files. Where a variable configures both a reprocessed (REP) and a
    near-real-time (NRT) dataset id, the recent tail comes from NRT and is
    superseded by REP once that catches up.

    Downloads are recorded in a manifest the convert step reads to stamp
    provenance onto the resulting Zarr.
    """

    def __init__(
        self,
        var_key: str,
        *,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        download_root: Optional[Path] = None,
    ) -> None:
        """
        Initialize CMEMS downloader. Handles both reprocessed (REP) and near-real-time (NRT) datasets.

        Args:
            var_key: Variable key from app_config.variables.
            app_config: Application configuration. If None, loads from settings.
            store_root: Root directory for zarr files. If None, uses settings.STORE_ROOT.
            download_root: Root directory for downloads. If None, uses settings.DOWNLOADS_DIR.

        Example:
            >>> downloader = CMEMSDownloader("ssh")
            >>> downloader.run(start_date="2023-01-01", end_date="2023-12-31")
        """
        super().__init__(
            var_key,
            app_config=app_config,
            store_root=store_root,
            download_root=download_root,
        )

        self._rep_availability: Optional[DateRange] = None
        self._nrt_availability: Optional[DateRange] = None

    # ==================== Dates and data coverage Resolution ====================

    def _last_complete_day(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Timestamp:
        """
        Day-level availability end: the last day the provider has published whole.

        A trailing *partial* day is the one case where taking the day the last
        stamp falls in loses data for good. Store coverage is day-granular, so a
        day ingested six hours old counts as covered and its remaining eighteen
        hours are never requested again. Held back, it is fetched whole once the
        provider finishes publishing it.

        Only the tail behaves this way. A partial *first* day is partial at the
        source forever, so it is taken as it is rather than skipped — hence the
        clamp to *start*, which also keeps a product whose entire history is one
        partial day from resolving to an empty range.

        Daily variables are unaffected: one step per day means the day holding
        the last stamp is complete by definition.
        """
        day = end.normalize()
        if step_freq(self.var_config) != "h":
            return day

        # Complete when the day's final step is present. Written against the
        # cadence rather than as a bare `hour == 23` so that adding a TimeStep
        # member — a 3-hourly product, say — has to come back through here.
        last_step_of_day = day + pd.Timedelta(hours=23)
        if end >= last_step_of_day:
            return day
        logger.info(
            f"[{self.var_key}] {day.date()} is still being published "
            f"(last stamp {end}); holding it back until it is complete."
        )
        return max(day - pd.Timedelta(days=1), start.normalize())

    def _get_dataset_availability(self, dataset_id: str) -> DateRange:
        """Query CMEMS API for dataset time range."""
        try:
            start, end = get_dataset_time_range(dataset_id)
            return DateRange(start=start, end=self._last_complete_day(start, end))
        except CMEMSAPIError as e:
            logger.error(f"Failed to get dataset availability: {e}")
            raise

    def get_rep_availability(self) -> DateRange:
        """Get REP dataset availability (cached after the first API call)."""
        if self._rep_availability is None:
            self._rep_availability = self._get_dataset_availability(
                self.var_config.dataset_id_rep
            )
        return self._rep_availability

    def get_nrt_availability(self) -> Optional[DateRange]:
        """Get near-real-time dataset availability (cached)."""
        if (
            not hasattr(self.var_config, "dataset_id_nrt")
            or self.var_config.dataset_id_nrt is None
        ):
            logger.debug("NRT dataset not available")
            return None

        if self._nrt_availability is None:
            self._nrt_availability = self._get_dataset_availability(
                self.var_config.dataset_id_nrt
            )

        return self._nrt_availability

    def _create_download_tasks(
        self,
        requested_range: DateRange,
    ) -> list[DownloadTask]:
        """
        Split date range into download tasks based on dataset availability.

        Strategy:
            - Uses REP for historical data up to its end date.
            - NRT dataset for any remaining period (if available).

        Args:
            requested_range: Date range to download

        Returns:
            List of DownloadTask objects
        """
        tasks = []

        # Get dataset availability
        rep_avail = self.get_rep_availability()
        nrt_avail = self.get_nrt_availability()

        # REP covers the requested range up to its end date
        rep_overlap = requested_range.intersection(rep_avail)
        if rep_overlap:
            tasks.append(
                DownloadTask(
                    dataset_id=self.var_config.dataset_id_rep,
                    date_range=rep_overlap,
                    dataset_type="rep",
                )
            )

        # NRT covers anything beyond REP's end date, if available
        if nrt_avail and self.var_config.dataset_id_nrt:
            nrt_start = (
                rep_avail.end + pd.Timedelta(days=1)
                if rep_overlap
                else requested_range.start
            )
            # Guard: if REP already covers the full requested range, nrt_start falls past
            # requested_range.end — no NRT period remains
            if nrt_start <= requested_range.end:
                nrt_request = DateRange(start=nrt_start, end=requested_range.end)
                nrt_overlap = nrt_request.intersection(nrt_avail)
                if nrt_overlap:
                    tasks.append(
                        DownloadTask(
                            dataset_id=self.var_config.dataset_id_nrt,
                            date_range=nrt_overlap,
                            dataset_type="nrt",
                        )
                    )

        return tasks

    # ==================== Download Execution ====================
    def run(
        self,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        time_split: FilePeriod = FilePeriod.MONTH,
    ) -> bool:
        """
        Run download for specified date range.

        Args:
            start_date: Start date (None = use default or infer)
            end_date: End date (None = use default or infer)
            output_dir: Optional directory to save downloads (defaults to self.download_dir)
            dry_run: If True, plan download but don't execute
            time_split: How to split downloads ('monthly' or 'yearly')

        Example:
            >>> downloader = CMEMSDownloader("ssh")
            >>>
            >>> # Download specific range
            >>> files = downloader.run("2023-01-01", "2023-12-31")
            >>>
            >>> # Dry run to see what would be downloaded
            >>> downloader.run("2023-01-01", "2023-12-31", dry_run=True)

        Returns:
            True if downloads were executed, False if skipped (no tasks or dry run).
        """

        # Resolve date range
        requested_range = resolve_date_range(
            self.var_key, start=start_date, end=end_date
        )
        if requested_range is None:
            logger.info(f"'{self.var_key}' is already up to date — skipping.")
            return False

        # Create download tasks
        tasks = self._create_download_tasks(requested_range)

        if not tasks:
            # Without an NRT dataset, get_nrt_availability re-logs "not available"
            # on every call; _create_download_tasks has already said so once.
            nrt_avail = (
                self.get_nrt_availability() if self.var_config.dataset_id_nrt else None
            )
            self._log_nothing_to_download(
                requested_range, self.get_rep_availability(), nrt_avail
            )
            return False

        # A task spans a whole dataset/date-range (one per rep/nrt source), not
        # one per file — log each task's range so the plan is readable.
        for task in tasks:
            logger.debug(
                f"Download task ({task.dataset_type}): {task.dataset_id} "
                f"covering {task.date_range}"
            )

        self._warn_if_rep_updated(pd.Timestamp(self.get_rep_availability().end))

        if dry_run:
            logger.info("DRY RUN - no downloads executed")
            self._cleanup_empty_download_dir()
            return False

        t0 = time.perf_counter()
        for task in tasks:
            self._execute_task(task=task, time_split=time_split, output_dir=output_dir)

            logger.debug(
                f"Task complete: {task.dataset_id} "
                f"for period {task.date_range} downloaded"
            )

        self._write_manifest(tasks, output_dir or self.download_dir)
        self._cleanup_empty_download_dir()
        logger.success(
            f"Download complete: {len(tasks)} task(s) "
            f"({requested_range.start.date()} → {requested_range.end.date()}) "
            f"in {time.perf_counter() - t0:.1f}s"
        )
        return True

    def _write_manifest(self, tasks: list[DownloadTask], output_dir: Path) -> None:
        """Write a JSON manifest recording which dataset_id/type covered each date range."""
        records = [
            {
                "dataset_id": t.dataset_id,
                "dataset_type": t.dataset_type,
                "start": t.date_range.start.strftime("%Y-%m-%d"),
                "end": t.date_range.end.strftime("%Y-%m-%d"),
            }
            for t in tasks
        ]
        manifest_path = output_dir / "h2mare_manifest.json"
        manifest_path.write_text(json.dumps(records, indent=2))
        logger.debug(f"Wrote download manifest to {manifest_path}")

    def _execute_task(
        self,
        task: DownloadTask,
        time_split: FilePeriod,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Execute a single download task.

        Args:
            task: Download task to execute
            time_split: Time splitting strategy
            force: Force redownload

        Returns:
            List of downloaded file paths
        """

        if self.var_config.subset:
            chunks = split_time_range(task.date_range, time_split)

            for i, chunk in enumerate(chunks, 1):
                # The toolbox's own lines name the dataset but not the dates,
                # so a long window gave no sign of how far it had got.
                if len(chunks) > 1:
                    logger.info(
                        f"Downloading chunk {i}/{len(chunks)}: "
                        f"{chunk.start.date()} → {chunk.end.date()}"
                    )
                self._retry_call(
                    self.download_subset,
                    task.dataset_id,
                    pd.to_datetime(chunk.start),
                    pd.to_datetime(chunk.end),
                    output_dir,
                )
        else:
            self._retry_call(
                self.download_original,
                task.dataset_id,
                pd.to_datetime(task.date_range.start),
                pd.to_datetime(task.date_range.end),
                output_dir,
            )

    def download_subset(
        self,
        dataset_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Download using copernicusmarine.subset, deriving spatial/variable config
        from var_config.

        *end* is a date, and the toolbox reads it as an instant. On an hourly
        variable that means the last day of every chunk arrives as its 00:00
        step alone and the other 23 hours are simply never requested — a whole-
        month split loses 23 hours per month, quietly: the loss sits at the tail
        of the converted span, where ``_verify_written_dates`` only warns.
        Widening the bound to the end of the day is what makes a chunk boundary
        land between two days instead of inside one.

        Left untouched on a daily variable, whose stores were all written under
        the midnight bound and whose one step per day it already covers.
        """
        if step_freq(self.var_config) == "h":
            end = end_of_day(end)

        download_subset(
            dataset_id=dataset_id,
            start=start,
            end=end,
            output_dir=output_dir or self.download_dir,
            variables=self.var_config.source_vars,
            bbox=getattr(self.var_config, "bbox", None),
            depth_range=getattr(self.var_config, "depth_range", None),
        )

    def generate_copernicus_patterns(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> list[str]:
        """Delegate to module-level ``generate_copernicus_patterns``."""
        return generate_copernicus_patterns(start, end)

    def _generate_date_patterns(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> list[str]:
        """Delegate to module-level ``_generate_date_patterns``."""
        return _generate_date_patterns(start_date, end_date)

    def download_original(
        self,
        dataset_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Delegate to module-level ``download_original``."""
        download_original(dataset_id, start, end, output_dir or self.download_dir)
