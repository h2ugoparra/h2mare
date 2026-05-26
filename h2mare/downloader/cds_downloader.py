"""
Download climate data from Copernicus ECMWF CLimate Data Store (CLS)
Go to https://cds.climate.copernicus.eu/datasets to check API request code

"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import cdsapi
import pandas as pd
from loguru import logger

from h2mare.config import AppConfig
from h2mare.downloader.base import BaseDownloader
from h2mare.storage import get_store_coverage, split_time_range
from h2mare.types import BBox, DateLike, DateRange, TimeResolution
from h2mare.utils.datetime_utils import normalize_date
from h2mare.utils.labels import create_filename_label

warnings.filterwarnings("ignore")


class CDSDownloader(BaseDownloader):
    def __init__(
        self,
        var_key: str,
        *,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        download_root: Optional[Path] = None,
    ):
        """
        Initializes CDS-ERA5 downloader.

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

    def run(
        self,
        start_date: Optional[DateLike] = None,
        end_date: Optional[DateLike] = None,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        time_split: TimeResolution = TimeResolution.MONTH,
    ) -> bool:
        """
        Run download for specified date range.

        Args:
            start_date: Start date (None = use default or infer)
            end_date: End date (None = use default or infer)
            output_dir: Optional directory to save downloads (defaults to self.download_dir)
            dry_run: If True, plan download but don't execute
            time_split: How to split downloads ('monthly' or 'yearly')

        Returns:
            True if downloads were executed, False if skipped (no tasks or dry run).
        """

        # Resolve date range
        requested_range = self._resolve_date_range(start_date, end_date)

        if not requested_range:
            logger.warning("No download tasks created.")
            return False

        logger.info(
            f"Downloading for dates: {requested_range.start} -> {requested_range.end}\n"
        )

        splits = split_time_range(requested_range, time_split)

        if not splits:
            logger.warning("No download tasks created - no data available")
            return False

        logger.info(f"Created {len(splits)} download task(s):")
        for i, dt in enumerate(splits, 1):
            logger.info(f"  {i}. {DateRange(start=dt.start, end=dt.end)}")

        if dry_run:
            logger.info("DRY RUN - no downloads executed")
            self._cleanup_empty_download_dir()
            return False

        for dt in splits:
            dt = DateRange(start=dt.start, end=dt.end)
            self.download_file(dt, output_dir=output_dir)

        self._cleanup_empty_download_dir()
        return True

    def _resolve_date_range(
        self,
        start_date: DateLike | None,
        end_date: DateLike | None,
    ) -> DateRange | None:
        """
        Resolve storage date range for download. Download monthly files by convention if date not specified.

        Priority:
            1. Explicit arguments
            2. Default dates from __init__
            3. Last day of the previous month if monthday +10 from today, else before previous month

        Args:
            start_date: Explicit start date
            end_date: Explicit end date

        Returns:
            Resolved DateRange

        Raises:
            ValueError: If dates cannot be resolved
        """
        # Use explicit args, fall back to defaults
        start = normalize_date(start_date) if start_date else None
        end = normalize_date(end_date) if end_date else None

        # If still None, try to infer from store
        if start is None or end is None:
            store_coverage = get_store_coverage(self.var_key)

            if store_coverage is None:
                raise ValueError(
                    f"No existing data found for '{self.var_key}'. "
                    f"Please provide start_date and end_date explicitly."
                )

            # Default: download from day after latest stored data 10 days previous to today
            if start is None:
                start = store_coverage.end + pd.Timedelta(days=1)

            if end is None:
                # If today's date is at +10 days from month start,
                # it get's the end of previous month
                now = pd.Timestamp.now().normalize()
                if now.day < 10:
                    end = now + pd.offsets.MonthEnd(-2)
                else:
                    end = now + pd.offsets.MonthEnd(-1)

            logger.info(
                f"Date range in store: {store_coverage.start.date()} -> {store_coverage.end.date()}"
            )

        # If True, means that store is up to date.
        if start > end:
            return None

        return DateRange(start=start, end=end)

    def download_file(
        self, date_range: DateRange, output_dir: Optional[Path] = None
    ) -> None:
        """
        Download files.

        Args:
            date_range (DateRange): Start and end dates. It only accepts a single year (defined in split_time_range func)
            output_dir (Optional[Path], optional): Output directory. Defaults to None (download_dir).
        """

        year = f"{date_range.start.year:04d}"
        full_year = (date_range.start == pd.Timestamp(year)) and (
            date_range.end == pd.Timestamp(year) + pd.offsets.YearEnd(0)
        )

        if full_year:
            months = [f"{m:02d}" for m in range(1, 13)]
            days = [f"{d:02d}" for d in range(1, 32)]
        else:
            months = [f"{date_range.start.month:02d}"]
            days = [
                f"{d:02d}" for d in range(date_range.start.day, date_range.end.day + 1)
            ]

        client = cdsapi.Client()

        if self.var_config.bbox is not None:
            bbox = BBox.from_tuple(self.var_config.bbox)

        request = {
            "product_type": ["reanalysis"],
            "variable": self.var_config.variables,
            "year": [year],
            "month": months,
            "day": days,  # [f"{d:02d}" for d in range(day_ini, day_fin + 1)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": [
                bbox.ymax if bbox else None,
                bbox.xmin if bbox else None,
                bbox.ymin if bbox else None,
                bbox.xmax if bbox else None,
            ],
        }

        geotime_label = create_filename_label(bbox, "date", date_range)
        out_dir = output_dir or self.download_dir
        outfile = (
            out_dir
            / f"{self.var_config.dataset_id_rep}_{self.var_key}_{geotime_label}.grib"
        )

        self._retry_call(
            lambda: client.retrieve(self.var_config.dataset_id_rep, request).download(
                outfile
            )
        )
        logger.success(f"Downloaded {outfile.name} to {outfile.parent}")
