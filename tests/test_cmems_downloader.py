"""Tests for CMEMSDownloader pattern generation and task creation logic."""

import json
from unittest.mock import patch

import msgspec
import pandas as pd
import pytest

from h2mare.downloader.cmems_downloader import (
    CMEMSDownloader,
    _generate_date_patterns,
    generate_copernicus_patterns,
)
from h2mare.models import AppConfig
from h2mare.types import DateRange, DownloadTask, FilePeriod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTRY = {
    "local_folder": "sst",
    "source_vars": ["analysed_sst"],
    "dataset_id_rep": "cmems-rep-sst",
    "dataset_id_nrt": "cmems-nrt-sst",
    "source": "cmems",
    "archive_raw": False,
    "pattern": r".*\.nc",
    "subset": True,
    "bbox": (-80, 0, 10, 70),
}

_ENTRY_NO_NRT = {**_ENTRY, "dataset_id_nrt": None}
_ENTRY_NO_SUBSET = {**_ENTRY, "subset": False, "dataset_id_nrt": None}
_ENTRY_HOURLY = {**_ENTRY, "time_step": "hourly"}


def _make_config(entry=_ENTRY) -> AppConfig:
    return msgspec.convert({"variables": {"sst": entry}, "secrets": {}}, AppConfig)


@pytest.fixture
def dl(tmp_path):
    return CMEMSDownloader(
        "sst",
        app_config=_make_config(),
        store_root=tmp_path / "store",
        download_root=tmp_path,
    )


@pytest.fixture
def dl_no_subset(tmp_path):
    return CMEMSDownloader(
        "sst",
        app_config=_make_config(_ENTRY_NO_SUBSET),
        store_root=tmp_path / "store",
        download_root=tmp_path,
    )


@pytest.fixture
def dl_hourly(tmp_path):
    return CMEMSDownloader(
        "sst",
        app_config=_make_config(_ENTRY_HOURLY),
        store_root=tmp_path / "store",
        download_root=tmp_path,
    )


@pytest.fixture
def dl_no_nrt(tmp_path):
    return CMEMSDownloader(
        "sst",
        app_config=_make_config(_ENTRY_NO_NRT),
        store_root=tmp_path / "store",
        download_root=tmp_path,
    )


# ---------------------------------------------------------------------------
# generate_copernicus_patterns — pure function
# ---------------------------------------------------------------------------


class TestGenerateCopernicusPatterns:
    def test_full_month_returns_month_shortcut(self):
        assert generate_copernicus_patterns("2023-01-01", "2023-01-31") == [
            "*2023/01/*"
        ]

    def test_full_year_returns_year_shortcut(self):
        assert generate_copernicus_patterns("2023-01-01", "2023-12-31") == ["*2023/*"]

    def test_partial_range_within_month(self):
        assert generate_copernicus_patterns("2023-01-21", "2023-01-23") == [
            "*2023012[1-3]*"
        ]

    def test_multi_year_range_produces_per_month_patterns(self):
        result = generate_copernicus_patterns("2022-11-01", "2023-02-28")
        assert "*2022/11/*" in result
        assert "*2022/12/*" in result
        assert "*2023/01/*" in result
        assert "*2023/02/*" in result

    def test_full_decade_in_single_bracket(self):
        # 2023-01-20 to 2023-01-29 → tens=2, ones 0–9
        assert generate_copernicus_patterns("2023-01-20", "2023-01-29") == [
            "*2023012[0-9]*"
        ]

    def test_single_day_contains_full_date(self):
        result = generate_copernicus_patterns("2023-06-15", "2023-06-15")
        assert len(result) == 1
        assert "202306" in result[0]


# ---------------------------------------------------------------------------
# _generate_date_patterns — pure function
# ---------------------------------------------------------------------------


class TestGenerateDatePatterns:
    def test_full_decade_bracket(self):
        start = pd.Timestamp("2023-01-20")
        end = pd.Timestamp("2023-01-29")
        patterns = _generate_date_patterns(start, end)
        assert patterns == ["*2023012[0-9]*"]

    def test_partial_range_within_decade(self):
        start = pd.Timestamp("2023-01-21")
        end = pd.Timestamp("2023-01-23")
        patterns = _generate_date_patterns(start, end)
        assert patterns == ["*2023012[1-3]*"]

    def test_single_day_exact_match(self):
        day = pd.Timestamp("2023-01-05")
        patterns = _generate_date_patterns(day, day)
        assert len(patterns) == 1
        assert "20230105" in patterns[0]


# ---------------------------------------------------------------------------
# CMEMSDownloader._create_download_tasks
# ---------------------------------------------------------------------------

_REP_AVAIL = DateRange("2000-01-01", "2023-12-31")
_NRT_AVAIL = DateRange("2024-01-01", "2025-06-30")


class TestCreateDownloadTasks:
    def test_request_fully_within_rep(self, dl):
        with (
            patch.object(dl, "get_rep_availability", return_value=_REP_AVAIL),
            patch.object(dl, "get_nrt_availability", return_value=_NRT_AVAIL),
        ):
            tasks = dl._create_download_tasks(DateRange("2020-01-01", "2020-12-31"))

        assert len(tasks) == 1
        assert tasks[0].dataset_type == "rep"
        assert tasks[0].dataset_id == "cmems-rep-sst"

    def test_request_fully_within_nrt(self, dl):
        with (
            patch.object(dl, "get_rep_availability", return_value=_REP_AVAIL),
            patch.object(dl, "get_nrt_availability", return_value=_NRT_AVAIL),
        ):
            tasks = dl._create_download_tasks(DateRange("2024-06-01", "2025-01-31"))

        assert len(tasks) == 1
        assert tasks[0].dataset_type == "nrt"

    def test_request_spanning_rep_and_nrt(self, dl):
        with (
            patch.object(dl, "get_rep_availability", return_value=_REP_AVAIL),
            patch.object(dl, "get_nrt_availability", return_value=_NRT_AVAIL),
        ):
            tasks = dl._create_download_tasks(DateRange("2023-06-01", "2024-03-31"))

        assert len(tasks) == 2
        types = {t.dataset_type for t in tasks}
        assert types == {"rep", "nrt"}

    def test_no_overlap_with_any_dataset_returns_empty(self, dl):
        rep = DateRange("2000-01-01", "2010-12-31")
        with (
            patch.object(dl, "get_rep_availability", return_value=rep),
            patch.object(dl, "get_nrt_availability", return_value=None),
        ):
            tasks = dl._create_download_tasks(DateRange("2020-01-01", "2020-12-31"))

        assert tasks == []

    def test_no_nrt_configured_produces_only_rep_task(self, dl_no_nrt):
        with (
            patch.object(dl_no_nrt, "get_rep_availability", return_value=_REP_AVAIL),
            patch.object(dl_no_nrt, "get_nrt_availability", return_value=None),
        ):
            tasks = dl_no_nrt._create_download_tasks(
                DateRange("2020-01-01", "2020-12-31")
            )

        assert len(tasks) == 1
        assert tasks[0].dataset_type == "rep"

    def test_rep_task_date_range_is_clipped_to_availability(self, dl):
        rep = DateRange("2000-01-01", "2020-06-30")
        with (
            patch.object(dl, "get_rep_availability", return_value=rep),
            patch.object(dl, "get_nrt_availability", return_value=None),
        ):
            tasks = dl._create_download_tasks(DateRange("2020-01-01", "2021-12-31"))

        assert len(tasks) == 1
        assert pd.Timestamp(tasks[0].date_range.end) == pd.Timestamp("2020-06-30")


class TestNothingToDownloadMessage:
    """A request the provider cannot serve must not be reported as up to date."""

    @pytest.fixture
    def messages(self):
        from loguru import logger

        captured: list[str] = []
        sink = logger.add(captured.append, level="INFO", format="{level}|{message}")
        yield captured
        logger.remove(sink)

    def _run(self, dl, start, end, rep, nrt):
        with (
            patch.object(dl, "get_rep_availability", return_value=rep),
            patch.object(dl, "get_nrt_availability", return_value=nrt),
        ):
            return dl.run(start, end)

    def test_before_coverage_warns_instead_of_up_to_date(self, dl, messages):
        rep = DateRange("2022-06-01", "2026-09-25")
        assert self._run(dl, "1994-01-01", "1994-01-31", rep, None) is False

        text = "".join(messages)
        assert "up to date" not in text
        assert "WARNING|'sst': requested 1994-01-01 to 1994-01-31" in text
        assert "REP 2022-06-01 to 2026-09-25" in text

    def test_gap_between_rep_and_nrt_warns(self, dl, messages):
        rep = DateRange("2000-01-01", "2020-12-31")
        nrt = DateRange("2022-01-01", "2025-06-30")
        assert self._run(dl, "2021-03-01", "2021-03-31", rep, nrt) is False

        text = "".join(messages)
        assert "up to date" not in text
        assert "NRT 2022-01-01 to 2025-06-30" in text

    def test_nrt_not_available_is_logged_once(self, dl_no_nrt):
        from loguru import logger

        rep = DateRange("2022-06-01", "2026-09-25")
        messages: list[str] = []
        sink = logger.add(messages.append, level="DEBUG", format="{message}")
        try:
            with patch.object(dl_no_nrt, "get_rep_availability", return_value=rep):
                assert dl_no_nrt.run("1994-01-01", "1994-01-31") is False
        finally:
            logger.remove(sink)

        assert sum("NRT dataset not available" in m for m in messages) == 1

    def test_past_published_end_still_reports_up_to_date(self, dl, messages):
        assert (
            self._run(dl, "2025-07-01", "2025-07-31", _REP_AVAIL, _NRT_AVAIL) is False
        )

        text = "".join(messages)
        assert "INFO|'sst' is already up to date" in text
        assert "2025-06-30" in text
        assert "WARNING" not in text


# ---------------------------------------------------------------------------
# CMEMSDownloader._write_manifest
# ---------------------------------------------------------------------------


class TestWriteManifest:
    def test_creates_manifest_file(self, dl, tmp_path):
        tasks = [
            DownloadTask(
                dataset_id="cmems-rep-sst",
                date_range=DateRange("2020-01-01", "2020-06-30"),
                dataset_type="rep",
            )
        ]
        dl._write_manifest(tasks, tmp_path)
        assert (tmp_path / "h2mare_manifest.json").exists()

    def test_manifest_content_matches_tasks(self, dl, tmp_path):
        tasks = [
            DownloadTask(
                dataset_id="cmems-rep-sst",
                date_range=DateRange("2021-01-01", "2021-12-31"),
                dataset_type="rep",
            ),
            DownloadTask(
                dataset_id="cmems-nrt-sst",
                date_range=DateRange("2022-01-01", "2022-06-30"),
                dataset_type="nrt",
            ),
        ]
        dl._write_manifest(tasks, tmp_path)
        records = json.loads((tmp_path / "h2mare_manifest.json").read_text())
        assert len(records) == 2
        assert records[0]["dataset_type"] == "rep"
        assert records[1]["dataset_type"] == "nrt"
        assert records[0]["start"] == "2021-01-01"
        assert records[1]["end"] == "2022-06-30"


# ---------------------------------------------------------------------------
# CMEMSDownloader._execute_task
# ---------------------------------------------------------------------------


class TestExecuteTask:
    # Patch _retry_call to call fn once without retry delay so tests are fast
    # and isolated from retry mechanics (which are tested in test_base_downloader).
    _NO_RETRY = {"side_effect": lambda fn, *a, **kw: fn(*a)}

    def test_subset_true_calls_download_subset_per_chunk(self, dl):
        task = DownloadTask(
            dataset_id="cmems-rep-sst",
            date_range=DateRange("2020-01-01", "2020-03-31"),
            dataset_type="rep",
        )
        with (
            patch.object(dl, "download_subset") as mock_subset,
            patch.object(dl, "_retry_call", **self._NO_RETRY),
        ):
            dl._execute_task(task, FilePeriod.MONTH)

        assert mock_subset.call_count == 3  # Jan, Feb, Mar

    def test_subset_false_calls_download_original_once(self, dl_no_subset):
        task = DownloadTask(
            dataset_id="cmems-rep-sst",
            date_range=DateRange("2020-01-01", "2020-12-31"),
            dataset_type="rep",
        )
        with (
            patch.object(dl_no_subset, "download_original") as mock_original,
            patch.object(dl_no_subset, "_retry_call", **self._NO_RETRY),
        ):
            dl_no_subset._execute_task(task, FilePeriod.MONTH)

        mock_original.assert_called_once()

    def test_exception_from_download_propagates(self, dl):
        # Verifies the old silent-swallow bug is gone: errors must bubble up.
        task = DownloadTask(
            dataset_id="cmems-rep-sst",
            date_range=DateRange("2020-01-01", "2020-01-31"),
            dataset_type="rep",
        )
        with (
            patch.object(
                dl, "download_subset", side_effect=ConnectionError("API down")
            ),
            patch.object(dl, "_retry_call", **self._NO_RETRY),
        ):
            with pytest.raises(ConnectionError, match="API down"):
                dl._execute_task(task, FilePeriod.MONTH)

    def test_subset_true_passes_chunk_dates_to_download_subset(self, dl):
        task = DownloadTask(
            dataset_id="cmems-rep-sst",
            date_range=DateRange("2020-06-01", "2020-06-30"),
            dataset_type="rep",
        )
        with (
            patch.object(dl, "download_subset") as mock_subset,
            patch.object(dl, "_retry_call", **self._NO_RETRY),
        ):
            dl._execute_task(task, FilePeriod.MONTH)

        call_args = mock_subset.call_args
        assert call_args[0][0] == "cmems-rep-sst"
        assert pd.Timestamp(call_args[0][1]) == pd.Timestamp("2020-06-01")
        assert pd.Timestamp(call_args[0][2]) == pd.Timestamp("2020-06-30")


# ---------------------------------------------------------------------------
# Dataset availability — the trailing day
# ---------------------------------------------------------------------------


class TestAvailabilityTrailingDay:
    """
    Store coverage is day-granular, so a day ingested while the provider is
    still publishing it counts as covered and its remaining hours are never
    fetched again. The trailing day is held back until it is whole.
    """

    @staticmethod
    def _availability(dl, first: str, last: str) -> pd.Timestamp:
        with patch(
            "h2mare.downloader.cmems_downloader.get_dataset_time_range",
            return_value=(pd.Timestamp(first), pd.Timestamp(last)),
        ):
            return pd.Timestamp(dl.get_rep_availability().end)

    def test_hourly_partial_last_day_is_held_back(self, dl_hourly):
        end = self._availability(dl_hourly, "2020-01-01", "2020-06-30 05:00")

        assert end == pd.Timestamp("2020-06-29")

    def test_hourly_complete_last_day_is_kept(self, dl_hourly):
        end = self._availability(dl_hourly, "2020-01-01", "2020-06-30 23:00")

        assert end == pd.Timestamp("2020-06-30")

    def test_daily_keeps_the_day_its_last_stamp_falls_in(self, dl):
        # One step per day, so the day holding the last stamp is complete by
        # definition — noon stamps must not be mistaken for a partial day.
        end = self._availability(dl, "2020-01-01", "2020-06-30 12:00")

        assert end == pd.Timestamp("2020-06-30")

    def test_a_history_of_one_partial_day_does_not_go_empty(self, dl_hourly):
        """Holding the day back would put end before start, which DateRange
        rejects outright — a brand-new product must not raise."""
        end = self._availability(dl_hourly, "2020-06-30 00:00", "2020-06-30 05:00")

        assert end == pd.Timestamp("2020-06-30")


# ---------------------------------------------------------------------------
# download_subset — the end bound handed to the toolbox
# ---------------------------------------------------------------------------


class TestDownloadSubsetEndBound:
    """
    A chunk end is a date, and the toolbox reads it as an instant. On an hourly
    variable that must be widened to the end of the day, or every chunk arrives
    23 hours short — silently, since the loss sits at the tail of the span where
    the convert step only warns.
    """

    @staticmethod
    def _requested_end(dl, end="2020-06-30") -> pd.Timestamp:
        with patch(
            "h2mare.downloader.cmems_downloader.download_subset"
        ) as mock_download:
            dl.download_subset(
                "cmems-rep-sst", pd.Timestamp("2020-06-01"), pd.Timestamp(end)
            )
        return pd.Timestamp(mock_download.call_args.kwargs["end"])

    def test_hourly_requests_the_whole_final_day(self, dl_hourly):
        end = self._requested_end(dl_hourly)

        assert end >= pd.Timestamp("2020-06-30 23:00")
        assert end < pd.Timestamp("2020-07-01")

    def test_daily_keeps_the_midnight_bound(self, dl):
        # Deliberate: every existing store was written under this bound, and a
        # daily product's single step of the day already falls inside it.
        assert self._requested_end(dl) == pd.Timestamp("2020-06-30")

    def test_hourly_start_is_untouched(self, dl_hourly):
        # Widening only applies to the upper bound — midnight already names the
        # first step of the start day.
        with patch(
            "h2mare.downloader.cmems_downloader.download_subset"
        ) as mock_download:
            dl_hourly.download_subset(
                "cmems-rep-sst", pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-30")
            )

        assert pd.Timestamp(mock_download.call_args.kwargs["start"]) == pd.Timestamp(
            "2020-06-01"
        )

    def test_hourly_chunks_do_not_overlap_at_the_boundary(self, dl_hourly):
        """Consecutive chunks must meet, not overlap: the widened end of one
        stops one nanosecond before the next chunk's start."""
        jan_end = self._requested_end(dl_hourly, end="2020-01-31")

        assert jan_end < pd.Timestamp("2020-02-01")
