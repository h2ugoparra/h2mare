"""Tests for CMEMSDownloader pattern generation and task creation logic."""
import json
from pathlib import Path
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
from h2mare.types import DateRange, DownloadTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTRY = {
    "local_folder": "sst",
    "variables": ["analysed_sst"],
    "dataset_id_rep": "cmems-rep-sst",
    "dataset_id_nrt": "cmems-nrt-sst",
    "source": "cmems",
    "pattern": r".*\.nc",
    "subset": True,
    "bbox": (-80, 0, 10, 70),
}

_ENTRY_NO_NRT = {**_ENTRY, "dataset_id_nrt": None}


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
        assert generate_copernicus_patterns("2023-01-01", "2023-01-31") == ["*2023/01/*"]

    def test_full_year_returns_year_shortcut(self):
        assert generate_copernicus_patterns("2023-01-01", "2023-12-31") == ["*2023/*"]

    def test_partial_range_within_month(self):
        assert generate_copernicus_patterns("2023-01-21", "2023-01-23") == ["*2023012[1-3]*"]

    def test_multi_year_range_produces_per_month_patterns(self):
        result = generate_copernicus_patterns("2022-11-01", "2023-02-28")
        assert "*2022/11/*" in result
        assert "*2022/12/*" in result
        assert "*2023/01/*" in result
        assert "*2023/02/*" in result

    def test_full_decade_in_single_bracket(self):
        # 2023-01-20 to 2023-01-29 → tens=2, ones 0–9
        assert generate_copernicus_patterns("2023-01-20", "2023-01-29") == ["*2023012[0-9]*"]

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
            tasks = dl_no_nrt._create_download_tasks(DateRange("2020-01-01", "2020-12-31"))

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
