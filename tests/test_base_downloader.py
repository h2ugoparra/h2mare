"""Tests for downloader/base.py — BaseDownloader helper methods."""

from unittest.mock import MagicMock, patch

import msgspec
import pandas as pd
import pytest

from h2mare.downloader.base import BaseDownloader
from h2mare.models import AppConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTRY = {
    "local_folder": "sst",
    "source_vars": ["analysed_sst"],
    "dataset_id_rep": "cmems-rep-sst",
    "source": "cmems",
    "pattern": r".*\.nc",
    "subset": True,
    "bbox": (-80, 0, 10, 70),
}


def _make_config() -> AppConfig:
    return msgspec.convert({"variables": {"sst": _ENTRY}, "secrets": {}}, AppConfig)


class ConcreteDownloader(BaseDownloader):
    """Minimal concrete subclass for testing abstract base."""

    def run(self, *args, **kwargs) -> bool:
        return True


@pytest.fixture
def dl(tmp_path):
    return ConcreteDownloader(
        "sst",
        app_config=_make_config(),
        store_root=tmp_path / "store",
        download_root=tmp_path / "downloads",
    )


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_var_key(self, dl):
        assert "sst" in repr(dl)

    def test_repr_contains_class_name(self, dl):
        assert "ConcreteDownloader" in repr(dl)


# ---------------------------------------------------------------------------
# _cleanup_empty_download_dir
# ---------------------------------------------------------------------------


class TestCleanupEmptyDownloadDir:
    def test_removes_empty_download_dir(self, dl):
        dl.download_dir.mkdir(parents=True, exist_ok=True)
        dl._cleanup_empty_download_dir()
        assert not dl.download_dir.exists()

    def test_keeps_non_empty_download_dir(self, dl):
        dl.download_dir.mkdir(parents=True, exist_ok=True)
        (dl.download_dir / "data.nc").touch()
        dl._cleanup_empty_download_dir()
        assert dl.download_dir.exists()

    def test_no_error_when_dir_does_not_exist(self, dl, tmp_path):
        dl.download_dir = tmp_path / "nonexistent" / "sst"
        dl._cleanup_empty_download_dir()  # must not raise


# ---------------------------------------------------------------------------
# _warn_if_rep_updated
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _retry_call
# ---------------------------------------------------------------------------


class TestRetryCall:
    def test_returns_result_on_immediate_success(self, dl):
        result = dl._retry_call(lambda: 99, max_attempts=3, wait_min=0, wait_max=0)
        assert result == 99

    def test_retries_on_failure_and_returns_on_eventual_success(self, dl):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "done"

        result = dl._retry_call(flaky, max_attempts=3, wait_min=0, wait_max=0)
        assert result == "done"
        assert call_count == 3

    def test_reraises_after_exhausting_all_attempts(self, dl):
        def always_fails():
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            dl._retry_call(always_fails, max_attempts=3, wait_min=0, wait_max=0)

    def test_attempt_count_matches_max_attempts(self, dl):
        calls = []

        def always_fails():
            calls.append(1)
            raise ValueError("fail")

        with pytest.raises(ValueError):
            dl._retry_call(always_fails, max_attempts=2, wait_min=0, wait_max=0)

        assert len(calls) == 2

    def test_logs_warning_before_each_retry(self, dl):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retry me")
            return True

        with patch("h2mare.downloader.base.logger") as mock_logger:
            dl._retry_call(flaky, max_attempts=3, wait_min=0, wait_max=0)

        # 2 failures before success → before_sleep called twice → 2 warnings
        assert mock_logger.warning.call_count == 2

    def test_warning_includes_attempt_number_and_exception(self, dl):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("connection timed out")
            return True

        with patch("h2mare.downloader.base.logger") as mock_logger:
            dl._retry_call(flaky, max_attempts=3, wait_min=0, wait_max=0)

        msg = mock_logger.warning.call_args[0][0]
        assert "attempt 1" in msg
        assert "TimeoutError" in msg
        assert "connection timed out" in msg

    def test_no_warning_logged_on_first_attempt_success(self, dl):
        with patch("h2mare.downloader.base.logger") as mock_logger:
            dl._retry_call(lambda: "ok", max_attempts=3, wait_min=0, wait_max=0)

        mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# _warn_if_rep_updated
# ---------------------------------------------------------------------------


class TestWarnIfRepUpdated:
    # ZarrCatalog is imported locally inside _warn_if_rep_updated, so patch
    # the class at its source module.
    _PATCH_TARGET = "h2mare.storage.zarr_catalog.ZarrCatalog"

    def test_no_error_when_catalog_is_empty(self, dl):
        mock_catalog = MagicMock()
        mock_catalog.df = pd.DataFrame()
        with patch(self._PATCH_TARGET, return_value=mock_catalog):
            dl._warn_if_rep_updated(pd.Timestamp("2024-01-01"))  # must not raise

    def test_no_error_when_zarr_catalog_raises(self, dl):
        with patch(self._PATCH_TARGET, side_effect=Exception("missing")):
            dl._warn_if_rep_updated(pd.Timestamp("2024-01-01"))  # must not raise

    def test_no_error_when_no_rep_rows_in_catalog(self, dl):
        mock_catalog = MagicMock()
        mock_catalog.df = pd.DataFrame(
            {"dataset": ["other-ds"], "end_date": ["2023-01-01"]}
        )
        with patch(self._PATCH_TARGET, return_value=mock_catalog):
            dl._warn_if_rep_updated(pd.Timestamp("2024-01-01"))  # must not raise
