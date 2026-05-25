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
    "variables": ["analysed_sst"],
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
