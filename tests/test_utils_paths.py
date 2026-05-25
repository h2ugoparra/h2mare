"""Tests for utils/paths.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from h2mare.models import AppConfig
from h2mare.utils.paths import resolve_download_path, resolve_store_path

_ENTRY = {
    "local_folder": "sst",
    "variables": ["analysed_sst"],
    "dataset_id_rep": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "source": "cmems",
    "pattern": r".*\.nc",
}
_CONFIG = msgspec.convert({"variables": {"sst": _ENTRY}, "secrets": {}}, AppConfig)
_VAR_CONFIG = _CONFIG.variables["sst"]


class TestResolveDownloadPath:
    def test_explicit_root_used(self, tmp_path):
        result = resolve_download_path(
            _VAR_CONFIG, download_root=tmp_path, warn_if_missing=False
        )
        assert result == tmp_path.resolve()

    def test_missing_path_still_returns_resolved_path(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        result = resolve_download_path(
            _VAR_CONFIG, download_root=missing, warn_if_missing=True
        )
        assert result == missing.resolve()

    def test_warn_if_missing_false_skips_check(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        result = resolve_download_path(
            _VAR_CONFIG, download_root=missing, warn_if_missing=False
        )
        assert result == missing.resolve()

    def test_falls_back_to_settings_downloads_dir(self, tmp_path):
        mock_settings = MagicMock()
        mock_settings.DOWNLOADS_DIR = tmp_path
        with patch("h2mare.utils.paths.get_settings", return_value=mock_settings):
            result = resolve_download_path(_VAR_CONFIG, warn_if_missing=False)
        assert result == (tmp_path / _VAR_CONFIG.local_folder).resolve()


class TestResolveStorePath:
    def test_explicit_root_used(self, tmp_path):
        result = resolve_store_path(
            _VAR_CONFIG, store_root=tmp_path, warn_if_missing=False
        )
        assert result == tmp_path.resolve()

    def test_store_dir_used_when_available(self, tmp_path):
        mock_settings = MagicMock()
        mock_settings.STORE_ROOT = tmp_path
        with patch("h2mare.utils.paths.get_settings", return_value=mock_settings):
            result = resolve_store_path(_VAR_CONFIG, warn_if_missing=False)
        assert result == (tmp_path / _VAR_CONFIG.local_folder).resolve()

    def test_falls_back_to_zarr_dir_when_store_root_none(self, tmp_path):
        mock_settings = MagicMock()
        mock_settings.STORE_ROOT = None
        mock_settings.ZARR_DIR = tmp_path
        with patch("h2mare.utils.paths.get_settings", return_value=mock_settings):
            result = resolve_store_path(_VAR_CONFIG, warn_if_missing=False)
        assert result == (tmp_path / _VAR_CONFIG.local_folder).resolve()
