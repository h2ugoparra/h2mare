"""Tests for AVISODownloader.get_rep_availability and get_nrt_availability."""

from unittest.mock import MagicMock, patch

import msgspec
import pandas as pd
import pytest

from h2mare.downloader.aviso_downloader import AVISODownloader
from h2mare.models import AppConfig
from h2mare.types import DateRange

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_ENTRY = {
    "local_folder": "AVISO_FSLE",
    "source_vars": ["fsle_max"],
    "dataset_id_rep": "/dataset/fsle/rep",
    "dataset_id_nrt": "/dataset/fsle/nrt",
    "source": "aviso",
    "pattern": r"(\d{8})",
    "subset": False,
}

_ENTRY_NO_NRT = {**_ENTRY, "dataset_id_nrt": None}


def _make_app_config(entry=_ENTRY) -> AppConfig:
    return msgspec.convert(
        {
            "variables": {"fsle": entry},
            "secrets": {
                "aviso_ftp_server": "ftp.example.com",
                "aviso_username": "user",
                "aviso_password": "pass",
            },
        },
        AppConfig,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dl(tmp_path):
    """AVISODownloader with rep+nrt configured and FTP mocked out."""
    with patch.object(AVISODownloader, "connect_ftp", return_value=MagicMock()):
        return AVISODownloader(
            "fsle",
            app_config=_make_app_config(_ENTRY),
            store_root=tmp_path,
            download_root=tmp_path,
        )


@pytest.fixture
def dl_no_nrt(tmp_path):
    """AVISODownloader with no NRT dataset configured."""
    with patch.object(AVISODownloader, "connect_ftp", return_value=MagicMock()):
        return AVISODownloader(
            "fsle",
            app_config=_make_app_config(_ENTRY_NO_NRT),
            store_root=tmp_path,
            download_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetRepAvailability:
    def test_calls_get_dataset_files_with_rep_id(self, dl):
        fake_files = ["rep/file1.nc", "rep/file2.nc"]
        expected = DateRange(pd.Timestamp("1993-01-01"), pd.Timestamp("2023-12-31"))

        with (
            patch.object(
                dl, "_get_dataset_files", return_value=fake_files
            ) as mock_files,
            patch.object(dl, "_get_dataset_availability", return_value=expected),
        ):
            dl.get_rep_availability()

        mock_files.assert_called_once_with(dl.var_config.dataset_id_rep)

    def test_calls_get_dataset_availability_with_file_list(self, dl):
        fake_files = ["rep/file1.nc"]
        expected = DateRange(pd.Timestamp("1993-01-01"), pd.Timestamp("2023-12-31"))

        with (
            patch.object(dl, "_get_dataset_files", return_value=fake_files),
            patch.object(
                dl, "_get_dataset_availability", return_value=expected
            ) as mock_avail,
        ):
            dl.get_rep_availability()

        mock_avail.assert_called_once_with(fake_files)

    def test_returns_date_range(self, dl):
        expected = DateRange(pd.Timestamp("1993-01-01"), pd.Timestamp("2023-12-31"))

        with (
            patch.object(dl, "_get_dataset_files", return_value=[]),
            patch.object(dl, "_get_dataset_availability", return_value=expected),
        ):
            result = dl.get_rep_availability()

        assert result == expected


class TestGetNrtAvailability:
    def test_calls_get_dataset_files_with_nrt_id(self, dl):
        fake_files = ["nrt/file1.nc"]
        expected = DateRange(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30"))

        with (
            patch.object(
                dl, "_get_dataset_files", return_value=fake_files
            ) as mock_files,
            patch.object(dl, "_get_dataset_availability", return_value=expected),
        ):
            dl.get_nrt_availability()

        mock_files.assert_called_once_with(dl.var_config.dataset_id_nrt)

    def test_returns_date_range(self, dl):
        expected = DateRange(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30"))

        with (
            patch.object(dl, "_get_dataset_files", return_value=[]),
            patch.object(dl, "_get_dataset_availability", return_value=expected),
        ):
            result = dl.get_nrt_availability()

        assert result == expected

    def test_returns_none_when_dataset_id_nrt_is_not_configured(self, dl_no_nrt):
        result = dl_no_nrt.get_nrt_availability()
        assert result is None

    def test_does_not_call_ftp_when_nrt_not_configured(self, dl_no_nrt):
        with patch.object(dl_no_nrt, "_get_dataset_files") as mock_files:
            dl_no_nrt.get_nrt_availability()

        mock_files.assert_not_called()


class TestGetRepAvailabilityCaching:
    def test_ftp_called_only_once_on_repeated_calls(self, dl):
        expected = DateRange(pd.Timestamp("1993-01-01"), pd.Timestamp("2023-12-31"))

        with (
            patch.object(dl, "_get_dataset_files", return_value=[]) as mock_files,
            patch.object(dl, "_get_dataset_availability", return_value=expected),
        ):
            dl.get_rep_availability()
            dl.get_rep_availability()

        mock_files.assert_called_once()

    def test_nrt_ftp_called_only_once_on_repeated_calls(self, dl):
        expected = DateRange(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30"))

        with (
            patch.object(dl, "_get_dataset_files", return_value=[]) as mock_files,
            patch.object(dl, "_get_dataset_availability", return_value=expected),
        ):
            dl.get_nrt_availability()
            dl.get_nrt_availability()

        mock_files.assert_called_once()


class TestWarnIfRepUpdated:
    def test_warning_emitted_when_api_end_date_is_newer(self, dl, tmp_path):
        import pandas as pd

        # Catalog shows rep data ending 2022-12-31
        catalog_df = pd.DataFrame(
            [
                {
                    "path": str(tmp_path / "dummy.zarr"),
                    "filename": "dummy.zarr",
                    "dataset": _ENTRY["dataset_id_rep"],
                    "start_date": pd.Timestamp("2020-01-01"),
                    "end_date": pd.Timestamp("2022-12-31"),
                }
            ]
        )

        from h2mare.storage.zarr_catalog import ZarrCatalog

        with patch.object(
            ZarrCatalog, "df", new_callable=lambda: property(lambda self: catalog_df)
        ):
            with patch("h2mare.downloader.base.logger") as mock_logger:
                # API reports rep ending 2023-12-31 — one year newer
                dl._warn_if_rep_updated(pd.Timestamp("2023-12-31"))

        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        assert "2022-12-31" in msg
        assert "2023-12-31" in msg

    def test_no_warning_when_api_end_date_matches_catalog(self, dl, tmp_path):
        catalog_df = pd.DataFrame(
            [
                {
                    "path": str(tmp_path / "dummy.zarr"),
                    "filename": "dummy.zarr",
                    "dataset": _ENTRY["dataset_id_rep"],
                    "start_date": pd.Timestamp("2020-01-01"),
                    "end_date": pd.Timestamp("2023-12-31"),
                }
            ]
        )

        from h2mare.storage.zarr_catalog import ZarrCatalog

        with patch.object(
            ZarrCatalog, "df", new_callable=lambda: property(lambda self: catalog_df)
        ):
            with patch("h2mare.downloader.base.logger") as mock_logger:
                dl._warn_if_rep_updated(pd.Timestamp("2023-12-31"))

        mock_logger.warning.assert_not_called()

    def test_no_warning_when_catalog_is_empty(self, dl):
        import pandas as pd

        from h2mare.storage.zarr_catalog import ZarrCatalog

        with patch.object(
            ZarrCatalog,
            "df",
            new_callable=lambda: property(lambda self: pd.DataFrame()),
        ):
            with patch("h2mare.downloader.base.logger") as mock_logger:
                dl._warn_if_rep_updated(pd.Timestamp("2023-12-31"))

        mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# adjust_ftp_path_to_dataset
# ---------------------------------------------------------------------------


class TestAdjustFtpPath:
    def test_stores_current_dataset_id(self, dl):
        dl.adjust_ftp_path_to_dataset("/dataset/fsle/rep")
        assert dl._current_dataset_id == "/dataset/fsle/rep"

    def test_updates_dataset_id_on_second_call(self, dl):
        dl.adjust_ftp_path_to_dataset("/dataset/fsle/rep")
        dl.adjust_ftp_path_to_dataset("/dataset/fsle/nrt")
        assert dl._current_dataset_id == "/dataset/fsle/nrt"

    def test_navigates_ftp_to_dataset_directory(self, dl):
        dl.adjust_ftp_path_to_dataset("/dataset/fsle/rep")
        dl.ftp.cwd.assert_called_with("/dataset/fsle/rep")


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    def _make_new_ftp(self):
        """FTP mock that succeeds: TYPE I works, size raises (→ file_size=None), retrbinary no-ops."""
        ftp = MagicMock()
        ftp.voidcmd.return_value = None
        ftp.size.side_effect = Exception("size unavailable")
        ftp.retrbinary.return_value = None
        return ftp

    def test_creates_output_file(self, dl, tmp_path):
        dl.ftp.voidcmd.return_value = None  # NOOP succeeds
        dl.ftp.size.side_effect = Exception("no size")

        dl.download_file("/dataset/fsle/rep/file.nc", output_dir=tmp_path)

        assert (tmp_path / "file.nc").exists()

    def test_reconnects_when_noop_raises(self, dl, tmp_path):
        dl.ftp.voidcmd.side_effect = Exception("connection lost")

        new_ftp = self._make_new_ftp()
        with patch.object(dl, "connect_ftp", return_value=new_ftp):
            dl.download_file("/dataset/fsle/rep/file.nc", output_dir=tmp_path)

        assert dl.ftp is new_ftp

    def test_navigates_to_current_dataset_after_reconnect(self, dl, tmp_path):
        dl._current_dataset_id = "/dataset/fsle/rep"
        dl.ftp.voidcmd.side_effect = Exception("connection lost")

        new_ftp = self._make_new_ftp()
        with patch.object(dl, "connect_ftp", return_value=new_ftp):
            dl.download_file("/dataset/fsle/rep/file.nc", output_dir=tmp_path)

        # After reconnect, adjust_ftp_path_to_dataset must navigate to the dataset dir.
        new_ftp.cwd.assert_called_with("/dataset/fsle/rep")
