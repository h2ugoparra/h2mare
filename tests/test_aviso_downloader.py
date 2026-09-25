"""Tests for AVISODownloader.get_rep_availability and get_nrt_availability."""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import msgspec
import pandas as pd
import pytest

from h2mare.downloader.aviso_downloader import AVISODownloader
from h2mare.models import AppConfig
from h2mare.types import DateRange, FTPDownloadTask

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_ENTRY = {
    "local_folder": "AVISO_FSLE",
    "source_vars": ["fsle_max"],
    "dataset_id_rep": "/dataset/fsle/rep",
    "dataset_id_nrt": "/dataset/fsle/nrt",
    "source": "aviso",
    "archive_raw": True,
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


class TestNothingToDownloadMessage:
    def test_before_coverage_warns_instead_of_up_to_date(self, dl_no_nrt):
        from loguru import logger

        rep = DateRange("2010-01-01", "2023-12-31")
        messages: list[str] = []
        sink = logger.add(messages.append, level="INFO", format="{level}|{message}")
        try:
            with (
                patch.object(dl_no_nrt, "_create_download_tasks", return_value=[]),
                patch.object(dl_no_nrt, "get_rep_availability", return_value=rep),
            ):
                assert dl_no_nrt.run("1994-01-01", "1994-01-31") is False
        finally:
            logger.remove(sink)

        text = "".join(messages)
        assert "up to date" not in text
        assert "WARNING|'fsle': requested 1994-01-01 to 1994-01-31" in text


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

    def test_the_catalog_consulted_is_this_downloader_s(self, dl):
        """
        Its own config and store, not whatever the machine has deployed.

        Resolving through get_settings() read another store's catalog, and for
        a var_key the deployed config did not define the constructor raised
        into the debug branch — the check turning itself off, silently, on any
        machine pointed elsewhere. It is why these tests passed or failed
        depending on H2MARE_ROOT.
        """
        with patch("h2mare.storage.zarr_catalog.ZarrCatalog") as MockCatalog:
            MockCatalog.return_value.df = pd.DataFrame()
            dl._warn_if_rep_updated(pd.Timestamp("2023-12-31"))

        kwargs = MockCatalog.call_args.kwargs
        assert kwargs["app_config"] is dl.app_config
        assert kwargs["store_root"] == dl.store_root

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


# ---------------------------------------------------------------------------
# Download manifest
#
# Netcdf2Zarr._write_provenance returns early when no manifest is present, so
# without one every converted AVISO Zarr is missing `source_datasets` and the
# catalog scanner falls back to dataset_id_rep — labelling near-real-time data
# as delayed-time. Only CMEMSDownloader used to write one.
# ---------------------------------------------------------------------------

_EDDIES_ENTRY = {
    "local_folder": "AVISO_Eddy_Trajectory",
    "source_vars": ["amplitude"],
    "dataset_id_rep": "/value-added/eddy-trajectory/delayed-time",
    "dataset_id_nrt": "/value-added/eddy-trajectory/near-real-time",
    "source": "aviso",
    "archive_raw": True,
    "pattern": r"(\d{8})_(\d{8})",
    "trajectory_format": True,
}


def _eddies_app_config() -> AppConfig:
    return msgspec.convert(
        {
            "variables": {"eddies": _EDDIES_ENTRY},
            "secrets": {
                "aviso_ftp_server": "ftp.example.com",
                "aviso_username": "user",
                "aviso_password": "pass",
            },
        },
        AppConfig,
    )


def _run_download(downloader, tasks, failed=None):
    """Drive run() with the FTP layer stubbed, returning the manifest contents.

    The manifest must land in ``download_dir`` — the same directory
    ``Netcdf2Zarr._read_manifest`` looks in — not the download root.

    ``failed`` is the list ``download_parallel`` reports back as undownloadable.
    """
    with (
        patch.object(downloader, "_create_download_tasks", return_value=tasks),
        patch.object(downloader, "download_parallel", return_value=failed or []),
        patch.object(
            downloader,
            "get_rep_availability",
            return_value=DateRange(
                pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31")
            ),
        ),
        patch.object(downloader, "_warn_if_rep_updated"),
        patch(
            "h2mare.downloader.aviso_downloader.resolve_date_range",
            return_value=DateRange(
                pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-29")
            ),
        ),
    ):
        downloader.run()
    manifest_path = downloader.download_dir / "h2mare_manifest.json"
    return manifest_path, (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )


class TestWriteManifest:
    def test_manifest_is_written(self, dl):
        """Regression: AVISO wrote no manifest, so provenance was never stamped."""
        tasks = [
            FTPDownloadTask(filepath="fsle_20200105.nc", source="rep"),
            FTPDownloadTask(filepath="fsle_20200210.nc", source="nrt"),
        ]

        path, records = _run_download(dl, tasks)

        assert path.exists()
        assert records is not None

    def test_manifest_records_both_datasets(self, dl):
        tasks = [
            FTPDownloadTask(filepath="fsle_20200105.nc", source="rep"),
            FTPDownloadTask(filepath="fsle_20200120.nc", source="rep"),
            FTPDownloadTask(filepath="fsle_20200210.nc", source="nrt"),
        ]

        _, records = _run_download(dl, tasks)

        by_type = {r["dataset_type"]: r for r in records}
        assert by_type["rep"]["dataset_id"] == "/dataset/fsle/rep"
        assert by_type["nrt"]["dataset_id"] == "/dataset/fsle/nrt"

    def test_spans_come_from_downloaded_filenames(self, dl):
        """The rep span must cover its files only, not the whole request."""
        tasks = [
            FTPDownloadTask(filepath="fsle_20200105.nc", source="rep"),
            FTPDownloadTask(filepath="fsle_20200120.nc", source="rep"),
            FTPDownloadTask(filepath="fsle_20200210.nc", source="nrt"),
        ]

        _, records = _run_download(dl, tasks)

        by_type = {r["dataset_type"]: r for r in records}
        assert by_type["rep"]["start"] == "2020-01-05"
        assert by_type["rep"]["end"] == "2020-01-20"
        assert by_type["nrt"]["start"] == "2020-02-10"
        assert by_type["nrt"]["end"] == "2020-02-10"

    def test_nrt_only_run_records_no_rep_entry(self, dl):
        """An incremental run downloading only NRT must not claim rep coverage."""
        tasks = [FTPDownloadTask(filepath="fsle_20200210.nc", source="nrt")]

        _, records = _run_download(dl, tasks)

        assert [r["dataset_type"] for r in records] == ["nrt"]

    def test_manifest_matches_schema_netcdf2zarr_reads(self, dl):
        """_write_provenance indexes dataset_id/dataset_type/start/end."""
        tasks = [FTPDownloadTask(filepath="fsle_20200105.nc", source="rep")]

        _, records = _run_download(dl, tasks)

        assert set(records[0]) == {"dataset_id", "dataset_type", "start", "end"}

    def test_eddies_range_filenames_span_start_to_end(self, tmp_path):
        """Trajectory filenames carry two dates; the span must use both."""
        with patch.object(AVISODownloader, "connect_ftp", return_value=MagicMock()):
            dl_eddies = AVISODownloader(
                "eddies",
                app_config=_eddies_app_config(),
                store_root=tmp_path,
                download_root=tmp_path,
            )
        tasks = [
            FTPDownloadTask(filepath="Eddy_20200101_20200131.nc", source="rep"),
            FTPDownloadTask(filepath="Eddy_20200201_20200229.nc", source="nrt"),
        ]

        _, records = _run_download(dl_eddies, tasks)

        by_type = {r["dataset_type"]: r for r in records}
        assert by_type["rep"]["start"] == "2020-01-01"
        assert by_type["rep"]["end"] == "2020-01-31"
        assert by_type["nrt"]["end"] == "2020-02-29"

    def test_no_manifest_when_nrt_not_configured_and_no_nrt_tasks(self, dl_no_nrt):
        tasks = [FTPDownloadTask(filepath="fsle_20200105.nc", source="rep")]

        _, records = _run_download(dl_no_nrt, tasks)

        assert [r["dataset_type"] for r in records] == ["rep"]


# ---------------------------------------------------------------------------
# Partial-download detection
#
# download_parallel used to log a failed transfer and carry on, after which
# run() wrote the manifest for every planned task and reported success with the
# *requested* file count. The missing day then vanished: store coverage is a
# min/max watermark a one-day hole cannot move, and convert derives what to
# write from the files that did arrive. AVISO_FSLE 1999 and 2025-06-02 are what
# that looks like months later.
# ---------------------------------------------------------------------------


class TestPartialDownloadIsReported:
    def test_run_raises_when_a_file_fails(self, dl):
        tasks = [
            FTPDownloadTask(filepath="fsle_20200105.nc", source="rep"),
            FTPDownloadTask(filepath="fsle_20200106.nc", source="rep"),
        ]

        with pytest.raises(RuntimeError, match="Download incomplete"):
            _run_download(dl, tasks, failed=["fsle_20200106.nc"])

    def test_error_names_the_failed_file(self, dl):
        tasks = [FTPDownloadTask(filepath="fsle_20200105.nc", source="rep")]

        with pytest.raises(RuntimeError, match="fsle_20200105.nc"):
            _run_download(dl, tasks, failed=["fsle_20200105.nc"])

    def test_manifest_still_written_when_a_file_fails(self, dl):
        """The requested span must survive so convert can detect the gap."""
        tasks = [FTPDownloadTask(filepath="fsle_20200105.nc", source="rep")]

        with pytest.raises(RuntimeError):
            _run_download(dl, tasks, failed=["fsle_20200105.nc"])

        manifest = dl.download_dir / "h2mare_manifest.json"
        assert manifest.exists()
        assert json.loads(manifest.read_text())[0]["start"] == "2020-01-05"

    def test_run_succeeds_when_nothing_fails(self, dl):
        tasks = [FTPDownloadTask(filepath="fsle_20200105.nc", source="rep")]

        _run_download(dl, tasks, failed=[])  # must not raise


class TestDownloadParallelCollectsFailures:
    @contextmanager
    def _dl_with_failing_paths(self, dl, bad: set[str]):
        """Make connect_ftp succeed, but RETR raise for paths in *bad*.

        ``_retry_call`` is stubbed to a single immediate attempt — its
        exponential backoff sleeps 10-60s between tries, and the retry policy
        is not what these tests are about.
        """

        def _connect():
            ftp = MagicMock()
            ftp.size.side_effect = Exception("no size")

            def _retr(cmd, _callback):
                path = cmd.split(" ", 1)[1]
                if path in bad:
                    raise OSError(f"connection reset for {path}")

            ftp.retrbinary.side_effect = _retr
            return ftp

        with (
            patch.object(dl, "connect_ftp", side_effect=_connect),
            patch.object(dl, "_retry_call", side_effect=lambda fn, *a, **kw: fn(*a)),
        ):
            yield

    def test_returns_the_failed_paths(self, dl, tmp_path):
        with self._dl_with_failing_paths(dl, {"b.nc"}):
            failed = dl.download_parallel(
                ["a.nc", "b.nc"], dataset_id="/d", output_dir=tmp_path
            )

        assert failed == ["b.nc"]

    def test_returns_empty_when_all_succeed(self, dl, tmp_path):
        with self._dl_with_failing_paths(dl, set()):
            failed = dl.download_parallel(
                ["a.nc", "b.nc"], dataset_id="/d", output_dir=tmp_path
            )

        assert failed == []

    def test_successful_file_is_renamed_into_place(self, dl, tmp_path):
        with self._dl_with_failing_paths(dl, set()):
            dl.download_parallel(["a.nc"], dataset_id="/d", output_dir=tmp_path)

        assert (tmp_path / "a.nc").exists()
        assert not (tmp_path / "a.nc.part").exists()

    def test_failed_transfer_leaves_no_partial_nc_behind(self, dl, tmp_path):
        """A truncated *.nc would be globbed by convert as if it were complete."""
        with self._dl_with_failing_paths(dl, {"b.nc"}):
            dl.download_parallel(["b.nc"], dataset_id="/d", output_dir=tmp_path)

        assert not (tmp_path / "b.nc").exists()
        assert not (tmp_path / "b.nc.part").exists()


# ---------------------------------------------------------------------------
# FTP connection lifecycle
#
# The control connection opened in __init__ used to be left for the garbage
# collector ("# self.ftp.quit()" sat commented out), which Python reports as
# ResourceWarning: unclosed <socket.socket ...>. ResourceWarning is ignored by
# default, so it only showed up under PYTHONWARNINGS=always.
# ---------------------------------------------------------------------------


class TestFtpConnectionLifecycle:
    def test_close_quits_the_connection(self, dl):
        ftp = dl.ftp
        dl.close()
        ftp.quit.assert_called_once()

    def test_close_falls_back_to_close_when_quit_raises(self, dl):
        """A server that already dropped the link makes QUIT raise."""
        ftp = dl.ftp
        ftp.quit.side_effect = OSError("connection reset")
        dl.close()
        ftp.close.assert_called_once()

    def test_close_is_idempotent(self, dl):
        dl.close()
        dl.close()  # must not raise

    def test_context_manager_closes(self, tmp_path):
        with patch.object(AVISODownloader, "connect_ftp", return_value=MagicMock()):
            with AVISODownloader(
                "fsle",
                app_config=_make_app_config(_ENTRY),
                store_root=tmp_path,
                download_root=tmp_path,
            ) as d:
                ftp = d.ftp
        ftp.quit.assert_called_once()

    def test_reconnect_releases_the_dead_socket(self, dl, tmp_path):
        """Each reconnect used to strand the previous socket."""
        dead = dl.ftp
        dead.voidcmd.side_effect = OSError("connection lost")
        fresh = MagicMock()

        with patch.object(AVISODownloader, "connect_ftp", return_value=fresh):
            with patch("h2mare.downloader.aviso_downloader._download_to_part"):
                dl.download_file("/dataset/fsle/rep/a.nc", tmp_path)

        dead.close.assert_called_once()
        assert dl.ftp is fresh
