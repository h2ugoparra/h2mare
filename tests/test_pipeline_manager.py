"""Tests for PipelineManager run() isolation and dry-run behaviour."""

from unittest.mock import MagicMock, patch

import msgspec
import pytest

from h2mare.models import AppConfig
from h2mare.pipeline_manager import PipelineManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTRY = {
    "local_folder": "sst",
    "variables": ["analysed_sst"],
    "dataset_id_rep": "cmems_mod_glo_phy_my",
    "source": "cmems",
    "pattern": r".*\.nc",
}


def _make_config(*var_keys: str, source: str = "cmems") -> AppConfig:
    variables = {k: {**_ENTRY, "local_folder": k} for k in var_keys}
    return msgspec.convert({"variables": variables, "secrets": {}}, AppConfig)


def _make_manager(cfg: AppConfig, tmp_path, **kwargs) -> PipelineManager:
    downloader_cls = MagicMock()
    downloader_cls.return_value = MagicMock()
    registry = {"cmems": downloader_cls}
    return PipelineManager(cfg, registry, tmp_path, **kwargs), registry["cmems"]


# ---------------------------------------------------------------------------
# Module-level isolation: prevent any test from touching real Compiler or
# Zarr2Parquet (both do disk I/O against the live data store).
# Tests that need to inspect these mocks patch them again inside the test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_heavy_steps():
    with (
        patch("h2mare.processing.compiler.Compiler"),
        patch("h2mare.format_converters.zarr2parquet.Zarr2Parquet"),
    ):
        yield


# ---------------------------------------------------------------------------
# Per-variable error isolation
# ---------------------------------------------------------------------------


class TestDownloadFailureIsolation:
    def test_download_error_skips_to_next_variable(self, tmp_path):
        """A RuntimeError during download must not stop processing of other variables."""
        cfg = _make_config("sst", "chl")
        manager, downloader_cls = _make_manager(cfg, tmp_path)

        # sst download fails; chl succeeds
        downloader_cls.return_value.run.side_effect = [
            RuntimeError("network error"),
            True,
        ]

        with patch("h2mare.pipeline_manager.Netcdf2Zarr") as MockConverter:
            MockConverter.return_value.run.return_value = None
            manager.run()

        # Converter called only for chl (sst was skipped after download failure)
        assert MockConverter.return_value.run.call_count == 1

    def test_converter_error_does_not_raise(self, tmp_path):
        """A RuntimeError in Netcdf2Zarr.run() must be logged but not propagate."""
        cfg = _make_config("sst")
        manager, _ = _make_manager(cfg, tmp_path)

        with patch("h2mare.pipeline_manager.Netcdf2Zarr") as MockConverter:
            MockConverter.return_value.run.side_effect = RuntimeError("zarr error")
            manager.run()  # should not raise

    def test_converter_error_continues_next_variable(self, tmp_path):
        """Converter failure on one variable must not prevent the next from running."""
        cfg = _make_config("sst", "chl")
        manager, _ = _make_manager(cfg, tmp_path)

        with patch("h2mare.pipeline_manager.Netcdf2Zarr") as MockConverter:
            MockConverter.return_value.run.side_effect = [
                RuntimeError("zarr error"),
                None,
            ]
            manager.run()

        assert MockConverter.return_value.run.call_count == 2


# ---------------------------------------------------------------------------
# dry_run flag
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_skips_converter(self, tmp_path):
        """With dry_run=True, Netcdf2Zarr.run() must never be called."""
        cfg = _make_config("sst", "chl")
        manager, _ = _make_manager(cfg, tmp_path, dry_run=True)

        with patch("h2mare.pipeline_manager.Netcdf2Zarr") as MockConverter:
            manager.run()

        MockConverter.assert_not_called()

    def test_no_process_skips_converter(self, tmp_path):
        """With no_process=True, Netcdf2Zarr.run() must never be called."""
        cfg = _make_config("sst")
        manager, _ = _make_manager(cfg, tmp_path, no_convert=True)

        with patch("h2mare.pipeline_manager.Netcdf2Zarr") as MockConverter:
            manager.run()

        MockConverter.assert_not_called()

    def test_dry_run_still_calls_downloader(self, tmp_path):
        """dry_run=True is forwarded to downloader.run(), not skipped entirely."""
        cfg = _make_config("sst")
        manager, downloader_cls = _make_manager(cfg, tmp_path, dry_run=True)

        with patch("h2mare.pipeline_manager.Netcdf2Zarr"):
            manager.run()

        downloader_cls.return_value.run.assert_called_once_with(
            dry_run=True, start_date=None, end_date=None
        )

    def test_dates_forwarded_to_run(self, tmp_path):
        """start_date/end_date set on PipelineManager must be forwarded to downloader.run()."""
        cfg = _make_config("sst")
        manager, downloader_cls = _make_manager(
            cfg,
            tmp_path,
            start_date="2020-01-01",
            end_date="2020-12-31",
        )

        with patch("h2mare.pipeline_manager.Netcdf2Zarr"):
            manager.run()

        downloader_cls.return_value.run.assert_called_once_with(
            dry_run=False,
            start_date="2020-01-01",
            end_date="2020-12-31",
        )


# ---------------------------------------------------------------------------
# Variable filtering
# ---------------------------------------------------------------------------


class TestVariableFiltering:
    def test_h2ds_bathy_moon_always_skipped(self, tmp_path):
        """h2ds, bathy, and moon are pipeline-internal and must never be downloaded."""
        cfg = _make_config("sst", "h2ds", "bathy", "moon")
        manager, downloader_cls = _make_manager(cfg, tmp_path)

        with patch("h2mare.pipeline_manager.Netcdf2Zarr"):
            manager.run()

        # Only 'sst' should have a downloader instantiated
        assert downloader_cls.call_count == 1

    def test_unknown_source_skips_variable(self, tmp_path):
        """A var_key whose source has no registered downloader is skipped gracefully."""
        cfg = _make_config("sst")
        # Registry missing 'cmems' → no downloader found
        manager = PipelineManager(cfg, {}, tmp_path)

        with patch("h2mare.pipeline_manager.Netcdf2Zarr") as MockConverter:
            manager.run()  # should not raise

        MockConverter.assert_not_called()


# ---------------------------------------------------------------------------
# Post-run cleanup
# ---------------------------------------------------------------------------


class TestParquetStep:
    """The Parquet conversion step runs after compile and respects all skip flags."""

    def _run_with_mocks(self, tmp_path, **manager_kwargs):
        """Run the pipeline with Netcdf2Zarr, Compiler, and Zarr2Parquet all mocked."""
        cfg = _make_config("sst", "h2ds")
        manager, _ = _make_manager(cfg, tmp_path, **manager_kwargs)
        with (
            patch("h2mare.pipeline_manager.Netcdf2Zarr"),
            patch("h2mare.format_converters.zarr2parquet.Zarr2Parquet") as MockZ2P,
        ):
            manager.run()
        return MockZ2P

    def test_parquet_step_runs_by_default(self, tmp_path):
        """Zarr2Parquet is instantiated when no skip flags are set."""
        MockZ2P = self._run_with_mocks(tmp_path)
        assert MockZ2P.called

    def test_no_parquet_skips_parquet(self, tmp_path):
        MockZ2P = self._run_with_mocks(tmp_path, no_parquet=True)
        assert not MockZ2P.called

    def test_no_compile_skips_parquet(self, tmp_path):
        MockZ2P = self._run_with_mocks(tmp_path, no_compile=True)
        assert not MockZ2P.called

    def test_no_convert_skips_parquet(self, tmp_path):
        MockZ2P = self._run_with_mocks(tmp_path, no_convert=True)
        assert not MockZ2P.called

    def test_dry_run_skips_parquet(self, tmp_path):
        MockZ2P = self._run_with_mocks(tmp_path, dry_run=True)
        assert not MockZ2P.called

    def test_parquet_error_does_not_raise(self, tmp_path):
        """An error in the Parquet step is caught and logged, not propagated."""
        cfg = _make_config("sst")
        manager, _ = _make_manager(cfg, tmp_path)
        with (
            patch("h2mare.pipeline_manager.Netcdf2Zarr"),
            patch("h2mare.format_converters.zarr2parquet.Zarr2Parquet") as MockZ2P,
        ):
            MockZ2P.side_effect = ValueError("no zarr data")
            manager.run()  # must not raise


class TestCleanup:
    def test_empty_download_dir_removed(self, tmp_path):
        """An empty per-variable download subdirectory is removed after the pipeline run."""
        cfg = _make_config("sst")
        manager, _ = _make_manager(cfg, tmp_path)

        # Simulate the empty folder the downloader would have created
        empty_dir = tmp_path / "downloads" / "sst"
        empty_dir.mkdir(parents=True)

        with (
            patch("h2mare.pipeline_manager.get_settings") as mock_get_settings,
            patch("h2mare.pipeline_manager.Netcdf2Zarr"),
        ):
            mock_get_settings.return_value.DOWNLOADS_DIR = tmp_path / "downloads"
            manager.run()

        assert not empty_dir.exists()

    def test_non_empty_download_dir_kept(self, tmp_path):
        """A download subdirectory that still has files must not be removed."""
        cfg = _make_config("sst")
        manager, _ = _make_manager(cfg, tmp_path)

        non_empty_dir = tmp_path / "downloads" / "sst"
        non_empty_dir.mkdir(parents=True)
        (non_empty_dir / "data.nc").write_text("data")

        with (
            patch("h2mare.pipeline_manager.get_settings") as mock_get_settings,
            patch("h2mare.pipeline_manager.Netcdf2Zarr"),
        ):
            mock_get_settings.return_value.DOWNLOADS_DIR = tmp_path / "downloads"
            manager.run()

        assert non_empty_dir.exists()

    def test_cleanup_empty_download_dir_removes_on_dry_run(self, tmp_path):
        """_cleanup_empty_download_dir removes an empty folder even during dry-run."""
        import msgspec

        from h2mare.downloader.base import BaseDownloader
        from h2mare.models import AppConfig

        cfg = msgspec.convert(
            {
                "variables": {
                    "sst": {
                        "local_folder": "sst",
                        "variables": ["analysed_sst"],
                        "dataset_id_rep": "cmems_mod",
                        "source": "cmems",
                        "pattern": r".*\.nc",
                    }
                },
                "secrets": {},
            },
            AppConfig,
        )

        empty_dir = tmp_path / "sst"
        empty_dir.mkdir()

        # Instantiate a concrete subclass just to test the base method
        class _DummyDownloader(BaseDownloader):
            def run(self, *a, **kw): ...

        with patch("h2mare.downloader.base.get_settings") as mock_get_settings:
            mock_get_settings.return_value.app_config = cfg
            mock_get_settings.return_value.DOWNLOADS_DIR = tmp_path
            mock_get_settings.return_value.STORE_ROOT = None
            mock_get_settings.return_value.ZARR_DIR = tmp_path / "zarr"
            d = _DummyDownloader("sst", app_config=cfg, download_root=tmp_path)

        d._cleanup_empty_download_dir()
        assert not empty_dir.exists()
