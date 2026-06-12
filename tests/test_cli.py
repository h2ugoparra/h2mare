"""Tests for CLI commands — argument validation and error paths."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
from typer.testing import CliRunner

from h2mare.cli.catalog import app as catalog_app
from h2mare.cli.compile import app as compile_app
from h2mare.cli.main import app as main_app
from h2mare.cli.nc2zarr import app as nc2zarr_app
from h2mare.models import AppConfig

_runner = CliRunner()

_MINIMAL_APP_CONFIG = msgspec.convert(
    {
        "variables": {
            "sst": {
                "local_folder": "sst",
                "source_vars": ["analysed_sst"],
                "dataset_id_rep": "cmems-sst",
                "source": "cmems",
                "pattern": r".*\.nc",
            }
        },
        "secrets": {},
    },
    AppConfig,
)


def _mock_settings(tmp_path: Path) -> MagicMock:
    m = MagicMock()
    m.LOGS_DIR = tmp_path
    m.STORE_ROOT = tmp_path / "store"
    m.PARQUET_DIR = tmp_path / "parquet"
    m.app_config = _MINIMAL_APP_CONFIG
    return m


# ---------------------------------------------------------------------------
# cli/main.py — run command
# ---------------------------------------------------------------------------


class TestRunCLI:
    def test_only_start_date_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(main_app, ["--start-date", "2021-01-01"])
        assert result.exit_code == 1

    def test_only_end_date_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(main_app, ["--end-date", "2021-12-31"])
        assert result.exit_code == 1

    def test_start_not_before_end_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(
                main_app,
                ["--start-date", "2021-12-31", "--end-date", "2021-01-01"],
            )
        assert result.exit_code == 1

    def test_single_day_range_is_accepted(self, tmp_path):
        # Regression: start == end used to be rejected, making a one-day
        # download impossible even though DateRange allows it.
        with (
            patch(
                "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.main.PipelineManager") as mock_pm,
        ):
            mock_pm.return_value.run.return_value = True
            result = _runner.invoke(
                main_app,
                ["-v", "sst", "--start-date", "2021-06-01", "--end-date", "2021-06-01"],
            )
        assert result.exit_code == 0

    def test_unknown_var_key_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(main_app, ["-v", "nonexistent"])
        assert result.exit_code == 1

    def test_missing_store_root_exits_with_code_1(self, tmp_path):
        ms = _mock_settings(tmp_path)
        ms.STORE_ROOT = None
        with patch("h2mare.cli.main.get_settings", return_value=ms):
            result = _runner.invoke(main_app, ["-v", "sst"])
        assert result.exit_code == 1

    def test_successful_run_exits_with_code_0(self, tmp_path):
        with (
            patch(
                "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.main.PipelineManager") as mock_pm,
        ):
            mock_pm.return_value.run.return_value = True
            result = _runner.invoke(main_app, ["-v", "sst"])
        assert result.exit_code == 0

    def test_failed_pipeline_exits_with_code_1(self, tmp_path):
        with (
            patch(
                "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.main.PipelineManager") as mock_pm,
        ):
            mock_pm.return_value.run.return_value = False
            result = _runner.invoke(main_app, ["-v", "sst"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# cli/compile.py — compile command
# ---------------------------------------------------------------------------


class TestCompileCLI:
    def test_only_start_date_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(compile_app, ["--start-date", "2021-01-01"])
        assert result.exit_code == 1

    def test_only_end_date_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(compile_app, ["--end-date", "2021-12-31"])
        assert result.exit_code == 1

    def test_start_not_before_end_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(
                compile_app,
                ["--start-date", "2021-12-31", "--end-date", "2021-01-01"],
            )
        assert result.exit_code == 1

    def test_unknown_var_key_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(compile_app, ["-v", "nonexistent"])
        assert result.exit_code == 1

    def test_single_day_range_is_accepted(self, tmp_path):
        # Regression: start == end used to be rejected (see TestRunCLI).
        with (
            patch(
                "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.processing.compiler.Compiler") as mock_compiler,
        ):
            result = _runner.invoke(
                compile_app,
                ["-v", "sst", "--start-date", "2021-06-01", "--end-date", "2021-06-01"],
            )
        assert result.exit_code == 0
        mock_compiler.return_value.run.assert_called_once()

    def test_valid_call_invokes_compiler(self, tmp_path):
        with (
            patch(
                "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.processing.compiler.Compiler") as mock_compiler,
        ):
            result = _runner.invoke(
                compile_app,
                ["-v", "sst", "--start-date", "2021-01-01", "--end-date", "2021-12-31"],
            )
        assert result.exit_code == 0
        mock_compiler.return_value.run.assert_called_once()


# ---------------------------------------------------------------------------
# cli/nc2zarr.py — convert command
# ---------------------------------------------------------------------------


class TestConvertCLI:
    def test_unknown_var_key_logs_error_and_exits_0(self, tmp_path):
        with patch(
            "h2mare.cli.nc2zarr.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(nc2zarr_app, ["-v", "nonexistent"])
        assert result.exit_code == 0

    def test_valid_var_key_invokes_netcdf2zarr(self, tmp_path):
        with (
            patch(
                "h2mare.cli.nc2zarr.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.nc2zarr.Netcdf2Zarr") as mock_n2z,
        ):
            result = _runner.invoke(nc2zarr_app, ["-v", "sst"])
        assert result.exit_code == 0
        mock_n2z.return_value.run.assert_called_once()


# ---------------------------------------------------------------------------
# cli/catalog.py — catalog command
# ---------------------------------------------------------------------------


class TestCatalogCLI:
    def test_no_var_key_and_no_all_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(catalog_app, [])
        assert result.exit_code == 1

    def test_unknown_var_key_prints_error_and_continues(self, tmp_path):
        with patch(
            "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(catalog_app, ["nonexistent"])
        assert (
            "Unknown" in result.output
            or "unknown" in result.output
            or result.exit_code == 0
        )

    def test_valid_var_key_calls_print_catalog(self, tmp_path):
        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.catalog._print_catalog") as mock_print,
        ):
            result = _runner.invoke(catalog_app, ["sst"])
        assert result.exit_code == 0
        mock_print.assert_called_once_with("sst", False)

    def test_all_flag_calls_print_catalog_for_each_var(self, tmp_path):
        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.catalog._print_catalog") as mock_print,
        ):
            result = _runner.invoke(catalog_app, ["--all"])
        assert result.exit_code == 0
        mock_print.assert_called_once_with("sst", False)
