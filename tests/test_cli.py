"""Tests for CLI commands — argument validation and error paths."""

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pandas as pd
from typer.testing import CliRunner

from h2mare.cli import _configure, _use_utf8_console
from h2mare.cli.audit import app as audit_app
from h2mare.cli.catalog import _print_catalog
from h2mare.cli.catalog import app as catalog_app
from h2mare.cli.compile import app as compile_app
from h2mare.cli.main import app as main_app
from h2mare.cli.nc2zarr import app as nc2zarr_app
from h2mare.cli.zarr2parquet import app as zarr2parquet_app
from h2mare.models import AppConfig
from h2mare.types import BBox

_runner = CliRunner()

_MINIMAL_APP_CONFIG = msgspec.convert(
    {
        "variables": {
            "sst": {
                "local_folder": "sst",
                "source_vars": ["analysed_sst"],
                "dataset_id_rep": "cmems-sst",
                "source": "cmems",
                "archive_raw": False,
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


# ---------------------------------------------------------------------------
# cli/__init__.py — console encoding
# ---------------------------------------------------------------------------


class TestConsoleEncoding:
    @staticmethod
    def _cp1252_stream() -> io.TextIOWrapper:
        """Stand-in for the cp1252 stdout Windows hands the CLI."""
        return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

    def test_catalog_summary_survives_cp1252_console(self, monkeypatch):
        cat = MagicMock()
        cat.df = pd.DataFrame()
        cat.summary.return_value = {
            "num_files": 29,
            "time_coverage": SimpleNamespace(
                start=pd.Timestamp("1998-01-01"), end=pd.Timestamp("2026-08-06")
            ),
            "variables": {"sst"},
            "total_timesteps": 10444,
            "store_root": "D:/data/CMEMS_SST",
            "catalog_path": "sst_zarr_catalog.parquet",
            "last_scanned": pd.Timestamp("2026-08-10"),
        }
        monkeypatch.setattr(
            "h2mare.storage.zarr_catalog.ZarrCatalog", lambda *a, **k: cat
        )
        monkeypatch.setattr(sys, "stdout", self._cp1252_stream())
        monkeypatch.setattr(sys, "stderr", self._cp1252_stream())

        _use_utf8_console()
        _print_catalog("sst", show_rows=False)
        sys.stdout.flush()

        printed = sys.stdout.buffer.getvalue().decode("utf-8")
        assert "1998-01-01 → 2026-08-06" in printed

    def test_utf8_console_leaves_utf8_streams_alone(self, monkeypatch):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", stream)
        _use_utf8_console()
        assert sys.stdout.errors == "strict"

    def test_every_command_configures_the_console(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            "h2mare.cli._use_utf8_console", lambda: calls.append("console")
        )
        monkeypatch.setattr("h2mare.cli.configure_logging", lambda *a, **k: None)
        _configure()
        assert calls == ["console"]


# ---------------------------------------------------------------------------
# cli/audit.py — audit command
# ---------------------------------------------------------------------------


class TestAuditCLI:
    def test_no_target_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(audit_app, [])
        assert result.exit_code == 1

    def test_unknown_var_key_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(audit_app, ["nonexistent"])
        assert result.exit_code == 1
        assert "Unknown" in result.output

    def test_clean_store_exits_zero(self, tmp_path):
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key") as mock_audit,
        ):
            mock_audit.return_value = SimpleNamespace(
                var_key="sst",
                n_files=3,
                gaps=[],
                slices=[],
                errors=[],
                ok=True,
                n_known_gaps=0,
                known_gaps=pd.DatetimeIndex([]),
                store_exists=True,
                store_expected=True,
            )
            result = _runner.invoke(audit_app, ["sst"])
        assert result.exit_code == 0
        assert "no gaps found" in result.output

    def test_findings_exit_non_zero(self, tmp_path):
        """The command gates a scheduled run, so a gap must fail the process."""
        gap = SimpleNamespace(
            path=Path("cmems_sst_2026.zarr"),
            span=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-06")),
            missing=pd.DatetimeIndex([pd.Timestamp("2026-07-31")]),
        )
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key") as mock_audit,
        ):
            mock_audit.return_value = SimpleNamespace(
                var_key="sst",
                n_files=1,
                gaps=[gap],
                slices=[],
                errors=[],
                ok=False,
                n_known_gaps=0,
                known_gaps=pd.DatetimeIndex([]),
                store_exists=True,
                store_expected=True,
            )
            result = _runner.invoke(audit_app, ["sst"])
        assert result.exit_code == 1
        assert "2026-07-31" in result.output


class TestAuditKnownGapsDisplay:
    """A suppression list nothing can print is one that grows unnoticed."""

    def _result(self, *, ok: bool, known: list[str]):
        gap = SimpleNamespace(
            path=Path("aviso_fsle_2025.zarr"),
            span=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
            missing=pd.DatetimeIndex([pd.Timestamp("2025-09-09")]),
        )
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in known])
        return SimpleNamespace(
            var_key="fsle",
            n_files=29,
            gaps=[] if ok else [gap],
            slices=[],
            errors=[],
            ok=ok,
            known_gaps=idx,
            n_known_gaps=len(idx),
            store_exists=True,
            store_expected=True,
        )

    def _run(self, tmp_path, result, args):
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key", return_value=result),
        ):
            return _runner.invoke(audit_app, args)

    def test_failing_var_still_reports_the_suppressed_count(self, tmp_path):
        """Regression: the count was built but never interpolated on [FAIL]."""
        out = self._run(
            tmp_path, self._result(ok=False, known=["2025-06-02"]), ["sst"]
        ).output

        assert "[FAIL]" in out
        assert "1 known source gap(s) excluded" in out

    def test_known_flag_lists_the_dates(self, tmp_path):
        out = self._run(
            tmp_path, self._result(ok=True, known=["2025-06-02"]), ["sst", "--known"]
        ).output

        assert "2025-06-02" in out

    def test_known_flag_shows_a_passing_var_without_show_ok(self, tmp_path):
        """Otherwise the one thing --known exists for is invisible when clean."""
        out = self._run(
            tmp_path, self._result(ok=True, known=["2025-06-02"]), ["sst", "--known"]
        ).output

        assert "fsle" in out

    def test_known_flag_is_quiet_for_a_var_without_gaps(self, tmp_path):
        out = self._run(
            tmp_path, self._result(ok=True, known=[]), ["sst", "--known"]
        ).output

        assert "known source gaps" not in out

    def test_dates_are_not_listed_without_the_flag(self, tmp_path):
        out = self._run(
            tmp_path, self._result(ok=True, known=["2025-06-02"]), ["sst", "--show-ok"]
        ).output

        assert "1 known source gap(s) excluded" in out
        assert "2025-06-02" not in out

    def test_an_interval_is_rendered_as_a_range(self, tmp_path):
        out = self._run(
            tmp_path,
            self._result(ok=True, known=["2025-06-02", "2025-06-03", "2025-06-04"]),
            ["sst", "--known"],
        ).output

        assert "2025-06-02 → 2025-06-04" in out


class TestAuditReporting:
    """Advice and labels have to match the check that produced the findings."""

    def _result(self, *, gaps=(), slices=(), store_exists=True, store_expected=True):
        return SimpleNamespace(
            var_key="chl",
            n_files=29,
            gaps=list(gaps),
            slices=list(slices),
            errors=[],
            ok=not (gaps or slices),
            known_gaps=pd.DatetimeIndex([]),
            n_known_gaps=0,
            store_exists=store_exists,
            store_expected=store_expected,
            store_root=Path("store/chl"),
        )

    def _gap(self):
        return SimpleNamespace(
            path=Path("a_2025.zarr"),
            span=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
            missing=pd.DatetimeIndex([pd.Timestamp("2025-06-02")]),
        )

    def _slice(self, variable="chl", date="1999-01-25"):
        return SimpleNamespace(
            path=Path("chl_1999.zarr"),
            variable=variable,
            date=pd.Timestamp(date),
            kind="empty",
            detail="no finite values",
        )

    def _run(self, tmp_path, result, args):
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key", return_value=result),
        ):
            return _runner.invoke(audit_app, args)

    def test_value_findings_do_not_advise_re_downloading(self, tmp_path):
        """Re-running cannot fill a day the provider never published."""
        out = self._run(
            tmp_path, self._result(slices=[self._slice()]), ["sst", "--values"]
        ).output

        assert "re-run the download" not in out.lower()
        assert "known_gaps" in out

    def test_axis_findings_do_advise_re_downloading(self, tmp_path):
        out = self._run(tmp_path, self._result(gaps=[self._gap()]), ["sst"]).output

        assert "re-run the" in out.lower()

    def test_both_kinds_get_their_own_advice(self, tmp_path):
        out = self._run(
            tmp_path,
            self._result(gaps=[self._gap()], slices=[self._slice()]),
            ["sst", "--values"],
        ).output

        assert "re-run the" in out.lower()
        assert "known_gaps" in out

    def test_header_names_the_value_check(self, tmp_path):
        out = self._run(
            tmp_path, self._result(), ["sst", "--values", "--show-ok"]
        ).output

        assert "axis + value check" in out

    def test_header_says_axis_only_by_default(self, tmp_path):
        out = self._run(tmp_path, self._result(), ["sst", "--show-ok"]).output

        assert "— axis check" in out

    def test_a_day_is_reported_once_across_variables(self, tmp_path):
        """chl reported 22 findings for 11 days, one per column."""
        out = self._run(
            tmp_path,
            self._result(slices=[self._slice("chl"), self._slice("chl_fdist")]),
            ["sst", "--values"],
        ).output

        assert out.count("1999-01-25") == 1
        assert "chl, chl_fdist" in out

    def test_a_missing_store_is_never_passed(self, tmp_path):
        out = self._run(
            tmp_path, self._result(store_exists=False), ["sst", "--show-ok"]
        ).output

        assert "[OK]" not in out

    def test_a_downloaded_variable_with_no_store_fails(self, tmp_path):
        """An unmounted drive must not read as a clean store."""
        result = self._run(
            tmp_path, self._result(store_exists=False), ["sst", "--show-ok"]
        )

        assert result.exit_code == 1
        assert "[FAIL]" in result.output
        assert "STORE_ROOT" in result.output
        assert "no gaps found" not in result.output.lower()

    def test_a_computed_variable_with_no_store_is_skipped(self, tmp_path):
        """`moon` never gets a store; reporting it forever teaches people to
        ignore the command."""
        result = self._run(
            tmp_path,
            self._result(store_exists=False, store_expected=False),
            ["sst", "--show-ok"],
        )

        assert "[SKIP]" in result.output
        assert "[FAIL]" not in result.output

    def test_nothing_checked_does_not_report_a_pass(self, tmp_path):
        """The only variable asked for was skipped, so there is no pass to
        report — and no finding either."""
        result = self._run(
            tmp_path,
            self._result(store_exists=False, store_expected=False),
            ["sst"],
        )

        assert result.exit_code == 1
        assert "Nothing was checked" in result.output
        assert "no gaps found" not in result.output.lower()

    def test_an_audit_that_raises_is_a_finding(self, tmp_path):
        """Not a skip: the check did not run and nobody knows why."""
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch(
                "h2mare.storage.audit.audit_var_key",
                side_effect=OSError("drive not ready"),
            ),
        ):
            result = _runner.invoke(audit_app, ["sst"])

        assert result.exit_code == 1
        assert "[ERROR]" in result.output
        assert "drive not ready" in result.output
        assert "no gaps found" not in result.output.lower()

    def test_since_without_values_is_flagged(self, tmp_path):
        result = self._run(tmp_path, self._result(), ["sst", "--since", "2025-01-01"])

        assert "only bounds --values" in result.output

    def test_unparseable_since_exits_1(self, tmp_path):
        result = self._run(
            tmp_path, self._result(), ["sst", "--values", "--since", "nonsense"]
        )

        assert result.exit_code == 1


class TestAuditScopeLine:
    """A verdict has to carry its own scope, or it claims more than it read.

    "No gaps found." reads as a statement about the store; what was checked may
    have been one variable, or one variable since 2020, or — with stores
    skipped — fewer variables than were asked for.
    """

    _TWO_VARS = msgspec.convert(
        {
            "variables": {
                "sst": {
                    "local_folder": "sst",
                    "source_vars": ["analysed_sst"],
                    "dataset_id_rep": "cmems-sst",
                    "source": "cmems",
                    "archive_raw": False,
                },
                "moon": {
                    "local_folder": "moon",
                    "source_vars": ["moon_phase"],
                    "dataset_id_rep": "ephem4.2",
                    "source": "python",
                    "archive_raw": False,
                },
            },
            "secrets": {},
        },
        AppConfig,
    )

    def _result(self, var_key, *, store_exists=True, store_expected=True):
        return SimpleNamespace(
            var_key=var_key,
            n_files=29,
            gaps=[],
            slices=[],
            errors=[],
            ok=True,
            known_gaps=pd.DatetimeIndex([]),
            n_known_gaps=0,
            store_exists=store_exists,
            store_expected=store_expected,
            store_root=Path(f"store/{var_key}"),
        )

    def _run_all(self, tmp_path, args):
        settings = _mock_settings(tmp_path)
        settings.app_config = self._TWO_VARS

        def _audit(key, **kwargs):
            if key == "moon":
                return self._result(key, store_exists=False, store_expected=False)
            return self._result(key)

        with (
            patch("h2mare.cli.audit.get_settings", return_value=settings),
            patch("h2mare.storage.audit.audit_var_key", side_effect=_audit),
        ):
            return _runner.invoke(audit_app, args)

    def test_the_verdict_states_how_many_were_checked(self, tmp_path):
        result = self._run_all(tmp_path, ["--all"])

        assert result.exit_code == 0
        assert "Checked 1 of 2 key variable(s)" in result.output

    def test_a_benign_skip_is_stated_once(self, tmp_path):
        """The [SKIP] line already carries the variable and the reason; the
        verdict's shortfall points at it rather than repeating it."""
        out = self._run_all(tmp_path, ["--all"]).output

        assert out.count("moon") == 1
        assert "[SKIP] moon" in out

    def test_counts_are_labelled_key_variables(self, tmp_path):
        """A var_key such as `eddies` is ~15 columns; a bare count reads as
        those."""
        out = self._run_all(tmp_path, ["--all"]).output

        assert "key variable(s)" in out
        assert "of 2 variable(s)" not in out

    def test_the_verdict_names_a_single_variable(self, tmp_path):
        out = self._run_all(tmp_path, ["sst"]).output

        assert "Checked key variable sst" in out

    def test_the_verdict_carries_the_value_window(self, tmp_path):
        """Otherwise a pass conceals that everything before --since was never
        opened."""
        out = self._run_all(
            tmp_path, ["sst", "--values", "--since", "2020-01-01"]
        ).output

        assert "axis + value check since 2020-01-01" in out

    def test_the_verdict_says_axis_check_by_default(self, tmp_path):
        out = self._run_all(tmp_path, ["sst"]).output

        assert "Checked key variable sst, axis check" in out

    def test_the_opening_and_closing_lines_agree(self, tmp_path):
        """Two wordings for the same check read as two different runs."""
        out = self._run_all(tmp_path, ["--all", "--values"]).output

        assert out.count("axis + value check") == 2

    def test_findings_carry_the_same_scope(self, tmp_path):
        settings = _mock_settings(tmp_path)
        settings.app_config = self._TWO_VARS
        failing = self._result("sst")
        failing.gaps = [
            SimpleNamespace(
                path=Path("sst_2026.zarr"),
                span=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")),
                missing=pd.DatetimeIndex([pd.Timestamp("2026-07-31")]),
            )
        ]
        with (
            patch("h2mare.cli.audit.get_settings", return_value=settings),
            patch("h2mare.storage.audit.audit_var_key", return_value=failing),
        ):
            result = _runner.invoke(audit_app, ["sst"])

        assert result.exit_code == 1
        assert "Checked key variable sst, axis check — 1 finding(s)." in result.output


class TestCatalogMissingStore:
    """The warning belongs in the output, attributable to the variable.

    resolve_store_path warns "will be created when data is added", which is
    false for a read-only inspector and doubly so for `moon` — computed at
    compile time, it never gets a store at all.
    """

    def test_catalog_does_not_warn_about_a_missing_store(self, tmp_path):
        from h2mare.storage.zarr_catalog import ZarrCatalog

        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch.object(ZarrCatalog, "__init__", return_value=None) as mock_init,
            patch.object(
                ZarrCatalog,
                "df",
                new_callable=lambda: property(lambda s: pd.DataFrame()),
            ),
            patch.object(ZarrCatalog, "summary", return_value={"num_files": 0}),
        ):
            _runner.invoke(catalog_app, ["sst"])

        assert mock_init.call_args.kwargs.get("warn_if_missing") is False

    def test_missing_store_is_annotated_in_the_output(self, tmp_path):
        from h2mare.storage.zarr_catalog import ZarrCatalog

        gone = tmp_path / "definitely_not_here"
        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch.object(ZarrCatalog, "__init__", return_value=None),
            patch.object(
                ZarrCatalog,
                "df",
                new_callable=lambda: property(lambda s: pd.DataFrame()),
            ),
            patch.object(
                ZarrCatalog,
                "summary",
                return_value={"num_files": 0, "store_root": str(gone)},
            ),
        ):
            result = _runner.invoke(catalog_app, ["sst"])

        assert "(does not exist)" in result.output

    def test_an_existing_store_is_not_annotated(self, tmp_path):
        from h2mare.storage.zarr_catalog import ZarrCatalog

        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch.object(ZarrCatalog, "__init__", return_value=None),
            patch.object(
                ZarrCatalog,
                "df",
                new_callable=lambda: property(lambda s: pd.DataFrame()),
            ),
            patch.object(
                ZarrCatalog,
                "summary",
                return_value={"num_files": 0, "store_root": str(tmp_path)},
            ),
        ):
            result = _runner.invoke(catalog_app, ["sst"])

        assert "(does not exist)" not in result.output


class TestCatalogBbox:
    """The inspector reports the extent on disk, and flags a config mismatch."""

    def _output(self, tmp_path, summary):
        from h2mare.storage.zarr_catalog import ZarrCatalog

        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch.object(ZarrCatalog, "__init__", return_value=None),
            patch.object(
                ZarrCatalog,
                "df",
                new_callable=lambda: property(lambda s: pd.DataFrame()),
            ),
            patch.object(ZarrCatalog, "summary", return_value=summary),
        ):
            return _runner.invoke(catalog_app, ["sst"]).output

    def test_store_bbox_is_printed(self, tmp_path):
        out = self._output(
            tmp_path,
            {"num_files": 1, "store_bbox": BBox(-10.0, 30.0, 0.0, 40.0)},
        )
        assert "-10, 30 → 0, 40" in out
        assert "10W-0E-30N-40N" in out

    def test_config_bbox_shown_only_when_it_differs(self, tmp_path):
        same = {
            "num_files": 1,
            "store_bbox": BBox(-10.0, 30.0, 0.0, 40.0),
            "bbox": BBox(-10.0, 30.0, 0.0, 40.0),
        }
        assert "BBox (cfg)" not in self._output(tmp_path, same)

        # Cell centres sit half a grid cell inside the requested edges — every
        # variable would otherwise print a mismatch line that means nothing.
        cell_centres = {
            "num_files": 1,
            "store_bbox": BBox(-9.975, 30.025, -0.025, 39.975),
            "bbox": BBox(-10.0, 30.0, 0.0, 40.0),
        }
        assert "BBox (cfg)" not in self._output(tmp_path, cell_centres)

        wider = {
            "num_files": 1,
            "store_bbox": BBox(-20.0, 20.0, 10.0, 50.0),
            "bbox": BBox(-10.0, 30.0, 0.0, 40.0),
        }
        out = self._output(tmp_path, wider)
        assert "BBox (cfg) : -10, 30 → 0, 40" in out

    def test_falls_back_to_config_bbox_for_an_empty_store(self, tmp_path):
        out = self._output(
            tmp_path,
            {"num_files": 0, "store_bbox": None, "bbox": BBox(-10.0, 30.0, 0.0, 40.0)},
        )
        assert "BBox (cfg) : -10, 30 → 0, 40" in out

    def test_unknown_bbox_renders_as_a_dash(self, tmp_path):
        # summary() writes the string "No data" into `bbox` when unset — it
        # must not reach the formatter.
        out = self._output(tmp_path, {"num_files": 0, "bbox": "No data"})
        assert "BBox       : —" in out


class TestAuditParquetFindingLocation:
    """A finding has to say which file it is about.

    Every partition in the store is a `part-N.parquet` and PARQUET_DIR holds
    more than one store root, so the bare filename identified 1493 candidates.
    """

    def _run(self, tmp_path, nulls):
        settings = _mock_settings(tmp_path)
        with (
            patch("h2mare.cli.audit.get_settings", return_value=settings),
            patch("h2mare.storage.audit.audit_parquet_nulls", return_value=nulls),
        ):
            return _runner.invoke(audit_app, ["--parquet"]).output

    def test_the_finding_names_the_store_and_partition(self, tmp_path):
        path = (
            tmp_path
            / "parquet"
            / "h2mare_compiled-data"
            / "year=2025"
            / "month=6"
            / "part-0.parquet"
        )
        out = self._run(tmp_path, [(path, "mnkc_hmlmeso")])

        assert "h2mare_compiled-data" in out
        assert "year=2025" in out
        assert "month=6" in out

    def test_the_audited_root_is_named_up_front(self, tmp_path):
        out = self._run(tmp_path, [])
        assert str(tmp_path / "parquet") in out

    def test_a_path_outside_the_root_still_prints(self, tmp_path):
        # relative_to raises for anything not under PARQUET_DIR; the finding
        # must survive that rather than take the command down.
        stray = tmp_path / "elsewhere" / "part-0.parquet"
        out = self._run(tmp_path, [(stray, "sst")])
        assert str(stray) in out


class TestAuditFindingCount:
    """The count has to match the lines printed, or it reads as a discrepancy."""

    def _run(self, tmp_path, slices, args):
        result = SimpleNamespace(
            var_key="chl",
            n_files=29,
            gaps=[],
            slices=list(slices),
            errors=[],
            ok=False,
            known_gaps=pd.DatetimeIndex([]),
            n_known_gaps=0,
            store_exists=True,
        )
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key", return_value=result),
        ):
            return _runner.invoke(audit_app, args)

    def _slice(self, variable, date="1999-01-25"):
        return SimpleNamespace(
            path=Path("chl_1999.zarr"),
            variable=variable,
            date=pd.Timestamp(date),
            kind="empty",
            detail="no finite values",
        )

    def test_one_day_across_two_columns_counts_once(self, tmp_path):
        out = self._run(
            tmp_path,
            [self._slice("chl"), self._slice("chl_fdist")],
            ["sst", "--values"],
        ).output

        assert "1 finding(s)" in out

    def test_distinct_days_count_separately(self, tmp_path):
        out = self._run(
            tmp_path,
            [
                self._slice("chl", "1999-01-25"),
                self._slice("chl_fdist", "1999-01-25"),
                self._slice("chl", "1999-11-17"),
            ],
            ["sst", "--values"],
        ).output

        assert "2 finding(s)" in out

    def test_the_count_matches_the_lines_printed(self, tmp_path):
        out = self._run(
            tmp_path,
            [self._slice("chl"), self._slice("chl_fdist")],
            ["sst", "--values"],
        ).output

        printed = sum(1 for line in out.splitlines() if "[empty]" in line)
        assert f"{printed} finding(s)" in out


# ---------------------------------------------------------------------------
# cli/zarr2parquet.py — parquet command
# ---------------------------------------------------------------------------


def _parquet_settings(tmp_path: Path) -> MagicMock:
    """_mock_settings plus the compiled_vars the --add-var path reads."""
    m = _mock_settings(tmp_path)
    cfg = msgspec.convert(
        {
            "variables": {
                "h2ds": {
                    "local_folder": "h2ds",
                    "source_vars": ["sst"],
                    "dataset_id_rep": "compiled",
                    "source": "cmems",
                    "archive_raw": False,
                    "pattern": r".*\.nc",
                },
                "sst": {
                    "local_folder": "sst",
                    "source_vars": ["analysed_sst"],
                    "dataset_id_rep": "cmems-sst",
                    "source": "cmems",
                    "archive_raw": False,
                    "pattern": r".*\.nc",
                    "compiled_vars": ["sst", "sst_std"],
                },
                "o2": {
                    "local_folder": "o2",
                    "source_vars": ["o2"],
                    "dataset_id_rep": "cmems-o2",
                    "source": "cmems",
                    "archive_raw": False,
                    "pattern": r".*\.nc",
                },
            },
            "secrets": {},
        },
        AppConfig,
    )
    m.app_config = cfg
    return m


class TestParquetCLIArgumentValidation:
    """
    The argument handling that decides which partitions get rewritten. Explicit
    dates re-read every variable and rewrite the affected partitions wholesale,
    so getting the window wrong is destructive — and this was the least
    exercised command in the CLI.
    """

    def _invoke(self, tmp_path, args):
        with patch(
            "h2mare.cli.zarr2parquet.get_settings",
            return_value=_parquet_settings(tmp_path),
        ):
            return _runner.invoke(zarr2parquet_app, args)

    def test_only_start_date_exits_with_code_1(self, tmp_path):
        result = self._invoke(tmp_path, ["--start-date", "2021-01-01"])
        assert result.exit_code == 1

    def test_only_end_date_exits_with_code_1(self, tmp_path):
        result = self._invoke(tmp_path, ["--end-date", "2021-12-31"])
        assert result.exit_code == 1

    def test_start_after_end_exits_with_code_1(self, tmp_path):
        result = self._invoke(
            tmp_path, ["--start-date", "2021-12-31", "--end-date", "2021-01-01"]
        )
        assert result.exit_code == 1

    def test_add_var_and_vars_together_exit_with_code_1(self, tmp_path):
        result = self._invoke(tmp_path, ["--add-var", "sst", "-v", "h2ds"])
        assert result.exit_code == 1
        assert "cannot be used together" in result.output

    def test_unknown_var_key_exits_with_code_1(self, tmp_path):
        result = self._invoke(tmp_path, ["-v", "nonexistent"])
        assert result.exit_code == 1
        assert "unknown variable key" in result.output

    def test_unknown_add_var_key_exits_with_code_1(self, tmp_path):
        result = self._invoke(tmp_path, ["--add-var", "nonexistent"])
        assert result.exit_code == 1
        assert "unknown variable key" in result.output

    def test_add_var_without_compiled_vars_exits_with_code_1(self, tmp_path):
        """o2 exists but declares no compiled_vars, so there is nothing to merge."""
        result = self._invoke(tmp_path, ["--add-var", "o2"])
        assert result.exit_code == 1
        assert "compiled_vars" in result.output


class TestParquetCLIDispatch:
    """What the validated arguments actually hand to Zarr2Parquet."""

    def _invoke(self, tmp_path, args, converter):
        with (
            patch(
                "h2mare.cli.zarr2parquet.get_settings",
                return_value=_parquet_settings(tmp_path),
            ),
            patch(
                "h2mare.format_converters.zarr2parquet.Zarr2Parquet",
                return_value=converter,
            ) as ctor,
        ):
            result = _runner.invoke(zarr2parquet_app, args)
        return result, ctor

    def test_defaults_to_h2ds_when_no_vars_given(self, tmp_path):
        conv = MagicMock()
        result, ctor = self._invoke(tmp_path, [], conv)
        assert result.exit_code == 0
        assert ctor.call_args.kwargs["var_key"] == "h2ds"

    def test_each_requested_var_key_is_converted(self, tmp_path):
        conv = MagicMock()
        result, ctor = self._invoke(tmp_path, ["-v", "sst", "-v", "o2"], conv)
        assert result.exit_code == 0
        assert [c.kwargs["var_key"] for c in ctor.call_args_list] == ["sst", "o2"]

    def test_date_window_is_passed_through(self, tmp_path):
        conv = MagicMock()
        self._invoke(
            tmp_path,
            ["-v", "sst", "--start-date", "2021-03-01", "--end-date", "2021-03-31"],
            conv,
        )
        kwargs = conv.run.call_args.kwargs
        assert kwargs["start_date"] == "2021-03-01"
        assert kwargs["end_date"] == "2021-03-31"

    def test_depth_is_passed_through(self, tmp_path):
        conv = MagicMock()
        self._invoke(tmp_path, ["-v", "o2", "--depth", "100"], conv)
        assert conv.run.call_args.kwargs["depth"] == 100.0

    def test_out_dir_overrides_the_parquet_root(self, tmp_path):
        conv = MagicMock()
        out = tmp_path / "elsewhere"
        _, ctor = self._invoke(tmp_path, ["--out-dir", str(out)], conv)
        assert ctor.call_args.kwargs["parquet_root"] == out

    def test_add_var_expands_to_compiled_vars_on_h2ds(self, tmp_path):
        conv = MagicMock()
        result, ctor = self._invoke(tmp_path, ["--add-var", "sst"], conv)
        assert result.exit_code == 0
        assert ctor.call_args.kwargs["var_key"] == "h2ds"
        assert conv.run.call_args.kwargs["variables"] == ["sst", "sst_std"]

    def test_backup_is_only_synced_when_requested(self, tmp_path):
        conv = MagicMock()
        self._invoke(tmp_path, ["-v", "sst"], conv)
        conv.sync_data.assert_not_called()

        conv2 = MagicMock()
        self._invoke(tmp_path, ["-v", "sst", "--parquet-backup"], conv2)
        conv2.sync_data.assert_called_once()

    def test_store_path_overrides_the_store_root(self, tmp_path):
        conv = MagicMock()
        settings = _parquet_settings(tmp_path)
        custom = tmp_path / "other_store"
        with (
            patch("h2mare.cli.zarr2parquet.get_settings", return_value=settings),
            patch(
                "h2mare.format_converters.zarr2parquet.Zarr2Parquet",
                return_value=conv,
            ),
        ):
            _runner.invoke(zarr2parquet_app, ["--store-path", str(custom)])
        settings.override_store_root.assert_called_once_with(custom)


class TestParquetCLIReportsFailure:
    """
    A failed conversion has to reach the exit code. Both loops used to log the
    error and return 0, so a script driving `h2mare parquet` saw success after
    every variable had failed. `h2mare run` has always done the opposite —
    PipelineManager.run() returns False and cli/main.py exits 1.
    """

    def _invoke(self, tmp_path, args, converter):
        with (
            patch(
                "h2mare.cli.zarr2parquet.get_settings",
                return_value=_parquet_settings(tmp_path),
            ),
            patch(
                "h2mare.format_converters.zarr2parquet.Zarr2Parquet",
                return_value=converter,
            ),
        ):
            return _runner.invoke(zarr2parquet_app, args)

    def test_single_failing_var_key_exits_1(self, tmp_path):
        conv = MagicMock()
        conv.run.side_effect = ValueError("no zarr data in range")
        result = self._invoke(tmp_path, ["-v", "sst"], conv)
        assert result.exit_code == 1

    def test_every_var_key_failing_exits_1(self, tmp_path):
        conv = MagicMock()
        conv.run.side_effect = ValueError("no zarr data in range")
        result = self._invoke(tmp_path, ["-v", "sst", "-v", "o2"], conv)
        assert result.exit_code == 1

    def test_one_failure_among_several_still_exits_1(self, tmp_path):
        """A partial failure is still a failure — the good ones are not a pass."""
        conv = MagicMock()
        conv.run.side_effect = [None, ValueError("no zarr data in range")]
        result = self._invoke(tmp_path, ["-v", "sst", "-v", "o2"], conv)
        assert result.exit_code == 1

    def test_the_other_var_keys_are_still_attempted(self, tmp_path):
        """Failing early must not abort the rest — only change the exit code."""
        conv = MagicMock()
        conv.run.side_effect = [ValueError("boom"), None]
        result = self._invoke(tmp_path, ["-v", "sst", "-v", "o2"], conv)
        assert conv.run.call_count == 2, "the second var_key was skipped"
        assert result.exit_code == 1

    def test_all_succeeding_still_exits_0(self, tmp_path):
        conv = MagicMock()
        result = self._invoke(tmp_path, ["-v", "sst", "-v", "o2"], conv)
        assert result.exit_code == 0

    def test_add_var_failure_exits_1(self, tmp_path):
        conv = MagicMock()
        conv.run.side_effect = ValueError("nothing to merge")
        result = self._invoke(tmp_path, ["--add-var", "sst"], conv)
        assert result.exit_code == 1

    def test_add_var_success_still_exits_0(self, tmp_path):
        conv = MagicMock()
        result = self._invoke(tmp_path, ["--add-var", "sst"], conv)
        assert result.exit_code == 0
