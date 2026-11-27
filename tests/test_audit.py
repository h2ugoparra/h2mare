"""Tests for storage/audit.py — interior gap and slice-health detection.

The load-bearing property is not "finds gaps" but "finds gaps and nothing
else". A check that also fires on chl's legitimate all-null days would be
switched off within a season, at which point it protects nothing — so the
false-positive cases below matter as much as the true-positive ones.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.storage.audit import (
    audit_zarr_file,
    check_slice_health,
    contiguous_blocks,
    format_date_blocks,
    interior_gaps,
)
from h2mare.types import DateRange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ds(dates, null_days=(), constant_days=(), seed: int = 0) -> xr.Dataset:
    times = pd.DatetimeIndex(dates)
    nulls, flats = pd.DatetimeIndex(null_days), pd.DatetimeIndex(constant_days)
    rng = np.random.default_rng(seed)
    data = rng.uniform(10.0, 30.0, size=(len(times), 3, 3))
    for i, t in enumerate(times):
        if t in nulls:
            data[i, :, :] = np.nan
        elif t in flats:
            data[i, :, :] = 7.5
    return xr.Dataset(
        {"testvar": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


def _write(ds: xr.Dataset, path):
    ds.to_zarr(path)
    return path


_JAN = pd.date_range("2020-01-01", "2020-01-10", freq="D")


# ---------------------------------------------------------------------------
# interior_gaps
# ---------------------------------------------------------------------------


class TestInteriorGaps:
    def test_finds_a_missing_middle_day(self):
        missing = interior_gaps(_JAN.drop(pd.Timestamp("2020-01-05")))
        assert list(missing) == [pd.Timestamp("2020-01-05")]

    def test_complete_range_has_no_gaps(self):
        assert len(interior_gaps(_JAN)) == 0

    def test_ignores_a_short_tail(self):
        """A store stopping short of today is provider lag, not a defect."""
        assert len(interior_gaps(pd.date_range("2020-01-01", periods=5))) == 0

    def test_finds_a_multi_day_block(self):
        dropped = _JAN.drop(pd.date_range("2020-01-04", "2020-01-07"))
        assert len(interior_gaps(dropped)) == 4

    def test_hourly_freq_finds_a_missing_hour(self):
        """An hourly store's gaps are hours; the daily grid cannot see them."""
        hours = pd.date_range("2020-01-01", periods=48, freq="h")
        dropped = hours.drop(pd.Timestamp("2020-01-01 07:00"))

        assert list(interior_gaps(dropped, freq="h")) == [
            pd.Timestamp("2020-01-01 07:00")
        ]
        # The same axis on a daily grid: the day is still present, so nothing
        # is reported — which is exactly why the cadence has to be passed in.
        assert len(interior_gaps(dropped, freq="D")) == 0

    def test_hourly_freq_does_not_invent_gaps_in_a_complete_axis(self):
        hours = pd.date_range("2020-01-01", periods=48, freq="h")
        assert len(interior_gaps(hours, freq="h")) == 0

    def test_daily_axis_on_the_daily_grid_is_unchanged(self):
        """Default stays daily, so existing callers keep their behaviour."""
        assert list(interior_gaps(_JAN.drop(pd.Timestamp("2020-01-05")))) == [
            pd.Timestamp("2020-01-05")
        ]

    def test_single_date_is_never_a_gap(self):
        assert len(interior_gaps(pd.DatetimeIndex(["2020-01-01"]))) == 0

    def test_empty_index_is_never_a_gap(self):
        assert len(interior_gaps(pd.DatetimeIndex([]))) == 0

    def test_duplicate_dates_do_not_create_gaps(self):
        doubled = _JAN.append(_JAN).sort_values()
        assert len(interior_gaps(doubled)) == 0

    def test_unsorted_input_is_handled(self):
        assert len(interior_gaps(_JAN[::-1])) == 0


# ---------------------------------------------------------------------------
# audit_zarr_file
# ---------------------------------------------------------------------------


class TestAuditZarrFile:
    def test_reports_the_missing_day(self, tmp_path):
        path = _write(_ds(_JAN.drop(pd.Timestamp("2020-01-05"))), tmp_path / "a.zarr")

        gap, slices, error = audit_zarr_file(path)

        assert error is None
        assert gap is not None
        assert list(gap.missing) == [pd.Timestamp("2020-01-05")]

    def test_records_the_span_it_checked(self, tmp_path):
        path = _write(_ds(_JAN.drop(pd.Timestamp("2020-01-05"))), tmp_path / "a.zarr")

        gap, _, _ = audit_zarr_file(path)

        assert gap.span == (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"))

    def test_complete_store_reports_nothing(self, tmp_path):
        path = _write(_ds(_JAN), tmp_path / "a.zarr")

        gap, slices, error = audit_zarr_file(path)

        assert (gap, slices, error) == (None, [], None)

    def test_all_null_day_is_not_an_axis_gap(self, tmp_path):
        """chl's 1999 null days: on the axis, so structurally invisible here."""
        path = _write(
            _ds(_JAN, null_days=[pd.Timestamp("2020-01-05")]), tmp_path / "a.zarr"
        )

        gap, slices, error = audit_zarr_file(path)

        assert gap is None
        assert slices == []

    def test_all_null_day_is_found_when_values_are_checked(self, tmp_path):
        path = _write(
            _ds(_JAN, null_days=[pd.Timestamp("2020-01-05")]), tmp_path / "a.zarr"
        )

        _, slices, _ = audit_zarr_file(path, check_values=True)

        assert [s.kind for s in slices] == ["empty"]
        assert slices[0].date == pd.Timestamp("2020-01-05")

    def test_unreadable_store_is_reported_not_raised(self, tmp_path):
        broken = tmp_path / "broken.zarr"
        broken.mkdir()

        gap, slices, error = audit_zarr_file(broken)

        assert error is not None
        assert gap is None

    def test_timeless_store_is_skipped(self, tmp_path):
        ds = xr.Dataset(
            {"bathy": (["lat", "lon"], np.ones((3, 3)))},
            coords={"lat": [30.0, 35.0, 40.0], "lon": [-10.0, -5.0, 0.0]},
        )
        path = _write(ds, tmp_path / "static.zarr")

        assert audit_zarr_file(path) == (None, [], None)


# ---------------------------------------------------------------------------
# check_slice_health
#
# Replaces have_vars_unique_values, which inspected isel(time=-1) only — the
# one position that cannot reveal an interior hole — and conflated "all
# missing" with "constant", since NaN collapses to a single unique value.
# ---------------------------------------------------------------------------


class TestCheckSliceHealth:
    def test_empty_slice_is_reported_as_empty(self):
        issues = check_slice_health(_ds(_JAN, null_days=[pd.Timestamp("2020-01-05")]))
        assert [(k, d) for _, d, k, _ in issues] == [
            ("empty", pd.Timestamp("2020-01-05"))
        ]

    def test_constant_slice_is_reported_as_degenerate(self):
        issues = check_slice_health(
            _ds(_JAN, constant_days=[pd.Timestamp("2020-01-05")])
        )
        assert [k for _, _, k, _ in issues] == ["degenerate"]

    def test_empty_and_degenerate_are_not_conflated(self):
        """The old np.unique check could not tell these apart."""
        issues = check_slice_health(
            _ds(
                _JAN,
                null_days=[pd.Timestamp("2020-01-03")],
                constant_days=[pd.Timestamp("2020-01-07")],
            )
        )
        kinds = {d: k for _, d, k, _ in issues}
        assert kinds[pd.Timestamp("2020-01-03")] == "empty"
        assert kinds[pd.Timestamp("2020-01-07")] == "degenerate"

    @pytest.mark.parametrize(
        "name, value",
        [("moon_phase", 42.0), ("ssrd", 0.0), ("tisr", 0.0)],
    )
    def test_uniform_by_design_is_not_degenerate(self, name, value):
        """One moon phase per day; zero sunlight over a domain in darkness."""
        ds = _ds(_JAN).rename({"testvar": name})
        ds[name][4] = value

        assert check_slice_health(ds) == []

    @pytest.mark.parametrize(
        "name, value",
        [("ssrd", 7.5), ("testvar", 0.0)],
    )
    def test_other_constants_still_report(self, name, value):
        """A flux stuck at a non-zero value, or zero where zero means nothing."""
        ds = _ds(_JAN).rename({"testvar": name})
        ds[name][4] = value

        assert [k for _, _, k, _ in check_slice_health(ds)] == ["degenerate"]

    def test_healthy_dataset_reports_nothing(self):
        assert check_slice_health(_ds(_JAN)) == []

    def test_interior_position_is_inspected_not_only_the_last(self):
        """The old check looked at isel(time=-1) and would have missed this."""
        issues = check_slice_health(_ds(_JAN, null_days=[pd.Timestamp("2020-01-04")]))
        assert issues and issues[0][1] == pd.Timestamp("2020-01-04")

    def test_timeless_variable_is_skipped(self):
        ds = xr.Dataset(
            {"bathy": (["lat", "lon"], np.full((3, 3), 5.0))},
            coords={"lat": [30.0, 35.0, 40.0], "lon": [-10.0, -5.0, 0.0]},
        )
        assert check_slice_health(ds) == []

    def test_window_end_covers_a_noon_stamped_last_day(self):
        """
        The window bound is a date; the store's stamps need not be midnight.
        Sliced to the bound verbatim, the last day falls outside the scan and
        its emptiness is never seen — the worst outcome for a check, which is
        to report a bad day clean.
        """
        noon = pd.DatetimeIndex(_JAN) + pd.Timedelta(hours=12)
        last = noon[-1]

        issues = check_slice_health(
            _ds(noon, null_days=[last]),
            window=DateRange(pd.Timestamp(noon[0]).normalize(), last.normalize()),
        )

        assert [(k, d) for _, d, k, _ in issues] == [("empty", last.normalize())]

    def test_window_end_covers_every_hour_of_an_hourly_last_day(self):
        """Same bound, hourly store: 23 of the last day's steps sit after it."""
        hours = pd.date_range("2020-01-01", "2020-01-03 23:00", freq="h")
        empty = hours[hours.normalize() == pd.Timestamp("2020-01-03")]

        issues = check_slice_health(
            _ds(hours, null_days=empty),
            window=DateRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")),
        )

        # One per hourly step, all on the last day — none of which a midnight
        # bound would have reached beyond 00:00.
        assert len(issues) == 24
        assert {d for _, d, _, _ in issues} == {pd.Timestamp("2020-01-03")}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestContiguousBlocks:
    def test_single_run(self):
        assert contiguous_blocks(pd.date_range("2020-01-01", periods=3)) == [
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))
        ]

    def test_two_runs(self):
        dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-09"])
        assert len(contiguous_blocks(dates)) == 2

    def test_empty(self):
        assert contiguous_blocks(pd.DatetimeIndex([])) == []


class TestFormatDateBlocks:
    def test_single_day_renders_bare(self):
        assert format_date_blocks(pd.DatetimeIndex(["2025-06-02"])) == "2025-06-02"

    def test_run_renders_as_a_range(self):
        out = format_date_blocks(pd.date_range("2020-01-01", periods=3))
        assert out == "2020-01-01→2020-01-03"

    def test_empty_renders_as_none(self):
        assert format_date_blocks(pd.DatetimeIndex([])) == "none"

    def test_long_lists_are_truncated(self):
        out = format_date_blocks(
            pd.date_range("2020-01-01", periods=40, freq="3D"), max_blocks=3
        )
        assert "more block(s)" in out


# ---------------------------------------------------------------------------
# audit_parquet_nulls
# ---------------------------------------------------------------------------


class TestAuditParquetNulls:
    @pytest.fixture
    def store(self, tmp_path):
        from h2mare.storage.parquet_store import ParquetStore

        return ParquetStore(tmp_path / "pq")

    def test_wholly_null_column_is_found_from_footer_stats(self, store, tmp_path):
        """The realistic shape: a real float column whose every value is null."""
        import polars as pl
        from conftest import make_grid_df

        from h2mare.storage.audit import audit_parquet_nulls

        df = make_grid_df(
            [pd.Timestamp("2020-01-01").date()], variables={"sst": 20.0, "chl": 1.0}
        ).with_columns(pl.lit(None, dtype=pl.Float64).alias("chl"))
        store.add_data(df)

        findings = audit_parquet_nulls(store.parquet_root)

        assert [c for _, c in findings] == ["chl"]

    def test_null_typed_column_is_found_without_statistics(self, store):
        """A null-typed column carries no stats, so it needs the schema path."""
        from conftest import make_grid_df

        from h2mare.storage.audit import audit_parquet_nulls

        df = make_grid_df(
            [pd.Timestamp("2020-01-01").date()], variables={"sst": 20.0, "chl": 1.0}
        ).with_columns(chl=None)
        store.add_data(df)

        assert [c for _, c in audit_parquet_nulls(store.parquet_root)] == ["chl"]

    def test_populated_columns_are_not_flagged(self, store):
        from conftest import make_grid_df

        from h2mare.storage.audit import audit_parquet_nulls

        store.add_data(
            make_grid_df([pd.Timestamp("2020-01-01").date()], variables={"sst": 20.0})
        )

        assert audit_parquet_nulls(store.parquet_root) == []

    def test_coordinate_columns_are_never_flagged(self, store):
        from conftest import make_grid_df

        from h2mare.storage.audit import audit_parquet_nulls

        store.add_data(
            make_grid_df([pd.Timestamp("2020-01-01").date()], variables={"sst": 20.0})
        )

        assert not any(
            c in {"time", "lon", "lat"}
            for _, c in audit_parquet_nulls(store.parquet_root)
        )


# ---------------------------------------------------------------------------
# known_gaps
#
# A source shipping one file per day produces an *axis* hole when it skips one,
# which is indistinguishable from data the pipeline lost. AVISO has no fsle
# file for 2025-06-02 — its remote listing jumps 20250601 → 20250603 — so that
# day can never be filled. Without somewhere to record it, the checks would
# report the same unfixable day forever, and a check that cries wolf is one
# people stop reading.
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, known_gaps=None):
        self.known_gaps = known_gaps


class TestKnownGapDays:
    def test_single_date(self):
        from h2mare.storage.audit import known_gap_days

        assert list(known_gap_days(_Cfg(["2025-06-02"]))) == [
            pd.Timestamp("2025-06-02")
        ]

    def test_closed_interval_expands(self):
        from h2mare.storage.audit import known_gap_days

        assert len(known_gap_days(_Cfg(["2025-06-02/2025-06-05"]))) == 4

    def test_dates_and_intervals_mix(self):
        from h2mare.storage.audit import known_gap_days

        out = known_gap_days(_Cfg(["2025-01-01", "2025-06-02/2025-06-03"]))
        assert len(out) == 3

    def test_none_is_empty(self):
        from h2mare.storage.audit import known_gap_days

        assert len(known_gap_days(_Cfg(None))) == 0

    def test_malformed_entry_is_skipped_not_raised(self):
        """A typo in a suppression list must not stop a pipeline run."""
        from h2mare.storage.audit import known_gap_days

        out = known_gap_days(_Cfg(["not-a-date", "2025-06-02"]))
        assert list(out) == [pd.Timestamp("2025-06-02")]

    def test_result_is_deduplicated_and_sorted(self):
        from h2mare.storage.audit import known_gap_days

        out = known_gap_days(_Cfg(["2025-06-03", "2025-06-02", "2025-06-03"]))
        assert list(out) == [pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-03")]


class TestKnownGapsSuppressReporting:
    def test_a_known_gap_is_not_reported(self, tmp_path):
        path = _write(_ds(_JAN.drop(pd.Timestamp("2020-01-05"))), tmp_path / "a.zarr")

        gap, _, _ = audit_zarr_file(
            path, known_gaps=pd.DatetimeIndex([pd.Timestamp("2020-01-05")])
        )

        assert gap is None

    def test_an_unlisted_gap_is_still_reported(self, tmp_path):
        path = _write(_ds(_JAN.drop(pd.Timestamp("2020-01-05"))), tmp_path / "a.zarr")

        gap, _, _ = audit_zarr_file(
            path, known_gaps=pd.DatetimeIndex([pd.Timestamp("2020-01-08")])
        )

        assert list(gap.missing) == [pd.Timestamp("2020-01-05")]

    def test_only_the_listed_days_are_dropped(self, tmp_path):
        dropped = _JAN.drop(pd.DatetimeIndex(["2020-01-05", "2020-01-08"]))
        path = _write(_ds(dropped), tmp_path / "a.zarr")

        gap, _, _ = audit_zarr_file(
            path, known_gaps=pd.DatetimeIndex([pd.Timestamp("2020-01-05")])
        )

        assert list(gap.missing) == [pd.Timestamp("2020-01-08")]


class TestVarAuditKnownGaps:
    def test_count_derives_from_the_dates(self, tmp_path):
        from h2mare.storage.audit import VarAudit

        v = VarAudit(
            var_key="fsle",
            store_root=tmp_path,
            n_files=1,
            gaps=[],
            slices=[],
            errors=[],
            known_gaps=pd.DatetimeIndex(["2025-06-02", "2025-06-03"]),
        )

        assert v.n_known_gaps == 2

    def test_defaults_to_none_suppressed(self, tmp_path):
        from h2mare.storage.audit import VarAudit

        v = VarAudit(
            var_key="sst",
            store_root=tmp_path,
            n_files=1,
            gaps=[],
            slices=[],
            errors=[],
        )

        assert v.n_known_gaps == 0
        assert len(v.known_gaps) == 0

    def test_suppressed_days_do_not_make_a_var_fail(self, tmp_path):
        from h2mare.storage.audit import VarAudit

        v = VarAudit(
            var_key="fsle",
            store_root=tmp_path,
            n_files=1,
            gaps=[],
            slices=[],
            errors=[],
            known_gaps=pd.DatetimeIndex(["2025-06-02"]),
        )

        assert v.ok is True


# ---------------------------------------------------------------------------
# Value-scan correctness
#
# Running --values on chl surfaced four defects at once: known_gaps suppressed
# axis gaps but not value findings, so 11 unfixable days reported forever; the
# reductions were computed one at a time, which on a 97 GB store is three real
# re-reads from disk; and the report gave advice ("re-run the download") that
# cannot fix a day the provider never published.
# ---------------------------------------------------------------------------


class TestKnownGapsSuppressValueFindings:
    def test_a_known_gap_is_not_reported_as_empty(self, tmp_path):
        path = _write(
            _ds(_JAN, null_days=[pd.Timestamp("2020-01-05")]), tmp_path / "a.zarr"
        )

        _, slices, error = audit_zarr_file(
            path,
            check_values=True,
            known_gaps=pd.DatetimeIndex([pd.Timestamp("2020-01-05")]),
        )

        # error must be None too: an exception here would empty `slices` and
        # make this pass for entirely the wrong reason.
        assert (slices, error) == ([], None)

    def test_an_unlisted_empty_day_is_still_reported(self, tmp_path):
        path = _write(
            _ds(_JAN, null_days=[pd.Timestamp("2020-01-05")]), tmp_path / "a.zarr"
        )

        _, slices, _ = audit_zarr_file(
            path,
            check_values=True,
            known_gaps=pd.DatetimeIndex([pd.Timestamp("2020-01-08")]),
        )

        assert [s.date for s in slices] == [pd.Timestamp("2020-01-05")]

    def test_a_degenerate_day_can_also_be_suppressed(self, tmp_path):
        path = _write(
            _ds(_JAN, constant_days=[pd.Timestamp("2020-01-05")]), tmp_path / "a.zarr"
        )

        _, slices, _ = audit_zarr_file(
            path,
            check_values=True,
            known_gaps=pd.DatetimeIndex([pd.Timestamp("2020-01-05")]),
        )

        assert slices == []


class TestValueWindow:
    def test_window_bounds_the_scan(self, tmp_path):
        from h2mare.types import DateRange

        path = _write(
            _ds(_JAN, null_days=[pd.Timestamp("2020-01-02")]), tmp_path / "a.zarr"
        )

        _, slices, _ = audit_zarr_file(
            path,
            check_values=True,
            value_window=DateRange(
                start=pd.Timestamp("2020-01-05"), end=pd.Timestamp("2020-01-10")
            ),
        )

        assert slices == []

    def test_a_file_wholly_outside_the_window_is_skipped(self, tmp_path):
        """min/max over a zero-length axis raises rather than returning empty."""
        from h2mare.types import DateRange

        path = _write(_ds(_JAN), tmp_path / "a.zarr")

        gap, slices, error = audit_zarr_file(
            path,
            check_values=True,
            value_window=DateRange(
                start=pd.Timestamp("2030-01-01"), end=pd.Timestamp("2030-12-31")
            ),
        )

        assert (slices, error) == ([], None)

    def test_a_finding_inside_the_window_is_kept(self, tmp_path):
        from h2mare.types import DateRange

        path = _write(
            _ds(_JAN, null_days=[pd.Timestamp("2020-01-07")]), tmp_path / "a.zarr"
        )

        _, slices, _ = audit_zarr_file(
            path,
            check_values=True,
            value_window=DateRange(
                start=pd.Timestamp("2020-01-05"), end=pd.Timestamp("2020-01-10")
            ),
        )

        assert [s.date for s in slices] == [pd.Timestamp("2020-01-07")]


class TestSliceHealthIsOnePass:
    """n_finite/min/max used to be computed one at a time.

    On a store far too large for page cache — chl alone is 97 GB — each
    separate materialisation is a real re-read from disk, so that was three
    times the I/O the check actually needs.
    """

    def _count_computes(self, ds) -> int:
        real = xr.Dataset.compute
        calls = []

        def counting(self, *args, **kwargs):
            calls.append(1)
            return real(self, *args, **kwargs)

        with patch.object(xr.Dataset, "compute", counting):
            check_slice_health(ds)
        return len(calls)

    def test_reductions_are_fused_into_one_compute(self):
        ds = _ds(_JAN, null_days=[pd.Timestamp("2020-01-05")]).chunk({"time": 2})

        assert self._count_computes(ds) == 1

    def test_multiple_variables_still_one_compute(self):
        ds = _ds(_JAN).chunk({"time": 2})
        ds["testvar2"] = ds["testvar"] * 2

        assert self._count_computes(ds) == 1


class TestStoreExists:
    def test_missing_store_is_flagged(self, tmp_path):
        from h2mare.storage.audit import VarAudit

        v = VarAudit(
            var_key="moon",
            store_root=tmp_path / "nope",
            n_files=0,
            gaps=[],
            slices=[],
            errors=[],
            store_exists=False,
        )

        assert v.store_exists is False

    def test_present_store_is_the_default(self, tmp_path):
        from h2mare.storage.audit import VarAudit

        v = VarAudit(
            var_key="sst",
            store_root=tmp_path,
            n_files=1,
            gaps=[],
            slices=[],
            errors=[],
        )

        assert v.store_exists is True


class TestExpectsStore:
    """Which absences are news, and which are the normal state of things."""

    class _Src:
        def __init__(self, source):
            self.source = source

    def test_a_downloaded_variable_should_have_a_store(self):
        from h2mare.storage.audit import expects_store

        assert expects_store(self._Src("cmems")) is True
        assert expects_store(self._Src("aviso")) is True
        assert expects_store(self._Src("cds")) is True

    def test_a_computed_variable_need_not(self):
        """`moon` is written straight into h2ds and never gets a directory."""
        from h2mare.storage.audit import expects_store

        assert expects_store(self._Src("python")) is False

    def test_the_shipped_config_marks_only_derived_keys_as_storeless(self):
        """Pins the discriminator against the real config: if a downloaded
        variable ever landed on the storeless side, a wrong STORE_ROOT would
        stop being reported for it."""
        import msgspec
        import yaml

        from h2mare.models import SYSTEM_VAR_KEYS, AppConfig
        from h2mare.storage.audit import expects_store

        raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        config = msgspec.convert(
            {"variables": raw["variables"], "secrets": {}}, AppConfig, strict=False
        )

        storeless = {
            key for key, cfg in config.variables.items() if not expects_store(cfg)
        }
        assert storeless == SYSTEM_VAR_KEYS


class TestShippedKnownGaps:
    """Guard the suppression lists in the tracked config.yaml.

    A malformed entry is skipped with a warning, so it fails safe — the day
    just keeps being reported. The dangerous typo is the opposite one: an
    interval wider than intended, which silently hides days nobody confirmed.
    """

    def _shipped(self, var_key):
        import msgspec
        import yaml

        from h2mare.models import AppConfig
        from h2mare.storage.audit import known_gap_days

        raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        config = msgspec.convert(
            {"variables": raw["variables"], "secrets": {}}, AppConfig, strict=False
        )
        return known_gap_days(config.variables[var_key])

    def test_chl_suppresses_exactly_the_confirmed_days(self):
        assert [str(d.date()) for d in self._shipped("chl")] == [
            "1998-11-17",
            "1998-11-18",
            "1998-11-19",
            "1998-11-20",
            "1998-12-17",
            "1999-01-25",
            "1999-11-17",
            "1999-11-18",
            "2000-11-17",
            "2001-11-18",
            "2002-12-31",
        ]

    def test_fsle_suppresses_exactly_the_confirmed_day(self):
        assert [str(d.date()) for d in self._shipped("fsle")] == ["2025-06-02"]
