"""Tests for write_append_zarr and atomic swap behaviour in storage.py."""

import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.storage.storage import _append_data, write_append_zarr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ds(start: str = "2020-01-01", n_days: int = 5, seed: int = 0) -> xr.Dataset:
    """Varied (non-constant) data so have_vars_unique_values does not fire."""
    times = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    data = rng.uniform(10.0, 30.0, size=(n_days, 3, 3))
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


# ---------------------------------------------------------------------------
# write_append_zarr — new write path
# ---------------------------------------------------------------------------


class TestNewWrite:
    def test_creates_zarr_directory(self, tmp_path):
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds(), path)
        assert path.exists()

    def test_written_data_is_readable(self, tmp_path):
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds(), path)
        ds = xr.open_zarr(path)
        assert "sst" in ds.data_vars
        assert len(ds.time) == 5
        ds.close()

    def test_verification_failure_removes_partial_write(self, tmp_path, monkeypatch):
        """If the post-write open_zarr verification fails, the zarr directory is
        cleaned up and RuntimeError is raised — no partial file left behind."""
        path = tmp_path / "sst.zarr"

        def bad_open(*args, **kwargs):
            raise OSError("simulated corruption")

        monkeypatch.setattr("h2mare.storage.storage.xr.open_zarr", bad_open)

        with pytest.raises(RuntimeError, match="verification failed"):
            write_append_zarr("sst", _make_ds(), path)

        assert not path.exists()


# ---------------------------------------------------------------------------
# _append_data — atomic backup-swap
# ---------------------------------------------------------------------------


class TestAtomicSwap:
    def test_no_bak_file_after_success(self, tmp_path):
        """.bak file must be removed after a successful append."""
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)
        _append_data("sst", _make_ds("2020-01-06", 5), path)
        assert not path.with_name(path.name + ".bak").exists()

    def test_no_tmp_file_after_success(self, tmp_path):
        """.tmp directory must be removed after a successful append."""
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)
        _append_data("sst", _make_ds("2020-01-06", 5), path)
        assert not path.with_name(path.name + ".tmp").exists()

    def test_result_spans_both_periods(self, tmp_path):
        """Appended zarr should contain all timesteps from both writes."""
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)
        _append_data("sst", _make_ds("2020-01-06", 5), path)
        ds = xr.open_zarr(path)
        assert len(ds.time) == 10
        ds.close()

    def test_original_restored_when_final_move_fails(self, tmp_path):
        """If renaming tmp → final fails, the original is restored from backup.

        Uses overlapping dates so the rewrite (swap) path is exercised — a
        clean trailing append now goes through the in-place fast path.
        """
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)

        call_count = [0]
        original_move = shutil.move

        def failing_move(src, dst):
            call_count[0] += 1
            if call_count[0] == 2:  # second call: tmp → final
                raise OSError("simulated disk full")
            return original_move(src, dst)

        with patch("h2mare.storage.storage.shutil.move", side_effect=failing_move):
            with pytest.raises(RuntimeError, match="original restored from backup"):
                _append_data("sst", _make_ds("2020-01-03", 5), path)

        # Original data still intact
        ds = xr.open_zarr(path)
        assert len(ds.time) == 5
        ds.close()

    def test_no_orphan_bak_after_swap_failure(self, tmp_path):
        """After a failed swap the .bak is moved back; no .bak should remain."""
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)

        call_count = [0]
        original_move = shutil.move

        def failing_move(src, dst):
            call_count[0] += 1
            if call_count[0] == 2:
                raise OSError("simulated disk full")
            return original_move(src, dst)

        with patch("h2mare.storage.storage.shutil.move", side_effect=failing_move):
            with pytest.raises(RuntimeError):
                _append_data("sst", _make_ds("2020-01-03", 5), path)

        assert not path.with_name(path.name + ".bak").exists()


# ---------------------------------------------------------------------------
# _append_data — in-place append fast path
# ---------------------------------------------------------------------------


class TestFastAppend:
    """A clean trailing append (same variables, same grid, strictly after the
    stored dates) must extend the zarr in place instead of rewriting it."""

    def test_fast_path_skips_rewrite(self, tmp_path):
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)

        # If the rewrite path were taken, this sentinel would raise.
        with patch(
            "h2mare.storage.storage._resolve_overlap",
            side_effect=AssertionError("rewrite path used for a clean append"),
        ):
            _append_data("sst", _make_ds("2020-01-06", 3), path)

        ds = xr.open_zarr(path, consolidated=False)
        times = pd.DatetimeIndex(ds.time.values)
        assert len(times) == 8
        assert times.is_monotonic_increasing and times.is_unique
        ds.close()

    def test_appended_values_match_source(self, tmp_path):
        path = tmp_path / "sst.zarr"
        ds_old = _make_ds("2020-01-01", 5, seed=1)
        ds_old.to_zarr(path)
        ds_new = _make_ds("2020-01-06", 3, seed=2)
        _append_data("sst", ds_new, path)

        result = xr.open_zarr(path, consolidated=False)
        np.testing.assert_allclose(
            result.sst.sel(time="2020-01-02").values,
            ds_old.sst.sel(time="2020-01-02").values,
        )
        np.testing.assert_allclose(
            result.sst.sel(time="2020-01-08").values,
            ds_new.sst.sel(time="2020-01-08").values,
        )
        result.close()

    def test_unaligned_chunk_boundary(self, tmp_path):
        """Appending onto a partially-filled boundary chunk must keep all values."""
        path = tmp_path / "sst.zarr"
        ds_old = _make_ds("2020-01-01", 7, seed=3)
        ds_old.chunk({"time": 5}).to_zarr(path)  # last zarr chunk holds 2 of 5
        ds_new = _make_ds("2020-01-08", 4, seed=4)
        _append_data("sst", ds_new.chunk({"time": 3}), path)

        result = xr.open_zarr(path, consolidated=False)
        assert len(result.time) == 11
        np.testing.assert_allclose(
            result.sst.sel(time="2020-01-07").values,
            ds_old.sst.sel(time="2020-01-07").values,
        )
        np.testing.assert_allclose(
            result.sst.sel(time="2020-01-11").values,
            ds_new.sst.sel(time="2020-01-11").values,
        )
        result.close()

    def test_grid_mismatch_falls_back_to_rewrite(self, tmp_path):
        from h2mare.storage import storage as storage_mod

        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)

        times = pd.date_range("2020-01-06", periods=3, freq="D")
        rng = np.random.default_rng(5)
        ds_new = xr.Dataset(
            {"sst": (["time", "lat", "lon"], rng.uniform(10, 30, (3, 3, 3)))},
            coords={
                "time": times,
                "lat": [30.0, 36.0, 40.0],  # differs from stored grid
                "lon": [-10.0, -5.0, 0.0],
            },
        )
        with patch(
            "h2mare.storage.storage._resolve_overlap",
            wraps=storage_mod._resolve_overlap,
        ) as spy:
            _append_data("sst", ds_new, path)
        spy.assert_called_once()

    def test_overlapping_dates_fall_back_to_rewrite(self, tmp_path):
        from h2mare.storage import storage as storage_mod

        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)

        with patch(
            "h2mare.storage.storage._resolve_overlap",
            wraps=storage_mod._resolve_overlap,
        ) as spy:
            _append_data("sst", _make_ds("2020-01-04", 5), path)
        spy.assert_called_once()

        ds = xr.open_zarr(path, consolidated=False)
        assert pd.DatetimeIndex(ds.time.values).is_unique
        ds.close()


# ---------------------------------------------------------------------------
# _append_data — variable-addition path
# ---------------------------------------------------------------------------


class TestVariableAddition:
    """
    When ds_new contains only variables absent from the existing zarr,
    _append_data must merge (not replace) so all existing data is preserved.
    """

    def _make_disjoint_ds(
        self,
        var_name: str,
        start: str = "2020-01-01",
        n_days: int = 5,
        seed: int = 1,
    ) -> xr.Dataset:
        times = pd.date_range(start, periods=n_days, freq="D")
        rng = np.random.default_rng(seed)
        data = rng.uniform(0, 1, size=(n_days, 3, 3))
        return xr.Dataset(
            {var_name: (["time", "lat", "lon"], data)},
            coords={
                "time": times,
                "lat": [30.0, 35.0, 40.0],
                "lon": [-10.0, -5.0, 0.0],
            },
        )

    def test_existing_variable_preserved(self, tmp_path):
        """The original variable must still be present after adding a new one."""
        path = tmp_path / "h2ds.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)  # writes 'sst'
        _append_data("h2ds", self._make_disjoint_ds("chl"), path)  # adds 'chl'

        ds = xr.open_zarr(path, consolidated=False)
        assert "sst" in ds.data_vars
        ds.close()

    def test_new_variable_added(self, tmp_path):
        """The new variable must be present in the result."""
        path = tmp_path / "h2ds.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)
        _append_data("h2ds", self._make_disjoint_ds("chl"), path)

        ds = xr.open_zarr(path, consolidated=False)
        assert "chl" in ds.data_vars
        ds.close()

    def test_time_steps_unchanged(self, tmp_path):
        """No time steps should be gained or lost during a variable-addition merge."""
        path = tmp_path / "h2ds.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)
        _append_data("h2ds", self._make_disjoint_ds("chl"), path)

        ds = xr.open_zarr(path, consolidated=False)
        assert len(ds.time) == 5
        ds.close()

    def test_multiple_new_variables_all_added(self, tmp_path):
        """All variables in the new dataset are added when all are disjoint."""
        path = tmp_path / "h2ds.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)

        times = pd.date_range("2020-01-01", periods=5, freq="D")
        rng = np.random.default_rng(99)
        ds_new = xr.Dataset(
            {
                "thetao_100": (["time", "lat", "lon"], rng.uniform(0, 1, (5, 3, 3))),
                "thetao_500": (["time", "lat", "lon"], rng.uniform(0, 1, (5, 3, 3))),
            },
            coords={
                "time": times,
                "lat": [30.0, 35.0, 40.0],
                "lon": [-10.0, -5.0, 0.0],
            },
        )
        _append_data("h2ds", ds_new, path)

        ds = xr.open_zarr(path, consolidated=False)
        assert "sst" in ds.data_vars
        assert "thetao_100" in ds.data_vars
        assert "thetao_500" in ds.data_vars
        ds.close()


# ---------------------------------------------------------------------------
# _append_data — partial variable set (subset compile)
# ---------------------------------------------------------------------------


def _make_two_var_ds(start: str, n_days: int, seed: int = 0) -> xr.Dataset:
    times = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "sst": (["time", "lat", "lon"], rng.uniform(10, 30, (n_days, 3, 3))),
            "adt": (["time", "lat", "lon"], rng.uniform(-1, 1, (n_days, 3, 3))),
        },
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


class TestPartialVariableAppend:
    """A ds_new carrying only a subset of the stored variables (e.g. a
    `run -v ssh` compile) must not NaN-wipe the other variables over its
    window — regression for the h2ds corruption seen in production."""

    def test_other_variables_survive_subset_extension(self, tmp_path):
        path = tmp_path / "h2ds.zarr"
        ds_orig = _make_two_var_ds("2020-01-01", 10)
        ds_orig.to_zarr(path)

        # adt-only update overlapping Jan 8-10 and extending to Jan 12
        ds_new = _make_two_var_ds("2020-01-08", 5, seed=1)[["adt"]]
        _append_data("h2ds", ds_new, path)

        out = xr.open_zarr(path, consolidated=False)
        assert len(out.time) == 12
        # sst preserved over the overlap window (was NaN-wiped before the fix)
        np.testing.assert_allclose(
            out.sst.sel(time="2020-01-09").values,
            ds_orig.sst.sel(time="2020-01-09").values,
        )
        # sst NaN only at genuinely new dates
        assert np.isnan(out.sst.sel(time="2020-01-12").values).all()
        # adt over the window comes from ds_new
        np.testing.assert_allclose(
            out.adt.sel(time="2020-01-09").values,
            ds_new.adt.sel(time="2020-01-09").values,
        )
        out.close()

    def test_subset_full_overlap_preserves_other_variables(self, tmp_path):
        """Full-overlap replace with a subset ds_new keeps the absent variables."""
        path = tmp_path / "h2ds.zarr"
        ds_orig = _make_two_var_ds("2020-01-01", 5)
        ds_orig.to_zarr(path)

        ds_new = _make_two_var_ds("2020-01-01", 5, seed=2)[["adt"]]
        _append_data("h2ds", ds_new, path)

        out = xr.open_zarr(path, consolidated=False)
        assert len(out.time) == 5
        np.testing.assert_allclose(out.sst.values, ds_orig.sst.values)
        np.testing.assert_allclose(out.adt.values, ds_new.adt.values)
        out.close()


# ---------------------------------------------------------------------------
# _append_data — overlap resolution
# ---------------------------------------------------------------------------


class TestOverlapResolution:
    def test_no_duplicate_timestamps_when_new_starts_before_old(self, tmp_path):
        """
        When new data starts before old data and ends before old data ends,
        _resolve_overlap produces an empty subset. Previously the fallback
        returned ds_old, causing duplicate time steps after concat.
        After the fix, the fallback returns None and no duplicates appear.
        """
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-05", n_days=5).to_zarr(path)  # Jan 5–9
        _append_data("sst", _make_ds("2020-01-03", n_days=5), path)  # Jan 3–7

        ds = xr.open_zarr(path, consolidated=False)
        times = pd.DatetimeIndex(ds.time.values)
        assert times.is_unique, "Duplicate timestamps after append"
        ds.close()
