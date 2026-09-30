"""Tests for write_append_zarr and atomic swap behaviour in storage.py."""

import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.storage.storage import _append_data, write_append_zarr
from h2mare.storage.xarray_helpers import drop_conflicting_missing_value

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ds(start: str = "2020-01-01", n_days: int = 5, seed: int = 0) -> xr.Dataset:
    """Varied (non-constant) data, so no slice looks degenerate."""
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

    def test_stale_backup_does_not_nest_the_store_on_rollback(self, tmp_path):
        """
        A crash between the tmp → path move and the rmtree that follows it
        leaves *both* path and path.bak present, and _restore_orphaned_backup
        only handles the other direction. shutil.move onto an existing
        directory moves *into* it, so the live store used to land at
        path.bak/sst.zarr — and the rollback then restored that directory,
        whose top level is the *stale* store. The failure is silent: the
        restored path opens fine and holds the wrong data.
        """
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)

        # A stale backup that is itself a valid zarr, holding different data.
        # If it survives the swap it is what the rollback restores.
        stale = path.with_name(path.name + ".bak")
        _make_ds("2019-01-01", 3, seed=9).to_zarr(stale)

        call_count = [0]
        original_move = shutil.move

        def failing_move(src, dst):
            call_count[0] += 1
            if call_count[0] == 2:  # second call: tmp → final, forces rollback
                raise OSError("simulated disk full")
            return original_move(src, dst)

        with patch("h2mare.storage.storage.shutil.move", side_effect=failing_move):
            with pytest.raises(RuntimeError, match="original restored from backup"):
                _append_data("sst", _make_ds("2020-01-03", 5), path)

        assert not (path / "sst.zarr").exists(), "the store was nested inside itself"
        ds = xr.open_zarr(path, consolidated=False)
        try:
            assert len(ds.time) == 5, (
                "rollback restored the stale backup, not the store"
            )
            assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("2020-01-01")
        finally:
            ds.close()

    def test_stale_backup_is_cleared_on_a_successful_swap(self, tmp_path):
        """The same stale backup must not survive a swap that succeeds."""
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)
        stale = path.with_name(path.name + ".bak")
        _make_ds("2019-01-01", 3, seed=9).to_zarr(stale)

        _append_data("sst", _make_ds("2020-01-03", 5), path)

        assert not stale.exists()
        assert not (path / "sst.zarr").exists()
        ds = xr.open_zarr(path, consolidated=False)
        try:
            assert len(ds.time) == 7  # Jan 1-2 retained + Jan 3-7 incoming
        finally:
            ds.close()


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

    def test_wider_extent_falls_back_to_rewrite(self, tmp_path):
        """
        The fast path appends along time in place, so it can only run when the
        other axes are identical. A wider bbox on the same lattice is a legal
        append — the cells at the ends are new — but not one it can do.
        """
        from h2mare.storage import storage as storage_mod

        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)

        times = pd.date_range("2020-01-06", periods=3, freq="D")
        rng = np.random.default_rng(5)
        ds_new = xr.Dataset(
            {"sst": (["time", "lat", "lon"], rng.uniform(10, 30, (3, 4, 3)))},
            coords={
                "time": times,
                "lat": [30.0, 35.0, 40.0, 45.0],  # same 5° lattice, one cell wider
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

    def test_static_variable_recompute_difference_does_not_conflict(self, tmp_path):
        """Time-less variables (e.g. bathy) are merged by xr.concat, not
        concatenated, and merging demands exact equality — a float-level
        recompute difference between the stored and freshly compiled copy
        raised MergeError. The fresh copy in ds_new must win instead."""
        path = tmp_path / "h2ds.zarr"
        ds_orig = _make_two_var_ds("2020-01-01", 5)
        ds_orig["bathy"] = (["lat", "lon"], np.full((3, 3), 100.0))
        ds_orig.to_zarr(path)

        ds_new = _make_two_var_ds("2020-01-04", 4, seed=3)
        ds_new["bathy"] = (["lat", "lon"], np.full((3, 3), 100.0001))
        _append_data("h2ds", ds_new, path)

        out = xr.open_zarr(path, consolidated=False)
        assert len(out.time) == 7  # Jan 1-7 (old 1-5 ∪ new 4-7)
        np.testing.assert_allclose(out.bathy.values, 100.0001)
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
# _append_data — middle-window rewrite (tail preservation)
# ---------------------------------------------------------------------------


class TestMiddleWindowAppend:
    """A ds_new covering a window in the middle of the stored range (explicit
    --start/--end compile) must preserve the existing dates after the window —
    the old head+new concat silently dropped them."""

    def test_tail_dates_survive_middle_window(self, tmp_path):
        path = tmp_path / "sst.zarr"
        ds_orig = _make_ds("2020-01-01", 20)
        ds_orig.to_zarr(path)

        ds_new = _make_ds("2020-01-05", 6, seed=7)  # Jan 5-10
        _append_data("sst", ds_new, path)

        out = xr.open_zarr(path, consolidated=False)
        assert len(out.time) == 20
        # head and tail keep the stored values; the window comes from ds_new
        np.testing.assert_allclose(
            out.sst.sel(time="2020-01-03").values,
            ds_orig.sst.sel(time="2020-01-03").values,
        )
        np.testing.assert_allclose(
            out.sst.sel(time="2020-01-07").values,
            ds_new.sst.sel(time="2020-01-07").values,
        )
        np.testing.assert_allclose(
            out.sst.sel(time="2020-01-15").values,
            ds_orig.sst.sel(time="2020-01-15").values,
        )
        out.close()

    def test_tail_survives_when_window_starts_at_file_start(self, tmp_path):
        """Head empty (window starts at the stored start) — the tail alone
        must still be retained."""
        path = tmp_path / "sst.zarr"
        ds_orig = _make_ds("2020-01-01", 20)
        ds_orig.to_zarr(path)

        ds_new = _make_ds("2020-01-01", 5, seed=8)  # Jan 1-5
        _append_data("sst", ds_new, path)

        out = xr.open_zarr(path, consolidated=False)
        assert len(out.time) == 20
        np.testing.assert_allclose(
            out.sst.sel(time="2020-01-12").values,
            ds_orig.sst.sel(time="2020-01-12").values,
        )
        out.close()

    def test_subset_middle_window_preserves_everything_around_it(self, tmp_path):
        """Subset vars + middle window combined: the absent variable keeps its
        full span, the updated one keeps head and tail."""
        path = tmp_path / "h2ds.zarr"
        ds_orig = _make_two_var_ds("2020-01-01", 20)
        ds_orig.to_zarr(path)

        ds_new = _make_two_var_ds("2020-01-05", 6, seed=9)[["adt"]]
        _append_data("h2ds", ds_new, path)

        out = xr.open_zarr(path, consolidated=False)
        assert len(out.time) == 20
        np.testing.assert_allclose(out.sst.values, ds_orig.sst.values)
        np.testing.assert_allclose(
            out.adt.sel(time="2020-01-07").values,
            ds_new.adt.sel(time="2020-01-07").values,
        )
        np.testing.assert_allclose(
            out.adt.sel(time="2020-01-15").values,
            ds_orig.adt.sel(time="2020-01-15").values,
        )
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

    def test_backfill_entirely_before_stored_window_is_not_duplicated(self, tmp_path):
        """
        A backfill sitting entirely *before* the stored data must not re-add
        that data on both sides of the incoming window.

        Zarr files are named per year, so backfilling January against a file
        that already holds June resolves to the same path and takes the append
        path. The disjoint fallback used to return ds_old whole, and
        _append_data concatenated it as the retained head *and* selected it
        again as the retained tail: every stored step was written twice and the
        time axis came back non-monotonic, logged as a success.
        """
        path = tmp_path / "sst.zarr"
        stored = _make_ds("2021-06-01", n_days=5, seed=1)
        stored.to_zarr(path)

        incoming = _make_ds("2021-01-01", n_days=3, seed=2)
        _append_data("sst", incoming, path)

        ds = xr.open_zarr(path, consolidated=False)
        try:
            times = pd.DatetimeIndex(ds.time.values)
            assert len(times) == 8, f"expected 3 + 5 steps, got {len(times)}"
            assert times.is_unique, "stored steps were re-added by the backfill"
            assert times.is_monotonic_increasing, "time axis is out of order"
            # Each window keeps its own values — neither side overwrote the other.
            np.testing.assert_allclose(
                ds.sst.sel(time=incoming.time).values, incoming.sst.values
            )
            np.testing.assert_allclose(
                ds.sst.sel(time=stored.time).values, stored.sst.values
            )
        finally:
            ds.close()

    def test_append_entirely_after_stored_window_keeps_both(self, tmp_path):
        """
        The mirror case, through the rewrite path.

        A clean trailing append normally short-circuits into
        _try_append_fast_path and never reaches _resolve_overlap, so the fast
        path is disabled here to exercise the disjoint branch itself — the same
        branch the backfill case above goes through, from the other side.
        """
        path = tmp_path / "sst.zarr"
        stored = _make_ds("2021-01-01", n_days=3, seed=1)
        stored.to_zarr(path)

        incoming = _make_ds("2021-06-01", n_days=5, seed=2)
        with patch("h2mare.storage.storage._try_append_fast_path", return_value=False):
            _append_data("sst", incoming, path)

        ds = xr.open_zarr(path, consolidated=False)
        try:
            times = pd.DatetimeIndex(ds.time.values)
            assert len(times) == 8, f"expected 3 + 5 steps, got {len(times)}"
            assert times.is_unique
            assert times.is_monotonic_increasing
            np.testing.assert_allclose(
                ds.sst.sel(time=stored.time).values, stored.sst.values
            )
            np.testing.assert_allclose(
                ds.sst.sel(time=incoming.time).values, incoming.sst.values
            )
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# Sub-daily (hourly) axes — the rewrite path resolves boundaries by instant
# ---------------------------------------------------------------------------


def _make_hourly_ds(
    start: str = "2020-01-01", n_hours: int = 48, seed: int = 0
) -> xr.Dataset:
    """Hourly counterpart of :func:`_make_ds`."""
    times = pd.date_range(start, periods=n_hours, freq="h")
    rng = np.random.default_rng(seed)
    data = rng.uniform(10.0, 30.0, size=(n_hours, 3, 3))
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


def _noon_ds(start: str, n_days: int, seed: int = 0) -> xr.Dataset:
    """Daily data stamped at 12:00 — one step per day, not sub-daily."""
    ds = _make_ds(start, n_days=n_days, seed=seed)
    return ds.assign_coords(time=ds.time.to_index() + pd.Timedelta(hours=12))


class TestSubDailyRewrite:
    """
    The overlap rewrite resolves its retained head and tail by *instant*, not by
    whole day. Day-granular boundaries used to delete 23 steps per boundary on
    an hourly axis — silently, since the surviving values were all correct.
    """

    def test_overlapping_subdaily_write_keeps_every_step(self, tmp_path):
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_hourly_ds("2020-01-01", n_hours=96), path)

        # Overlaps the middle of the stored window — the rewrite path.
        write_append_zarr("sst", _make_hourly_ds("2020-01-02", n_hours=48), path)

        ds = xr.open_zarr(path, consolidated=False)
        times = pd.DatetimeIndex(ds.time.values)
        ds.close()
        expected = pd.date_range("2020-01-01", periods=96, freq="h")
        assert times.is_unique and times.is_monotonic_increasing
        assert len(times) == 96, f"lost {96 - len(times)} step(s)"
        assert list(times) == list(expected)

    def test_overlapping_subdaily_write_keeps_the_tail(self, tmp_path):
        """Stored steps later the same day as the incoming end must survive."""
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_hourly_ds("2020-01-01", n_hours=96), path)
        # Ends mid-day (Jan 2 11:00), so the tail resumes at Jan 2 12:00.
        write_append_zarr("sst", _make_hourly_ds("2020-01-02", n_hours=12), path)

        ds = xr.open_zarr(path, consolidated=False)
        times = pd.DatetimeIndex(ds.time.values)
        ds.close()
        assert len(times) == 96
        assert pd.Timestamp("2020-01-02 12:00") in times

    def test_incoming_wins_inside_its_window(self, tmp_path):
        """Merge semantics are unchanged: incoming data wins where it has rows."""
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_hourly_ds("2020-01-01", n_hours=96), path)
        incoming = _make_hourly_ds("2020-01-02", n_hours=48, seed=7)
        write_append_zarr("sst", incoming, path)

        ds = xr.open_zarr(path, consolidated=False)
        np.testing.assert_allclose(
            ds.sst.sel(time="2020-01-02T05:00").values,
            incoming.sst.sel(time="2020-01-02T05:00").values,
        )
        ds.close()

    def test_subdaily_strictly_after_append_still_works(self, tmp_path):
        """The fast path stays in play for a clean trailing append."""
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_hourly_ds("2020-01-01", n_hours=48), path)
        write_append_zarr(
            "sst", _make_hourly_ds("2020-01-03", n_hours=48, seed=1), path
        )

        ds = xr.open_zarr(path, consolidated=False)
        times = pd.DatetimeIndex(ds.time.values)
        assert len(times) == 96
        assert times.is_unique and times.is_monotonic_increasing
        ds.close()

    def test_daily_overlap_is_unaffected(self, tmp_path):
        """Daily merge semantics are deliberate and must not move."""
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds("2020-01-01", n_days=10), path)
        write_append_zarr("sst", _make_ds("2020-01-05", n_days=5, seed=1), path)

        ds = xr.open_zarr(path, consolidated=False)
        times = pd.DatetimeIndex(ds.time.values)
        assert len(times) == 10, "daily overlap merge changed behaviour"
        assert times.is_unique
        ds.close()

    def test_noon_stamped_daily_overlap_keeps_every_day(self, tmp_path):
        """
        Regression for a defect the instant-based boundaries also fix: with a
        normalized head cutoff, a noon-stamped store lost the day before the
        incoming window (2020-01-04T12:00 fell after the midnight cutoff).
        """
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _noon_ds("2020-01-01", 10), path)
        write_append_zarr("sst", _noon_ds("2020-01-05", 5, seed=1), path)

        ds = xr.open_zarr(path, consolidated=False)
        times = pd.DatetimeIndex(ds.time.values)
        ds.close()
        # 2020-01-04T12:00 is dropped today.
        assert len(times) == 10, f"lost {10 - len(times)} day(s): {times}"


# ---------------------------------------------------------------------------
# Root attributes across appends
#
# The in-place fast path ends in to_zarr(append_dim="time"), which rewrites the
# group attrs from the incoming dataset — and that carries none, so the store's
# own metadata was wiped. The rewrite paths keep them, which is why this only
# bit strictly-after appends: the common case, and the one every daily run
# takes. source_datasets is the casualty that matters — provenance stamped by
# one run vanished on the next, so merge_records never found anything to merge
# with. The merge/variable-addition cases below are guards, not regressions.
# ---------------------------------------------------------------------------


def _stamp(path, **attrs) -> None:
    import zarr

    zarr.open_group(str(path), mode="r+").attrs.update(attrs)


def _attrs(path) -> dict:
    import zarr

    return dict(zarr.open_group(str(path), mode="r").attrs)


class TestRootAttrsSurviveAppend:
    def test_attrs_survive_a_strictly_after_append(self, tmp_path):
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds("2020-01-01", 5), path)
        _stamp(path, source_datasets='[{"dataset_id": "a"}]')

        write_append_zarr("sst", _make_ds("2020-01-06", 5), path)

        assert _attrs(path)["source_datasets"] == '[{"dataset_id": "a"}]'

    def test_attrs_survive_an_overlapping_merge(self, tmp_path):
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds("2020-01-01", 5), path)
        _stamp(path, source_datasets='[{"dataset_id": "a"}]')

        write_append_zarr("sst", _make_ds("2020-01-03", 5), path)

        assert _attrs(path)["source_datasets"] == '[{"dataset_id": "a"}]'

    def test_attrs_survive_a_variable_addition(self, tmp_path):
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds("2020-01-01", 5), path)
        _stamp(path, source_datasets='[{"dataset_id": "a"}]')

        other = _make_ds("2020-01-01", 5).rename({"sst": "chl"})
        write_append_zarr("sst", other, path)

        assert _attrs(path)["source_datasets"] == '[{"dataset_id": "a"}]'

    def test_multiple_attrs_are_all_kept(self, tmp_path):
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds("2020-01-01", 5), path)
        _stamp(path, source_datasets="[]", note="keep me")

        write_append_zarr("sst", _make_ds("2020-01-06", 5), path)

        assert _attrs(path)["note"] == "keep me"

    def test_a_value_written_by_the_append_still_wins(self, tmp_path):
        """Restoration fills gaps; it must not clobber what the write set."""
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds("2020-01-01", 5), path)
        _stamp(path, note="old")

        fresh = _make_ds("2020-01-06", 5)
        fresh.attrs["note"] = "new"
        write_append_zarr("sst", fresh, path)

        assert _attrs(path)["note"] == "new"

    def test_a_store_without_attrs_is_unaffected(self, tmp_path):
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds("2020-01-01", 5), path)

        write_append_zarr("sst", _make_ds("2020-01-06", 5), path)

        ds = xr.open_zarr(path)
        assert len(ds.time) == 10
        ds.close()


# ---------------------------------------------------------------------------
# int16 store encoding (opt-in)
# ---------------------------------------------------------------------------


class TestInt16Encoding:
    """Packing is opt-in and must survive both append paths."""

    @staticmethod
    def _on_disk_dtype(path, var="sst"):
        import zarr

        return zarr.open_group(str(path), mode="r")[var].dtype

    def test_default_write_is_unchanged(self, tmp_path):
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds("2020-01-01", 5), path)
        assert self._on_disk_dtype(path) == np.dtype("float64")

    def test_encoding_is_applied_on_first_write(self, tmp_path):
        from h2mare.storage.xarray_helpers import int16_encoding

        path = tmp_path / "sst.zarr"
        ds = _make_ds("2020-01-01", 5)
        write_append_zarr("sst", ds, path, encoding=int16_encoding(ds))
        assert self._on_disk_dtype(path) == np.dtype("int16")

    def test_values_survive_the_round_trip(self, tmp_path):
        from h2mare.storage.xarray_helpers import int16_encoding

        path = tmp_path / "sst.zarr"
        ds = _make_ds("2020-01-01", 5)
        write_append_zarr("sst", ds, path, encoding=int16_encoding(ds))

        back = xr.open_zarr(path, consolidated=False)
        err = float(np.abs(back.sst - ds.sst).max())
        rng = float(ds.sst.max() - ds.sst.min())
        back.close()
        assert err / rng < 1e-4, f"quantisation error too large: {err}"

    def test_encoding_survives_a_strictly_after_append(self, tmp_path):
        from h2mare.storage.xarray_helpers import int16_encoding

        path = tmp_path / "sst.zarr"
        ds = _make_ds("2020-01-01", 5)
        write_append_zarr("sst", ds, path, encoding=int16_encoding(ds))
        write_append_zarr("sst", _make_ds("2020-01-06", 5, seed=1), path)

        assert self._on_disk_dtype(path) == np.dtype("int16")
        back = xr.open_zarr(path, consolidated=False)
        assert back.sizes["time"] == 10
        back.close()

    def test_encoding_survives_an_overlapping_rewrite(self, tmp_path):
        from h2mare.storage.xarray_helpers import int16_encoding

        path = tmp_path / "sst.zarr"
        ds = _make_ds("2020-01-01", 10)
        write_append_zarr("sst", ds, path, encoding=int16_encoding(ds))
        write_append_zarr("sst", _make_ds("2020-01-05", 5, seed=1), path)

        assert self._on_disk_dtype(path) == np.dtype("int16"), (
            "the rewrite path dropped the packing"
        )

    def test_degenerate_variable_is_left_alone(self, tmp_path):
        """A constant field has no range to scale, so packing is skipped."""
        from h2mare.storage.xarray_helpers import int16_encoding

        ds = _make_ds("2020-01-01", 3)
        ds["sst"] = ds["sst"] * 0 + 7.0
        assert int16_encoding(ds) == {}


class TestPackedRangeGuard:
    """
    A packed store's scale is fixed at creation and inherited by every append,
    so data outside it has no encoding. It does not clip — it overflows int16
    and wraps back into the middle of the range, which reads as ordinary data.
    The store must be repacked over the wider range before the append, or —
    where no scale can be derived — the append refused.
    """

    @staticmethod
    def _tight_encoding(lo: float, hi: float) -> dict:
        """
        Packing that spans exactly [lo, hi] with no headroom.

        Built by hand rather than via int16_encoding so this pins the guard
        itself and not whatever margin _INT16_HEADROOM currently carries.
        """
        import zarr

        return {
            "sst": {
                "dtype": "int16",
                "scale_factor": (hi - lo) / 65000.0,
                "add_offset": (hi + lo) / 2.0,
                "_FillValue": -32767,
                "compressors": [zarr.codecs.ZstdCodec(level=1)],
            }
        }

    def _store_scaled_for(self, path, lo, hi):
        ds = _make_ds("2020-01-01", 5)
        ds["sst"] = ds["sst"] * 0 + np.linspace(lo, hi, ds["sst"].size).reshape(
            ds["sst"].shape
        )
        write_append_zarr("sst", ds, path, encoding=self._tight_encoding(lo, hi))
        return ds

    @staticmethod
    def _far(value: float = 200.0) -> xr.Dataset:
        """An append strictly after the store, well outside its scale."""
        far = _make_ds("2020-01-06", 5, seed=1)
        far["sst"] = far["sst"] * 0 + value
        return far

    def _unrepackable_store(self, path):
        """
        A packed store holding no data at all: with a constant increment the
        union range is a single value, so there is no span to scale over.
        """
        ds = _make_ds("2020-01-01", 5)
        ds["sst"] = ds["sst"] * np.nan
        write_append_zarr("sst", ds, path, encoding=self._tight_encoding(10.0, 20.0))

    def test_append_outside_the_frozen_scale_repacks_the_store(self, tmp_path):
        path = tmp_path / "sst.zarr"
        stored = self._store_scaled_for(path, 10.0, 20.0)

        # Well outside: the stored scale cannot reach 200.
        far = self._far()
        write_append_zarr("sst", far, path)

        back = xr.open_zarr(path, consolidated=False)
        try:
            assert back.sizes["time"] == 10
            assert back.sst.encoding["dtype"] == np.dtype("int16")
            head = back.sst.sel(time=stored.time).values
            tail = back.sst.sel(time=far.time).values
            assert np.abs(head - stored.sst.values).max() < 0.01, (
                "stored values did not survive the repack"
            )
            assert np.abs(tail - 200.0).max() < 0.01, "the append wrapped"
        finally:
            back.close()

    def test_repacked_scale_spans_the_data_not_the_old_window(self, tmp_path):
        """
        Re-deriving from the old *representable* window would widen the scale
        by the headroom again on every repack, losing resolution each time.
        """
        from h2mare.storage.xarray_helpers import _INT16_HEADROOM

        path = tmp_path / "sst.zarr"
        self._store_scaled_for(path, 10.0, 20.0)
        write_append_zarr("sst", self._far(), path)

        back = xr.open_zarr(path, consolidated=False)
        try:
            enc = back.sst.encoding
            assert enc["scale_factor"] == pytest.approx(
                (200.0 - 10.0) * _INT16_HEADROOM / 65000.0
            )
            assert enc["add_offset"] == pytest.approx((200.0 + 10.0) / 2.0)
        finally:
            back.close()

    def test_variable_that_fits_keeps_its_scale(self, tmp_path):
        """Only the overflowing variable is rescaled; the rest lose nothing."""
        path = tmp_path / "sst.zarr"
        ds = _make_ds("2020-01-01", 5)
        ds["sst"] = ds["sst"] * 0 + np.linspace(10.0, 20.0, ds["sst"].size).reshape(
            ds["sst"].shape
        )
        ds["t2m"] = ds["sst"]
        encoding = self._tight_encoding(10.0, 20.0)
        encoding["t2m"] = dict(encoding["sst"])
        write_append_zarr("sst", ds, path, encoding=encoding)

        far = self._far()
        far["t2m"] = far["sst"] * 0 + 15.0
        write_append_zarr("sst", far, path)

        back = xr.open_zarr(path, consolidated=False)
        try:
            assert back.t2m.encoding["scale_factor"] == pytest.approx(10.0 / 65000.0)
            assert back.sst.encoding["scale_factor"] != pytest.approx(10.0 / 65000.0)
        finally:
            back.close()

    def test_repack_leaves_no_temp_or_backup(self, tmp_path):
        path = tmp_path / "sst.zarr"
        self._store_scaled_for(path, 10.0, 20.0)
        write_append_zarr("sst", self._far(), path)

        assert not list(path.parent.glob("*.tmp")), "a temp store was left behind"
        assert not list(path.parent.glob("*.bak")), "a backup was left behind"

    def test_unrepackable_append_is_refused(self, tmp_path):
        path = tmp_path / "sst.zarr"
        self._unrepackable_store(path)

        with pytest.raises(ValueError, match="cannot represent the incoming data"):
            write_append_zarr("sst", self._far(), path)

    def test_refusal_names_the_variable_and_the_incoming_range(self, tmp_path):
        path = tmp_path / "sst.zarr"
        self._unrepackable_store(path)

        with pytest.raises(ValueError) as excinfo:
            write_append_zarr("sst", self._far(), path)
        msg = str(excinfo.value)
        assert "sst" in msg and "int16" in msg
        assert "200" in msg, "the offending incoming range is not reported"
        assert "Re-convert" in msg, "the message gives no way forward"

    def test_the_store_is_untouched_when_the_append_is_refused(self, tmp_path):
        """The refusal comes before any write, so it leaves no damage."""
        path = tmp_path / "sst.zarr"
        self._unrepackable_store(path)
        before = xr.open_zarr(path, consolidated=False)
        n_before = before.sizes["time"]
        before.close()

        with pytest.raises(ValueError):
            write_append_zarr("sst", self._far(), path)

        after = xr.open_zarr(path, consolidated=False)
        try:
            assert after.sizes["time"] == n_before
            assert not list(path.parent.glob("*.tmp")), "a temp store was left behind"
            assert not list(path.parent.glob("*.bak")), "a backup was left behind"
        finally:
            after.close()

    def test_append_inside_the_frozen_scale_is_allowed(self, tmp_path):
        path = tmp_path / "sst.zarr"
        self._store_scaled_for(path, 10.0, 20.0)

        near = _make_ds("2020-01-06", 5, seed=1)
        near["sst"] = near["sst"] * 0 + 15.0
        write_append_zarr("sst", near, path)

        back = xr.open_zarr(path, consolidated=False)
        try:
            assert back.sizes["time"] == 10
            tail = back.sst.sel(time=near.time).values
            assert np.abs(tail - 15.0).max() < 0.01
        finally:
            back.close()

    def test_float32_store_is_not_checked(self, tmp_path):
        """An unpacked store has no frozen scale, so the guard is a no-op."""
        path = tmp_path / "sst.zarr"
        write_append_zarr("sst", _make_ds("2020-01-01", 5), path)

        wild = _make_ds("2020-01-06", 5, seed=1)
        wild["sst"] = wild["sst"] * 0 + 1e6
        write_append_zarr("sst", wild, path)

        back = xr.open_zarr(path, consolidated=False)
        try:
            assert back.sizes["time"] == 10
            assert float(back.sst.max()) == pytest.approx(1e6)
        finally:
            back.close()

    def test_all_nan_increment_is_not_refused(self, tmp_path):
        """Nothing to encode means nothing to overflow."""
        path = tmp_path / "sst.zarr"
        self._store_scaled_for(path, 10.0, 20.0)

        empty = _make_ds("2020-01-06", 5, seed=1)
        empty["sst"] = empty["sst"] * np.nan
        write_append_zarr("sst", empty, path)

        back = xr.open_zarr(path, consolidated=False)
        try:
            assert back.sizes["time"] == 10
        finally:
            back.close()


class TestInt16Headroom:
    """
    The scale is widened past the observed range because it has to serve every
    later append, not just the batch it was derived from.
    """

    def test_representable_range_extends_beyond_the_observed_range(self):
        from h2mare.storage.xarray_helpers import int16_encoding

        ds = _make_ds("2020-01-01", 5)
        lo = float(ds.sst.min())
        hi = float(ds.sst.max())

        enc = int16_encoding(ds)["sst"]
        scale = enc["scale_factor"]
        offset = enc["add_offset"]
        repr_lo = offset + np.iinfo(np.int16).min * scale
        repr_hi = offset + np.iinfo(np.int16).max * scale

        assert repr_lo < lo and repr_hi > hi
        # Comfortably more than the ~0.4% the un-widened scale left.
        margin = min(lo - repr_lo, repr_hi - hi) / (hi - lo)
        assert margin > 0.25, f"headroom is only {margin:.1%} of the observed range"

    def test_quantisation_is_still_far_inside_source_precision(self):
        from h2mare.storage.xarray_helpers import int16_encoding

        ds = _make_ds("2020-01-01", 10)
        enc = int16_encoding(ds)["sst"]
        rng = float(ds.sst.max() - ds.sst.min())
        assert enc["scale_factor"] / rng < 1e-4


class TestInt16EncodingReadsSourceOnce:
    """
    Finding the ranges must cost one pass over the source, not one per
    reduction. A min and a max computed separately each re-decode the whole
    input — on a year of hourly ERA5 that is tens of GB of GRIB read again,
    and it happens before the write so nothing in the log explains the wait.
    """

    N_CHUNKS = 4

    @classmethod
    def _counting_ds(cls, n_vars: int) -> tuple[xr.Dataset, list[int]]:
        """Dataset whose every chunk read bumps a counter, one array per var."""
        import dask.array as darr

        reads = [0]
        chunk = 5

        def _load(block_id=None):
            reads[0] += 1
            rng = np.random.default_rng(sum(block_id or (0,)))
            return rng.random((chunk, 2, 2))

        def _array():
            return darr.map_blocks(
                _load,
                chunks=((chunk,) * cls.N_CHUNKS, (2,), (2,)),
                dtype="float64",
            )

        times = pd.date_range("2020-01-01", periods=chunk * cls.N_CHUNKS, freq="D")
        ds = xr.Dataset(
            {f"v{i}": (["time", "lat", "lon"], _array()) for i in range(n_vars)},
            coords={"time": times, "lat": [30.0, 30.25], "lon": [-10.0, -9.75]},
        )
        # dask calls the loader while building the graph to infer meta; only
        # reads from here on are the pass this test is about.
        reads[0] = 0
        return ds, reads

    def test_one_variable_is_read_once_not_twice(self):
        from h2mare.storage.xarray_helpers import int16_encoding

        ds, reads = self._counting_ds(n_vars=1)
        int16_encoding(ds)

        assert reads[0] == self.N_CHUNKS, (
            f"expected one pass over {self.N_CHUNKS} chunks, got {reads[0]} reads "
            f"— min and max are not sharing a graph"
        )

    def test_cost_stays_one_pass_with_several_variables(self):
        from h2mare.storage.xarray_helpers import int16_encoding

        n_vars = 3
        ds, reads = self._counting_ds(n_vars=n_vars)
        int16_encoding(ds)

        assert reads[0] == n_vars * self.N_CHUNKS, (
            f"expected one pass over {n_vars} variables, got {reads[0]} reads"
        )

    def test_bounds_are_still_the_real_min_and_max(self):
        """Sharing the graph must not change the numbers it produces."""
        from h2mare.storage.xarray_helpers import _INT16_HEADROOM, int16_encoding

        ds = _make_ds("2020-01-01", 10)
        enc = int16_encoding(ds)

        # Recovering lo/hi from the encoding is what pins the bounds: the
        # headroom scales the span and the offset stays the midpoint, so both
        # ends invert back to the dataset's true min and max.
        lo, hi = float(ds.sst.min()), float(ds.sst.max())
        assert enc["sst"]["scale_factor"] == pytest.approx(
            (hi - lo) * _INT16_HEADROOM / 65000.0
        )
        assert enc["sst"]["add_offset"] == pytest.approx((hi + lo) / 2.0)


# ---------------------------------------------------------------------------
# A store that carries both _FillValue and missing_value
#
# chl inherited missing_value: -999.0 from the CMEMS ocean-colour product, and
# it was never true of what we stored: the array's fill is NaN. Reading a Zarr
# moves both into .encoding, and xarray refuses to write them back when they
# disagree — so the store could not be rewritten from itself, which is what a
# front recompute, a rechunk or any overlapping append does.
# ---------------------------------------------------------------------------


class TestConflictingMissingValue:
    def _stored(self, tmp_path):
        """A store whose data variable declares both, as chl's does."""
        path = tmp_path / "chl.zarr"
        ds = _make_ds(n_days=4).rename_vars({"sst": "chl"})
        ds["chl"].attrs["missing_value"] = -999.0
        ds.to_zarr(path)
        return path

    def test_an_append_that_keeps_the_stored_head_can_rewrite_it(self, tmp_path):
        """The retained head carries the store's encoding into the concat."""
        path = self._stored(tmp_path)
        later = _make_ds(start="2020-01-03", n_days=4, seed=1).rename_vars(
            {"sst": "chl"}
        )

        write_append_zarr("chl", later, path)

        with xr.open_zarr(path, consolidated=False) as out:
            assert out.sizes["time"] == 6
            assert "missing_value" not in out["chl"].encoding

    def test_a_new_variable_can_be_added_to_it(self, tmp_path):
        """The layer a front recompute writes: one variable, the rest re-read
        from the store and merged back — which is where the conflict surfaced."""
        path = self._stored(tmp_path)
        layer = _make_ds(n_days=4, seed=2).rename_vars({"sst": "chl_fdist"})

        write_append_zarr("chl", layer, path)

        with xr.open_zarr(path, consolidated=False) as out:
            assert {"chl", "chl_fdist"} <= set(map(str, out.data_vars))

    def test_an_agreeing_missing_value_is_left_alone(self, tmp_path):
        """Nothing is wrong when the two say the same thing, so nothing is dropped."""
        ds = _make_ds(n_days=3).rename_vars({"sst": "chl"})
        ds["chl"].encoding.update({"_FillValue": -999.0, "missing_value": -999.0})

        drop_conflicting_missing_value(ds)

        assert ds["chl"].encoding["missing_value"] == -999.0
