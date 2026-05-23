"""Tests for write_append_zarr and atomic swap behaviour in storage.py."""
import shutil
from pathlib import Path
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
        """If renaming tmp → final fails, the original is restored from backup."""
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
                _append_data("sst", _make_ds("2020-01-06", 5), path)

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
                _append_data("sst", _make_ds("2020-01-06", 5), path)

        assert not path.with_name(path.name + ".bak").exists()


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
        _make_ds("2020-01-01", 5).to_zarr(path)           # writes 'sst'
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
            coords={"time": times, "lat": [30.0, 35.0, 40.0], "lon": [-10.0, -5.0, 0.0]},
        )
        _append_data("h2ds", ds_new, path)

        ds = xr.open_zarr(path, consolidated=False)
        assert "sst" in ds.data_vars
        assert "thetao_100" in ds.data_vars
        assert "thetao_500" in ds.data_vars
        ds.close()
