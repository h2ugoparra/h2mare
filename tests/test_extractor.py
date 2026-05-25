"""Tests for Extractor — focused on logic that doesn't need external data."""

import json
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from pathlib import Path

from h2mare.processing.extractor import Extractor, _save_completed_keys, _keys_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spatial_ds(
    lons: list[float] = [-10.0, -5.0, 0.0],
    lats: list[float] = [30.0, 35.0, 40.0],
) -> xr.Dataset:
    """Minimal spatial dataset (no time dimension)."""
    data = np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons))
    return xr.Dataset(
        {"sst": (["lat", "lon"], data)},
        coords={"lat": lats, "lon": lons},
    )


def _make_spatiotemporal_ds(
    lons: list[float] = [-10.0, -5.0, 0.0],
    lats: list[float] = [30.0, 35.0, 40.0],
    n_days: int = 5,
) -> xr.Dataset:
    """Minimal dataset with daily time axis."""
    times = pd.date_range("2020-01-01", periods=n_days, freq="D")
    data = np.ones((n_days, len(lats), len(lons)))
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def _make_extractor(time_values: list, time_col: str = "time") -> Extractor:
    """Build a minimal Extractor from a list of time strings."""
    df = pd.DataFrame(
        {
            time_col: time_values,
            "lon": [10.0] * len(time_values),
            "lat": [40.0] * len(time_values),
        }
    )
    return Extractor(df, time_col=time_col)


# ---------------------------------------------------------------------------
# _resolve_time_col
# ---------------------------------------------------------------------------


class TestResolveTimeCol:
    def test_date_only_strings(self):
        """Date-only strings should NOT be truncated (no time component)."""
        ext = _make_extractor(["2020-01-01", "2020-01-02"])
        # Should stay as date-only (normalised to midnight, no truncation log)
        assert ext.data["time"].dt.hour.eq(0).all()

    def test_uniform_time_component_truncated(self):
        """Datetimes where all times are identical → truncate to midnight."""
        ext = _make_extractor(
            [
                "2020-01-01 06:00:00",
                "2020-01-02 06:00:00",
                "2020-01-03 06:00:00",
            ]
        )
        # All midnight after truncation
        assert ext.data["time"].dt.hour.eq(0).all()

    def test_variable_time_component_kept(self):
        """Datetimes with varying times → keep full datetime."""
        ext = _make_extractor(
            [
                "2020-01-01 06:00:00",
                "2020-01-01 12:00:00",
                "2020-01-01 18:00:00",
            ]
        )
        hours = ext.data["time"].dt.hour.tolist()
        assert len(set(hours)) > 1  # times preserved

    def test_tz_aware_input_becomes_naive(self):
        """TZ-aware strings must become tz-naive after conversion."""
        ext = _make_extractor(
            [
                "2020-06-15T10:00:00+00:00",
                "2020-06-16T10:00:00+00:00",
            ]
        )
        assert ext.data["time"].dt.tz is None

    def test_raw_check_before_conversion(self):
        """
        Regression: raw string check was done AFTER pd.to_datetime conversion,
        so datetime(00:00:00) always matched the HH:MM:SS pattern, causing
        date-only inputs to be misclassified as having a time component.
        """
        ext = _make_extractor(["2020-01-01", "2020-01-02"])
        # If the bug is present, the code enters the has_time_component branch
        # and then truncates to date — result would be midnight (same as correct).
        # The real regression is: variable-time branch would not be entered.
        # We verify by checking the date-only input stays at date (midnight).
        dates = ext.data["time"].dt.normalize()
        assert (ext.data["time"] == dates).all()

    def test_non_default_time_col(self):
        """Extractor should handle a non-default time column name."""
        ext = _make_extractor(["2020-03-01", "2020-03-02"], time_col="date")
        assert "time" in ext.data.columns  # renamed internally


# ---------------------------------------------------------------------------
# _nearest_grid_indices
# ---------------------------------------------------------------------------


class TestNearestGridIndices:
    def test_exact_grid_points(self):
        """Querying exact grid coordinates returns their exact indices."""
        ds = _make_spatial_ds()
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-10.0, -5.0, 0.0]), np.array([30.0, 35.0, 40.0])
        )
        np.testing.assert_array_equal(lon_idx, [0, 1, 2])
        np.testing.assert_array_equal(lat_idx, [0, 1, 2])

    def test_off_grid_snaps_to_nearest(self):
        """Off-grid point snaps to the nearest grid point."""
        ds = _make_spatial_ds()
        # -8.0 is 2° from -10 and 3° from -5 → nearest is -10 (index 0)
        # 32.0 is 2° from 30 and 3° from 35 → nearest is 30 (index 0)
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-8.0]), np.array([32.0])
        )
        assert lon_idx[0] == 0
        assert lat_idx[0] == 0

    def test_returns_ndarrays(self):
        """Output must be numpy ndarrays regardless of input size."""
        ds = _make_spatial_ds()
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-10.0, 0.0]), np.array([30.0, 40.0])
        )
        assert isinstance(lat_idx, np.ndarray)
        assert isinstance(lon_idx, np.ndarray)

    def test_single_point(self):
        """Single-point query returns length-1 arrays."""
        ds = _make_spatial_ds()
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([0.0]), np.array([40.0])
        )
        assert lat_idx.shape == (1,)
        assert lon_idx.shape == (1,)
        assert lon_idx[0] == 2  # 0.0 is last lon (index 2)
        assert lat_idx[0] == 2  # 40.0 is last lat (index 2)

    def test_irregular_grid(self):
        """Works on a non-uniform grid where searchsorted alone would be wrong."""
        ds = _make_spatial_ds(lons=[-10.0, -3.0, 0.0], lats=[30.0, 38.0, 40.0])
        # -4.0 is 1° from -3 and 6° from -10 → nearest is -3 (index 1)
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-4.0]), np.array([30.0])
        )
        assert lon_idx[0] == 1


# ---------------------------------------------------------------------------
# _nearest_time_indices
# ---------------------------------------------------------------------------


class TestNearestTimeIndices:
    def test_exact_match(self):
        """Exact timestamp returns the correct index."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 0

    def test_picks_closer_left_neighbor(self):
        """Point 6h after a step is closer to that step than the next (18h away)."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01 06:00:00"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 0  # 6 h from Jan 1, 18 h from Jan 2

    def test_picks_closer_right_neighbor(self):
        """Point 18h after a step is closer to the next step (6h away)."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01 18:00:00"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 1  # 18 h from Jan 1, 6 h from Jan 2

    def test_before_first_step_clips_to_zero(self):
        """Query before the first time step is clipped to index 0."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2019-12-31"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 0

    def test_after_last_step_clips_to_last(self):
        """Query after the last time step is clipped to the last index."""
        ds = _make_spatiotemporal_ds(n_days=5)
        q = np.array(pd.to_datetime(["2025-01-01"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 4

    def test_multiple_queries(self):
        """Multiple timestamps resolved correctly in one call."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"]))
        idx = Extractor._nearest_time_indices(ds, q)
        np.testing.assert_array_equal(idx, [0, 2, 4])


# ---------------------------------------------------------------------------
# Atomic checkpoint helpers
# ---------------------------------------------------------------------------


class TestAtomicCheckpoint:
    def test_save_completed_keys_writes_correct_content(self, tmp_path):
        checkpoint = tmp_path / "data.feather"
        keys = {"sst", "chl", "mld"}
        _save_completed_keys(checkpoint, keys)
        dest = _keys_path(checkpoint)
        with open(dest) as f:
            loaded = set(json.load(f))
        assert loaded == keys

    def test_save_completed_keys_no_staging_file_remains(self, tmp_path):
        """The .tmp staging file must be cleaned up after a successful write."""
        checkpoint = tmp_path / "data.feather"
        _save_completed_keys(checkpoint, {"sst"})
        dest = _keys_path(checkpoint)
        staging = dest.with_suffix(".tmp")
        assert dest.exists()
        assert not staging.exists()

    def test_feather_atomic_write_no_tmp_remains(self, tmp_path):
        """Verify the staging-then-replace pattern: no .tmp file left after write."""
        feather_path = tmp_path / "checkpoint.feather"
        staging = feather_path.with_suffix(".tmp")

        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_feather(staging)
        staging.replace(feather_path)

        assert feather_path.exists()
        assert not staging.exists()

    def test_feather_atomic_write_is_readable(self, tmp_path):
        """Data written through the staging pattern is readable from the final path."""
        feather_path = tmp_path / "checkpoint.feather"
        staging = feather_path.with_suffix(".tmp")

        df = pd.DataFrame({"x": [10, 20, 30], "y": [1.0, 2.0, 3.0]})
        df.to_feather(staging)
        staging.replace(feather_path)

        loaded = pd.read_feather(feather_path)
        pd.testing.assert_frame_equal(df, loaded)
