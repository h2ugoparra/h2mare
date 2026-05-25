"""Tests for processing/core/fronts.py — pure functions and FrontProcessor helpers."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.processing.core.fronts import (
    BOA_aplication,
    FrontProcessor,
    boa,
    create_base_grid,
    filt3,
    filt5,
)


# ---------------------------------------------------------------------------
# filt5
# ---------------------------------------------------------------------------

class TestFilt5:

    def test_output_shape_matches_input(self):
        arr = np.random.rand(8, 10)
        assert filt5(arr).shape == (8, 10)

    def test_returns_int8(self):
        arr = np.random.rand(5, 5)
        assert filt5(arr).dtype == np.int8

    def test_peak_pixel_is_flagged(self):
        arr = np.zeros((7, 7))
        arr[3, 3] = 10.0
        result = filt5(arr)
        assert result[3, 3] == 1

    def test_uniform_array_center_flagged(self):
        # Center pixel has a full 5×5 window of equal values → both max and min → flagged
        arr = np.ones((5, 5)) * 5.0
        result = filt5(arr)
        assert result[2, 2] == 1


# ---------------------------------------------------------------------------
# filt3
# ---------------------------------------------------------------------------

class TestFilt3:

    def test_output_shape_matches_input(self):
        arr = np.random.rand(6, 6)
        grid5 = filt5(arr)
        assert filt3(arr, grid5).shape == (6, 6)

    def test_filt5_flagged_pixels_kept_unchanged(self):
        arr = np.zeros((7, 7))
        arr[3, 3] = 100.0
        grid5 = filt5(arr)
        out = filt3(arr, grid5)
        # Peak is a local extremum → flagged by filt5 → grid5==1 → kept as-is
        assert out[3, 3] == pytest.approx(arr[3, 3])

    def test_returns_float_array(self):
        arr = np.random.rand(6, 6)
        grid5 = filt5(arr)
        assert filt3(arr, grid5).dtype.kind == "f"


# ---------------------------------------------------------------------------
# boa
# ---------------------------------------------------------------------------

class TestBOA:

    def _step_field(self):
        lat = np.linspace(30.0, 35.0, 8)
        lon = np.linspace(-10.0, -5.0, 10)
        ingrid = np.ones((8, 10)) * 20.0
        ingrid[:, 5:] = 10.0
        return lat, lon, ingrid

    def test_step_function_returns_front_pixels(self):
        lat, lon, ingrid = self._step_field()
        result = boa(lon, lat, ingrid, threshold=0.1)
        assert result.ndim == 2
        assert result.shape[1] == 2
        assert result.shape[0] > 0

    def test_front_coords_within_input_bounds(self):
        lat, lon, ingrid = self._step_field()
        result = boa(lon, lat, ingrid, threshold=0.1)
        assert float(result[:, 0].min()) >= lat.min()
        assert float(result[:, 0].max()) <= lat.max()
        assert float(result[:, 1].min()) >= lon.min()
        assert float(result[:, 1].max()) <= lon.max()

    def test_very_high_threshold_no_fronts(self):
        lat, lon, ingrid = self._step_field()
        result = boa(lon, lat, ingrid, threshold=1e9)
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# BOA_aplication
# ---------------------------------------------------------------------------

class TestBOAApplication:

    def test_xarray_wrapper_returns_same_as_boa(self):
        lat = np.linspace(30.0, 35.0, 8)
        lon = np.linspace(-10.0, -5.0, 10)
        ingrid = np.ones((8, 10)) * 20.0
        ingrid[:, 5:] = 10.0
        da = xr.DataArray(
            ingrid,
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
        result_xr = BOA_aplication(da, threshold=0.1)
        result_np = boa(lon, lat, ingrid, threshold=0.1)
        np.testing.assert_array_equal(result_xr, result_np)


# ---------------------------------------------------------------------------
# create_base_grid
# ---------------------------------------------------------------------------

class TestCreateBaseGrid:

    def test_output_shapes(self):
        lat = np.array([30.0, 35.0, 40.0])
        lon = np.array([-10.0, -5.0, 0.0])
        all_ocean = np.zeros((3, 3), dtype=bool)
        with patch("h2mare.processing.core.fronts.globe.is_land", return_value=all_ocean):
            latlon_arr, sea_mask = create_base_grid(lat, lon)
        assert sea_mask.shape == (3, 3)
        assert latlon_arr.ndim == 2
        assert latlon_arr.shape[1] == 2

    def test_all_ocean_includes_all_pixels(self):
        lat = np.array([30.0, 35.0])
        lon = np.array([-10.0, -5.0])
        all_ocean = np.zeros((2, 2), dtype=bool)
        with patch("h2mare.processing.core.fronts.globe.is_land", return_value=all_ocean):
            latlon_arr, sea_mask = create_base_grid(lat, lon)
        assert latlon_arr.shape[0] == 4

    def test_all_land_excludes_all_pixels(self):
        lat = np.array([30.0, 35.0])
        lon = np.array([-10.0, -5.0])
        all_land = np.ones((2, 2), dtype=bool)
        with patch("h2mare.processing.core.fronts.globe.is_land", return_value=all_land):
            latlon_arr, sea_mask = create_base_grid(lat, lon)
        assert latlon_arr.shape[0] == 0


# ---------------------------------------------------------------------------
# FrontProcessor.__init__
# ---------------------------------------------------------------------------

class TestFrontProcessorInit:

    def test_sst_is_valid(self):
        fp = FrontProcessor("sst")
        assert fp.var_key == "sst"

    def test_chl_is_valid(self):
        fp = FrontProcessor("chl")
        assert fp.var_key == "chl"

    def test_invalid_key_raises_value_error(self):
        with pytest.raises(ValueError, match="var_key"):
            FrontProcessor("ssh")

    def test_empty_key_raises_value_error(self):
        with pytest.raises(ValueError):
            FrontProcessor("")


# ---------------------------------------------------------------------------
# FrontProcessor._process_daily
# ---------------------------------------------------------------------------

class TestProcessDaily:

    def test_returns_dataarray_with_correct_name_and_time(self):
        fp = FrontProcessor("sst")
        lat = np.array([30.0, 35.0])
        lon = np.array([-10.0, -5.0])
        date = pd.Timestamp("2020-01-15")

        da = xr.DataArray(
            np.random.rand(1, 2, 2).astype("float32"),
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-01-15", periods=1, freq="D"),
                "lat": lat,
                "lon": lon,
            },
        )

        sea_mask = np.ones((2, 2), dtype=bool)
        latlon_arr = np.array([[30.0, -10.0], [30.0, -5.0], [35.0, -10.0], [35.0, -5.0]])
        fake_fronts = np.array([[32.0, -7.5]])

        with (
            patch("h2mare.processing.core.fronts.create_base_grid", return_value=(latlon_arr, sea_mask)),
            patch("h2mare.processing.core.fronts.BOA_aplication", return_value=fake_fronts),
        ):
            result = fp._process_daily(da, date)

        assert isinstance(result, xr.DataArray)
        assert result.name == "sst_fdist"
        assert "time" in result.dims
        assert result.sel(time=date).shape == (2, 2)

    def test_sea_pixels_have_finite_distances(self):
        fp = FrontProcessor("sst")
        lat = np.array([30.0, 35.0])
        lon = np.array([-10.0, -5.0])
        date = pd.Timestamp("2020-06-01")

        da = xr.DataArray(
            np.random.rand(1, 2, 2).astype("float32"),
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range("2020-06-01", periods=1, freq="D"),
                "lat": lat,
                "lon": lon,
            },
        )

        sea_mask = np.ones((2, 2), dtype=bool)
        latlon_arr = np.array([[30.0, -10.0], [30.0, -5.0], [35.0, -10.0], [35.0, -5.0]])
        fake_fronts = np.array([[32.0, -7.5]])

        with (
            patch("h2mare.processing.core.fronts.create_base_grid", return_value=(latlon_arr, sea_mask)),
            patch("h2mare.processing.core.fronts.BOA_aplication", return_value=fake_fronts),
        ):
            result = fp._process_daily(da, date)

        values = result.sel(time=date).values
        assert np.all(np.isfinite(values))
