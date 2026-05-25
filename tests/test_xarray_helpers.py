"""Tests for storage/xarray_helpers.py."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.storage.xarray_helpers import (
    chunk_dataset,
    convert360_180,
    get_dataset_encoding,
    have_vars_unique_values,
    rename_dims,
    unified_time_chunk,
    xr_float64_to_float32,
)


def _make_ds(n_time=10, n_lat=4, n_lon=4, dtype=np.float32):
    times = pd.date_range("2020-01-01", periods=n_time, freq="D")
    data = np.random.rand(n_time, n_lat, n_lon).astype(dtype)
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": np.linspace(30, 40, n_lat),
            "lon": np.linspace(-10, 0, n_lon),
        },
    )


class TestGetDatasetEncoding:
    def test_returns_encoding_for_each_var(self):
        ds = _make_ds()
        enc = get_dataset_encoding(ds)
        assert "sst" in enc
        assert "chunks" in enc["sst"]

    def test_chunk_tuple_length_matches_dims(self):
        ds = _make_ds()
        enc = get_dataset_encoding(ds)
        assert len(enc["sst"]["chunks"]) == 3  # time, lat, lon


class TestUnifiedTimeChunk:
    def test_returns_positive_int(self):
        ds = _make_ds(n_time=365)
        chunk = unified_time_chunk(ds)
        assert isinstance(chunk, int)
        assert chunk >= 1

    def test_no_time_vars_raises(self):
        ds = xr.Dataset(
            {"sst": (["lat", "lon"], np.ones((4, 4)))},
            coords={"lat": [30.0, 31.0, 32.0, 33.0], "lon": [-10.0, -9.0, -8.0, -7.0]},
        )
        with pytest.raises(ValueError, match="time"):
            unified_time_chunk(ds)


class TestHaveVarsUniqueValues:
    def test_nonexistent_path_returns_false(self, tmp_path):
        bad_path = tmp_path / "nonexistent.zarr"
        assert have_vars_unique_values(bad_path) is False

    def test_dataset_with_varied_values_returns_false(self):
        ds = _make_ds(n_time=5)
        assert have_vars_unique_values(ds) is False

    def test_dataset_with_constant_last_slice_returns_true(self):
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        data = np.random.rand(3, 4, 4).astype(np.float32)
        data[-1, :, :] = 5.0  # last time step is constant
        ds = xr.Dataset(
            {"sst": (["time", "lat", "lon"], data)},
            coords={"time": times, "lat": range(4), "lon": range(4)},
        )
        assert have_vars_unique_values(ds) is True


class TestConvert360To180:
    def test_converts_0_360_to_minus180_180(self):
        ds = xr.Dataset(
            {"sst": (["lat", "lon"], np.ones((3, 4)))},
            coords={"lat": [0.0, 1.0, 2.0], "lon": [0.0, 90.0, 180.0, 270.0]},
        )
        result = convert360_180(ds)
        assert float(result["lon"].min()) >= -180
        assert float(result["lon"].max()) <= 180

    def test_already_negative_lon_unchanged(self):
        ds = xr.Dataset(
            {"sst": (["lat", "lon"], np.ones((2, 3)))},
            coords={"lat": [0.0, 1.0], "lon": [-10.0, 0.0, 10.0]},
        )
        result = convert360_180(ds)
        assert list(result["lon"].values) == [-10.0, 0.0, 10.0]


class TestChunkDataset:
    def test_2d_spatial_dims_stay_full_size(self):
        """lat and lon are always kept at full size."""
        ds = _make_ds(n_time=10, n_lat=50, n_lon=60)
        result = chunk_dataset(ds, target_mb=32)
        assert result.chunks["lat"] == (50,)
        assert result.chunks["lon"] == (60,)

    def test_converts_float64_to_float32(self):
        """float64 variables are downcast to float32."""
        ds = _make_ds(dtype=np.float64)
        result = chunk_dataset(ds)
        assert result["sst"].dtype == np.float32

    def test_depth_chunked_to_1_when_payload_exceeds_target(self):
        """depth is chunked to 1 when per-step payload exceeds target_mb."""
        times = pd.date_range("2020-01-01", periods=10, freq="D")
        # 5 × 300 × 300 × 4 bytes ≈ 1.7 MB > target_mb=1 → depth must chunk to 1
        n_depth, n_lat, n_lon = 5, 300, 300
        data = np.ones((10, n_depth, n_lat, n_lon), dtype=np.float32)
        ds = xr.Dataset(
            {"thetao": (["time", "depth", "lat", "lon"], data)},
            coords={
                "time": times,
                "depth": np.arange(n_depth, dtype=np.float32),
                "lat": np.linspace(0, 70, n_lat),
                "lon": np.linspace(-80, 10, n_lon),
            },
        )
        result = chunk_dataset(ds, target_mb=1)
        assert result.chunks["depth"] == (1,) * n_depth

    def test_depth_not_chunked_when_payload_under_target(self):
        """depth stays at full size when the per-step payload is under target_mb."""
        times = pd.date_range("2020-01-01", periods=10, freq="D")
        # 3 × 10 × 10 × 4 bytes = 1 200 bytes ≪ 32 MB → depth must NOT chunk
        n_depth = 3
        data = np.ones((10, n_depth, 10, 10), dtype=np.float32)
        ds = xr.Dataset(
            {"o2": (["time", "depth", "lat", "lon"], data)},
            coords={
                "time": times,
                "depth": np.arange(n_depth, dtype=np.float32),
                "lat": np.arange(10, dtype=np.float32),
                "lon": np.arange(10, dtype=np.float32),
            },
        )
        result = chunk_dataset(ds, target_mb=32)
        assert result.chunks["depth"] == (n_depth,)

    def test_time_chunk_recomputed_after_depth_reduction(self):
        """After depth is chunked to 1, time chunk should be larger than 1."""
        times = pd.date_range("2020-01-01", periods=30, freq="D")
        n_depth, n_lat, n_lon = 5, 300, 300
        data = np.ones((30, n_depth, n_lat, n_lon), dtype=np.float32)
        ds = xr.Dataset(
            {"thetao": (["time", "depth", "lat", "lon"], data)},
            coords={
                "time": times,
                "depth": np.arange(n_depth, dtype=np.float32),
                "lat": np.linspace(0, 70, n_lat),
                "lon": np.linspace(-80, 10, n_lon),
            },
        )
        result = chunk_dataset(ds, target_mb=1)
        # With depth=1, lat=300, lon=300: 1*300*300*4 = 360 000 bytes ≈ 0.34 MB
        # time_chunk = floor(1 MB / 0.34 MB) = 2 → must be > 1
        assert result.chunks["time"][0] > 1


class TestRenameDims:
    def test_renames_longitude_latitude(self):
        ds = xr.Dataset(
            {"sst": (["latitude", "longitude"], np.ones((3, 3)))},
            coords={
                "latitude": [30.0, 35.0, 40.0],
                "longitude": [-10.0, -5.0, 0.0],
            },
        )
        result = rename_dims(ds)
        assert "lat" in result.dims
        assert "lon" in result.dims

    def test_renames_valid_time(self):
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        ds = xr.Dataset(
            {"sst": (["valid_time", "lat", "lon"], np.ones((3, 2, 2)))},
            coords={"valid_time": times, "lat": [30.0, 31.0], "lon": [-10.0, -9.0]},
        )
        result = rename_dims(ds)
        assert "time" in result.dims

    def test_no_rename_needed(self):
        ds = _make_ds()
        result = rename_dims(ds)
        assert "time" in result.dims
        assert "lat" in result.dims
        assert "lon" in result.dims
