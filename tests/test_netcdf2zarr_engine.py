"""Tests for the config-free convert_netcdf_to_zarr engine function.

The whole point of this function is to convert arbitrary NetCDF/GRIB files
*without* a configured var_key, so these tests deliberately use names that are
not present in config.yaml's variables.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.format_converters.netcdf2zarr import convert_netcdf_to_zarr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ds(start: str = "2020-01-01", n_days: int = 5, seed: int = 0) -> xr.Dataset:
    """Varied (non-constant) data so have_vars_unique_values does not fire."""
    times = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    data = rng.uniform(10.0, 30.0, size=(n_days, 3, 3))
    return xr.Dataset(
        {"adhoc_var": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


def _write_nc(ds: xr.Dataset, path) -> str:
    ds.to_netcdf(path)
    return str(path)


# ---------------------------------------------------------------------------
# Core conversion — no config / no var_key
# ---------------------------------------------------------------------------


def test_single_file_roundtrips_without_config(tmp_path):
    """A name that is NOT in config.variables must still convert cleanly."""
    src = tmp_path / "input.nc"
    _write_nc(_make_ds(), src)
    out = tmp_path / "out.zarr"

    result = convert_netcdf_to_zarr(src, out, name="adhoc")

    assert result == out
    assert out.exists()
    ds = xr.open_zarr(out)
    assert "adhoc_var" in ds.data_vars
    assert len(ds.time) == 5
    assert {"time", "lat", "lon"} <= set(ds.dims)
    ds.close()


def test_accepts_str_path(tmp_path):
    src = tmp_path / "input.nc"
    _write_nc(_make_ds(), src)
    out = tmp_path / "out.zarr"

    convert_netcdf_to_zarr(str(src), str(out), name="adhoc")

    assert out.exists()


def test_empty_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_netcdf_to_zarr([], tmp_path / "out.zarr")


# ---------------------------------------------------------------------------
# Processor hook
# ---------------------------------------------------------------------------


def test_processor_hook_is_applied(tmp_path):
    src = tmp_path / "input.nc"
    _write_nc(_make_ds(), src)
    out = tmp_path / "out.zarr"

    def processor(ds: xr.Dataset) -> xr.Dataset:
        return ds.assign(doubled=ds["adhoc_var"] * 2)

    convert_netcdf_to_zarr(src, out, name="adhoc", processor=processor)

    ds = xr.open_zarr(out)
    assert "doubled" in ds.data_vars
    np.testing.assert_allclose(
        ds["doubled"].values, ds["adhoc_var"].values * 2, rtol=1e-5
    )
    ds.close()


# ---------------------------------------------------------------------------
# rename_dims toggle
# ---------------------------------------------------------------------------


def test_apply_rename_canonicalizes_dims(tmp_path):
    """longitude/latitude/valid_time should be renamed to lon/lat/time."""
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"adhoc_var": (["valid_time", "latitude", "longitude"], rng.random((3, 3, 3)))},
        coords={
            "valid_time": times,
            "latitude": [30.0, 35.0, 40.0],
            "longitude": [-10.0, -5.0, 0.0],
        },
    )
    src = tmp_path / "input.nc"
    _write_nc(ds, src)
    out = tmp_path / "out.zarr"

    convert_netcdf_to_zarr(src, out, name="adhoc")

    res = xr.open_zarr(out)
    assert {"time", "lat", "lon"} <= set(res.dims)
    assert "longitude" not in res.dims
    res.close()


# ---------------------------------------------------------------------------
# Append semantics (reuses write_append_zarr)
# ---------------------------------------------------------------------------


def test_second_call_extends_existing_store(tmp_path):
    out = tmp_path / "out.zarr"

    first = tmp_path / "first.nc"
    _write_nc(_make_ds("2020-01-01", 5, seed=1), first)
    convert_netcdf_to_zarr(first, out, name="adhoc")

    second = tmp_path / "second.nc"
    _write_nc(_make_ds("2020-01-06", 5, seed=2), second)
    convert_netcdf_to_zarr(second, out, name="adhoc")

    ds = xr.open_zarr(out)
    assert len(ds.time) == 10
    assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("2020-01-01")
    assert pd.Timestamp(ds.time.values[-1]) == pd.Timestamp("2020-01-10")
    ds.close()


# ---------------------------------------------------------------------------
# Multi-file (list) input
# ---------------------------------------------------------------------------


def test_list_of_files_combined_by_coords(tmp_path):
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _write_nc(_make_ds("2020-01-01", 3, seed=1), a)
    _write_nc(_make_ds("2020-01-04", 3, seed=2), b)
    out = tmp_path / "out.zarr"

    convert_netcdf_to_zarr([b, a], out, name="adhoc")  # unsorted on purpose

    ds = xr.open_zarr(out)
    assert len(ds.time) == 6
    assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("2020-01-01")
    ds.close()
