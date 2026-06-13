"""Tests for the config-free convert_zarr_to_parquet engine function.

Like convert_netcdf_to_zarr, the point is converting an arbitrary Zarr store to
Parquet *without* a configured var_key, so these tests open a store written
straight to disk and never go through ZarrCatalog / app_config.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.format_converters.zarr2parquet import convert_zarr_to_parquet
from h2mare.storage.parquet_indexer import ParquetIndexer
from h2mare.types import TimeResolution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ds(start: str = "2020-01-01", n_days: int = 40, seed: int = 0) -> xr.Dataset:
    """Varied data spanning >1 month so monthly chunking exercises >1 period."""
    times = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    data = rng.uniform(10.0, 30.0, size=(n_days, 2, 2))
    return xr.Dataset(
        {"adhoc_var": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0],
            "lon": [-10.0, -5.0],
        },
    )


def _write_zarr(ds: xr.Dataset, path):
    ds.to_zarr(path)
    return path


# ---------------------------------------------------------------------------
# Core conversion — no config / no var_key
# ---------------------------------------------------------------------------


def test_roundtrips_without_config(tmp_path):
    ds = _make_ds()
    src = tmp_path / "store.zarr"
    _write_zarr(ds, src)
    out = tmp_path / "parquet"

    result = convert_zarr_to_parquet(src, out)

    assert result == out
    df = ParquetIndexer(out).load()
    assert "adhoc_var" in df.columns
    assert {"time", "lat", "lon"} <= set(df.columns)
    assert df.height == 40 * 2 * 2  # n_days * lat * lon


def test_accepts_str_paths(tmp_path):
    src = tmp_path / "store.zarr"
    _write_zarr(_make_ds(n_days=5), src)
    out = tmp_path / "parquet"

    convert_zarr_to_parquet(str(src), str(out))

    assert ParquetIndexer(out).load().height == 5 * 2 * 2


def test_values_preserved(tmp_path):
    ds = _make_ds(n_days=5)
    src = tmp_path / "store.zarr"
    _write_zarr(ds, src)
    out = tmp_path / "parquet"

    convert_zarr_to_parquet(src, out)

    df = ParquetIndexer(out).load().sort(["time", "lat", "lon"])
    expected = ds.to_dataframe().reset_index().sort_values(["time", "lat", "lon"])
    np.testing.assert_allclose(
        df["adhoc_var"].to_numpy(), expected["adhoc_var"].to_numpy(), rtol=1e-5
    )


# ---------------------------------------------------------------------------
# Date window + variable subset
# ---------------------------------------------------------------------------


def test_explicit_date_window_subsets(tmp_path):
    src = tmp_path / "store.zarr"
    _write_zarr(_make_ds(n_days=40), src)
    out = tmp_path / "parquet"

    convert_zarr_to_parquet(src, out, start_date="2020-01-05", end_date="2020-01-09")

    cov = ParquetIndexer(out).get_time_coverage()
    assert pd.Timestamp(cov.start) == pd.Timestamp("2020-01-05")
    assert pd.Timestamp(cov.end) == pd.Timestamp("2020-01-09")


def test_variables_subset(tmp_path):
    ds = _make_ds(n_days=5)
    ds = ds.assign(other=ds["adhoc_var"] * 0.5)
    src = tmp_path / "store.zarr"
    _write_zarr(ds, src)
    out = tmp_path / "parquet"

    convert_zarr_to_parquet(src, out, variables=["adhoc_var"])

    cols = ParquetIndexer(out).load().columns
    assert "adhoc_var" in cols
    assert "other" not in cols


# ---------------------------------------------------------------------------
# Append semantics (reuses ParquetIndexer.add_data)
# ---------------------------------------------------------------------------


def test_second_call_extends_store(tmp_path):
    out = tmp_path / "parquet"

    a = tmp_path / "a.zarr"
    _write_zarr(_make_ds("2020-01-01", 5, seed=1), a)
    convert_zarr_to_parquet(a, out)

    b = tmp_path / "b.zarr"
    _write_zarr(_make_ds("2020-02-01", 5, seed=2), b)
    convert_zarr_to_parquet(b, out)

    cov = ParquetIndexer(out).get_time_coverage()
    assert pd.Timestamp(cov.start) == pd.Timestamp("2020-01-01")
    assert pd.Timestamp(cov.end) == pd.Timestamp("2020-02-05")


# ---------------------------------------------------------------------------
# Depth guard
# ---------------------------------------------------------------------------


def test_depth_dim_without_depth_raises(tmp_path):
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"adhoc_var": (["time", "depth", "lat", "lon"], rng.random((3, 2, 2, 2)))},
        coords={
            "time": times,
            "depth": [0.0, 10.0],
            "lat": [30.0, 35.0],
            "lon": [-10.0, -5.0],
        },
    )
    src = tmp_path / "store.zarr"
    _write_zarr(ds, src)

    with pytest.raises(ValueError, match="depth"):
        convert_zarr_to_parquet(src, tmp_path / "parquet")


def test_depth_selection(tmp_path):
    times = pd.date_range("2020-01-01", periods=3, freq="D")
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"adhoc_var": (["time", "depth", "lat", "lon"], rng.random((3, 2, 2, 2)))},
        coords={
            "time": times,
            "depth": [0.0, 10.0],
            "lat": [30.0, 35.0],
            "lon": [-10.0, -5.0],
        },
    )
    src = tmp_path / "store.zarr"
    _write_zarr(ds, src)
    out = tmp_path / "parquet"

    convert_zarr_to_parquet(src, out, depth=0.0)

    df = ParquetIndexer(out).load()
    assert "depth" not in df.columns or df["depth"].n_unique() == 1
    assert df.height == 3 * 2 * 2


# ---------------------------------------------------------------------------
# Multi-store input
# ---------------------------------------------------------------------------


def test_list_of_stores(tmp_path):
    a = tmp_path / "a.zarr"
    b = tmp_path / "b.zarr"
    _write_zarr(_make_ds("2020-01-01", 5, seed=1), a)
    _write_zarr(_make_ds("2020-01-06", 5, seed=2), b)
    out = tmp_path / "parquet"

    convert_zarr_to_parquet([a, b], out, time_resolution=TimeResolution.MONTH)

    cov = ParquetIndexer(out).get_time_coverage()
    assert pd.Timestamp(cov.start) == pd.Timestamp("2020-01-01")
    assert pd.Timestamp(cov.end) == pd.Timestamp("2020-01-10")


# ---------------------------------------------------------------------------
# time_resolution accepts a plain string
# ---------------------------------------------------------------------------


def test_time_resolution_accepts_plain_string(tmp_path):
    src = tmp_path / "store.zarr"
    _write_zarr(_make_ds(n_days=40), src)
    out = tmp_path / "parquet"

    # "year" instead of TimeResolution.YEAR — no enum import needed by the caller.
    convert_zarr_to_parquet(src, out, time_resolution="year")

    assert ParquetIndexer(out).load().height == 40 * 2 * 2


def test_invalid_time_resolution_raises(tmp_path):
    src = tmp_path / "store.zarr"
    _write_zarr(_make_ds(n_days=5), src)

    with pytest.raises(ValueError, match="(?i)period|month|year"):
        convert_zarr_to_parquet(src, tmp_path / "parquet", time_resolution="monthly")
