"""Tests for processing/core/cmems.py — pure dataset transform functions."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from h2mare.processing.core.cmems import (
    process_chl,
    process_mld,
    process_ssh,
    process_sst,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_times(n=2):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def _spatial_coords():
    return {"lat": np.array([30.0, 35.0, 40.0]), "lon": np.array([-10.0, -5.0, 0.0])}


def _fake_fdist_ds(var_name: str, times, coords) -> xr.Dataset:
    da = xr.DataArray(
        np.ones((len(times), 3, 3)),
        dims=["time", "lat", "lon"],
        coords={"time": times, **coords},
        name=var_name,
    )
    return da.to_dataset()


# ---------------------------------------------------------------------------
# process_mld
# ---------------------------------------------------------------------------


class TestProcessMld:
    def test_renames_mlotst_to_mld(self):
        times = _make_times()
        coords = _spatial_coords()
        ds = xr.Dataset(
            {"mlotst": (["time", "lat", "lon"], np.ones((2, 3, 3)))},
            coords={"time": times, **coords},
        )
        result = process_mld(ds)
        assert "mld" in result
        assert "mlotst" not in result

    def test_values_preserved_after_rename(self):
        times = _make_times(1)
        coords = _spatial_coords()
        values = np.random.rand(1, 3, 3)
        ds = xr.Dataset(
            {"mlotst": (["time", "lat", "lon"], values)},
            coords={"time": times, **coords},
        )
        result = process_mld(ds)
        np.testing.assert_array_equal(result["mld"].values, values)


# ---------------------------------------------------------------------------
# process_ssh
# ---------------------------------------------------------------------------


class TestProcessSsh:
    def _make_ssh_ds(self, n=2):
        times = _make_times(n)
        coords = _spatial_coords()
        shape = (n, 3, 3)
        return xr.Dataset(
            {
                "adt": (["time", "lat", "lon"], np.random.rand(*shape)),
                "sla": (["time", "lat", "lon"], np.random.rand(*shape)),
                "ugos": (["time", "lat", "lon"], np.full(shape, 3.0)),
                "vgos": (["time", "lat", "lon"], np.full(shape, 4.0)),
            },
            coords={"time": times, **coords},
        )

    def test_adds_adt_std(self):
        result = process_ssh(self._make_ssh_ds())
        assert "adt_std" in result

    def test_adds_sla_std(self):
        result = process_ssh(self._make_ssh_ds())
        assert "sla_std" in result

    def test_adds_gke(self):
        result = process_ssh(self._make_ssh_ds())
        assert "gke" in result

    def test_gke_value_equals_half_speed_squared(self):
        # ugos=3, vgos=4 → speed²=25 → gke=12.5
        result = process_ssh(self._make_ssh_ds())
        np.testing.assert_allclose(result["gke"].values, 12.5, rtol=1e-5)


# ---------------------------------------------------------------------------
# process_sst  (FrontProcessor mocked — too heavy for unit tests)
# ---------------------------------------------------------------------------


class TestProcessSst:
    def _make_ds(self, n=1):
        times = _make_times(n)
        coords = _spatial_coords()
        return xr.Dataset(
            {"analysed_sst": (["time", "lat", "lon"], np.full((n, 3, 3), 300.0))},
            coords={"time": times, **coords},
        )

    def test_renames_analysed_sst_to_sst(self):
        ds = self._make_ds()
        fake_fdist = _fake_fdist_ds("sst_fdist", _make_times(1), _spatial_coords())
        with patch("h2mare.processing.core.cmems.FrontProcessor") as MockFP:
            MockFP.return_value.from_dataset.return_value = fake_fdist
            result = process_sst(ds)
        assert "sst" in result
        assert "analysed_sst" not in result

    def test_converts_kelvin_to_celsius(self):
        ds = self._make_ds()
        fake_fdist = _fake_fdist_ds("sst_fdist", _make_times(1), _spatial_coords())
        with patch("h2mare.processing.core.cmems.FrontProcessor") as MockFP:
            MockFP.return_value.from_dataset.return_value = fake_fdist
            result = process_sst(ds)
        # 300 K − 273.15 = 26.85 °C
        np.testing.assert_allclose(result["sst"].values, 300.0 - 273.15, rtol=1e-4)

    def test_adds_sst_std(self):
        ds = self._make_ds()
        fake_fdist = _fake_fdist_ds("sst_fdist", _make_times(1), _spatial_coords())
        with patch("h2mare.processing.core.cmems.FrontProcessor") as MockFP:
            MockFP.return_value.from_dataset.return_value = fake_fdist
            result = process_sst(ds)
        assert "sst_std" in result


# ---------------------------------------------------------------------------
# process_chl  (FrontProcessor mocked)
# ---------------------------------------------------------------------------


class TestProcessChl:
    def _make_ds(self, n=1):
        times = _make_times(n)
        coords = _spatial_coords()
        return xr.Dataset(
            {"CHL": (["time", "lat", "lon"], np.random.rand(n, 3, 3))},
            coords={"time": times, **coords},
        )

    def test_renames_chl_uppercase_to_lowercase(self):
        ds = self._make_ds()
        fake_fdist = _fake_fdist_ds("chl_fdist", _make_times(1), _spatial_coords())
        with patch("h2mare.processing.core.cmems.FrontProcessor") as MockFP:
            MockFP.return_value.from_dataset.return_value = fake_fdist
            result = process_chl(ds)
        assert "chl" in result
        assert "CHL" not in result
