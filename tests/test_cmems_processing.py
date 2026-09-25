"""Tests for processing/core/cmems.py — pure dataset transform functions."""

import numpy as np
import pandas as pd
import xarray as xr

from h2mare.processing.core.cmems import (
    process_chl,
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

    def test_casts_to_float32(self):
        result = process_ssh(self._make_ssh_ds())
        assert all(result[v].dtype == np.float32 for v in result.data_vars)

    def test_leaves_derived_vars_to_config(self):
        """adt_std, sla_std and gke come from ssh's derived_vars now."""
        result = process_ssh(self._make_ssh_ds())
        assert set(result.data_vars) == {"adt", "sla", "ugos", "vgos"}


# ---------------------------------------------------------------------------
# process_sst
# ---------------------------------------------------------------------------


class TestProcessSst:
    def _make_ds(self, n=1):
        """As the processor gets it: analysed_sst -> sst is config's
        (``source_renames``), applied before this runs."""
        times = _make_times(n)
        coords = _spatial_coords()
        return xr.Dataset(
            {"sst": (["time", "lat", "lon"], np.full((n, 3, 3), 300.0))},
            coords={"time": times, **coords},
        )

    def test_converts_kelvin_to_celsius(self):
        result = process_sst(self._make_ds())
        # 300 K − 273.15 = 26.85 °C
        np.testing.assert_allclose(result["sst"].values, 300.0 - 273.15, rtol=1e-4)

    def test_leaves_the_derived_layers_to_config(self):
        """sst_std comes from derived_vars and sst_fdist from boa_fronts; both
        are applied after the processor, by the convert step."""
        result = process_sst(self._make_ds())
        assert set(map(str, result.data_vars)) == {"sst"}


# ---------------------------------------------------------------------------
# process_chl
# ---------------------------------------------------------------------------


class TestProcessChl:
    def _make_ds(self, n=1):
        """As the processor gets it: CHL -> chl is config's (``source_renames``),
        applied before this runs."""
        times = _make_times(n)
        coords = _spatial_coords()
        return xr.Dataset(
            {"chl": (["time", "lat", "lon"], np.random.rand(n, 3, 3))},
            coords={"time": times, **coords},
        )

    def test_casts_to_float32(self):
        result = process_chl(self._make_ds())
        assert result["chl"].dtype == np.float32

    def test_leaves_the_front_layer_to_config(self):
        """chl_fdist comes from boa_fronts, after the processor."""
        result = process_chl(self._make_ds())
        assert set(map(str, result.data_vars)) == {"chl"}
