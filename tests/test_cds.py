"""Tests for processing/core/cds.py pure transformation functions."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.processing.core.cds import (
    _get_ds_for_month,
    daily_cloud_cover,
    daily_sea_level_pressure,
    daily_total_rain,
    daily_waves,
    daily_wind,
    direction_to_uv,
    drop_dims,
    hourly_radiation,
    resample_daily_mean,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_ds(n_days: int = 3, **data_vars) -> xr.Dataset:
    """Minimal hourly dataset (time × lat × lon)."""
    n = n_days * 24
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    lat, lon = [30.0, 35.0], [-10.0, -5.0]
    return xr.Dataset(
        {k: (["time", "lat", "lon"], v) for k, v in data_vars.items()},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def _rad_da(values: list[float], name: str = "ssrd") -> xr.DataArray:
    """Radiation DataArray with lat/lon dims (needed for hourly_radiation transpose)."""
    times = pd.date_range("2020-01-01", periods=len(values), freq="h")
    arr = np.array(values)[:, None, None] * np.ones((1, 2, 2))
    return xr.DataArray(
        arr,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        name=name,
    )


# ---------------------------------------------------------------------------
# _get_ds_for_month
# ---------------------------------------------------------------------------


class TestGetDsForMonth:
    def test_returns_dominant_month_only(self):
        # 17 Jan days + 10 Feb days → January is dominant
        times = pd.date_range("2020-01-15", periods=27, freq="D")
        ds = xr.Dataset({"x": ("time", np.zeros(27))}, coords={"time": times})
        result = _get_ds_for_month(ds)
        assert all(pd.Timestamp(t).month == 1 for t in result.time.values)

    def test_single_month_is_unchanged(self):
        times = pd.date_range("2020-03-01", "2020-03-31", freq="D")
        ds = xr.Dataset({"x": ("time", np.zeros(31))}, coords={"time": times})
        assert len(_get_ds_for_month(ds).time) == 31


# ---------------------------------------------------------------------------
# drop_dims
# ---------------------------------------------------------------------------


class TestDropDims:
    def test_removes_listed_variables(self):
        ds = xr.Dataset({"a": 1.0, "b": 2.0, "c": 3.0})
        result = drop_dims(ds, dims_to_drop=["a", "b"])
        assert "a" not in result
        assert "c" in result

    def test_ignores_absent_names_without_error(self):
        ds = xr.Dataset({"x": 1.0})
        result = drop_dims(ds, dims_to_drop=["x", "does_not_exist"])
        assert "x" not in result


# ---------------------------------------------------------------------------
# resample_daily_mean
# ---------------------------------------------------------------------------


class TestResampleDailyMean:
    def test_collapses_24_steps_to_one_day(self):
        times = pd.date_range("2020-01-01", periods=48, freq="h")
        ds = xr.Dataset({"x": ("time", np.ones(48))}, coords={"time": times})
        assert len(resample_daily_mean(ds).time) == 2

    def test_mean_value_is_correct(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"x": ("time", np.arange(24, dtype=float))}, coords={"time": times}
        )
        np.testing.assert_allclose(resample_daily_mean(ds)["x"].values, [11.5])


# ---------------------------------------------------------------------------
# daily_wind
# ---------------------------------------------------------------------------


class TestDailyWind:
    def test_all_output_variables_present(self):
        ds = _hourly_ds(2, u10=np.ones((48, 2, 2)), v10=np.ones((48, 2, 2)))
        result = daily_wind(ds)
        for v in ("wind_mean", "wind_std", "wind_max", "u10", "v10"):
            assert v in result

    def test_daily_resolution(self):
        ds = _hourly_ds(3, u10=np.ones((72, 2, 2)), v10=np.zeros((72, 2, 2)))
        assert len(daily_wind(ds).time) == 3

    def test_wind_speed_magnitude(self):
        # u=3, v=4 → |w|=5
        ds = _hourly_ds(1, u10=np.full((24, 2, 2), 3.0), v10=np.full((24, 2, 2), 4.0))
        np.testing.assert_allclose(daily_wind(ds)["wind_mean"].values, 5.0, rtol=1e-5)

    def test_raises_without_time_dim(self):
        with pytest.raises(ValueError, match="time"):
            daily_wind(xr.Dataset({"u10": 1.0, "v10": 1.0}))


# ---------------------------------------------------------------------------
# daily_cloud_cover
# ---------------------------------------------------------------------------


class TestDailyCloudCover:
    def test_daily_resolution(self):
        ds = _hourly_ds(3, tcc=np.ones((72, 2, 2)) * 0.5)
        assert len(daily_cloud_cover(ds).time) == 3

    def test_mean_value_preserved(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"tcc": (["time", "lat", "lon"], np.full((24, 2, 2), 0.7))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        np.testing.assert_allclose(daily_cloud_cover(ds)["tcc"].values, 0.7, rtol=1e-5)


# ---------------------------------------------------------------------------
# daily_sea_level_pressure
# ---------------------------------------------------------------------------


class TestDailySeaLevelPressure:
    def test_pa_to_hpa_conversion(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"msl": (["time", "lat", "lon"], np.full((24, 2, 2), 101325.0))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        np.testing.assert_allclose(
            daily_sea_level_pressure(ds)["msl"].values, 1013.25, rtol=1e-5
        )

    def test_units_attribute_set_to_hpa(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"msl": (["time", "lat", "lon"], np.ones((24, 2, 2)))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        assert daily_sea_level_pressure(ds)["msl"].attrs.get("units") == "hPa"


# ---------------------------------------------------------------------------
# hourly_radiation
# ---------------------------------------------------------------------------


class TestHourlyRadiation:
    def test_accumulated_to_watt_rate(self):
        # 3600 J/m² per hour → 1 W/m²
        da = _rad_da([0.0, 3600.0, 7200.0, 10800.0])
        result = hourly_radiation(da)
        assert result.shape[0] == 3  # diff reduces by 1
        np.testing.assert_allclose(result.values, 1.0, rtol=1e-5)

    def test_clips_large_negative_rates_to_zero(self):
        # diff = −36000 J in one step → rate = −10 W/m² → clipped to 0
        da = _rad_da([0.0, -36000.0, 0.0])
        result = hourly_radiation(da, clip_small_negatives=True)
        assert float(result.values[0, 0, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_preserves_negative_when_clip_disabled(self):
        da = _rad_da([0.0, -36000.0, 0.0])
        result = hourly_radiation(da, clip_small_negatives=False)
        assert float(result.values[0, 0, 0]) == pytest.approx(-10.0, rel=1e-5)


# ---------------------------------------------------------------------------
# daily_total_rain
# ---------------------------------------------------------------------------


class TestDailyTotalRain:
    def test_m_to_mm_and_daily_sum(self):
        # 0.001 m/h × 24 h = 24 mm/day
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"tp": (["time", "lat", "lon"], np.full((24, 2, 2), 0.001))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        np.testing.assert_allclose(daily_total_rain(ds)["tp"].values, 24.0, rtol=1e-5)

    def test_units_attribute_is_mm(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"tp": (["time", "lat", "lon"], np.zeros((24, 2, 2)))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        assert daily_total_rain(ds)["tp"].attrs.get("units") == "mm"


# ---------------------------------------------------------------------------
# direction_to_uv
# ---------------------------------------------------------------------------


class TestDirectionToUv:
    def test_east_zero_degrees(self):
        # 0° → u=cos(0)=1, v=sin(0)=0
        da = xr.DataArray([0.0], dims=["time"], name="mdts")
        r = direction_to_uv(da)
        np.testing.assert_allclose(r["u_ts"].values, 1.0, atol=1e-7)
        np.testing.assert_allclose(r["v_ts"].values, 0.0, atol=1e-7)

    def test_north_ninety_degrees(self):
        # 90° → u=0, v=1
        da = xr.DataArray([90.0], dims=["time"], name="mdts")
        r = direction_to_uv(da)
        np.testing.assert_allclose(r["u_ts"].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(r["v_ts"].values, 1.0, atol=1e-7)

    def test_output_variable_names(self):
        da = xr.DataArray([45.0, 135.0], dims=["time"], name="mdts")
        r = direction_to_uv(da)
        assert "u_ts" in r and "v_ts" in r


# ---------------------------------------------------------------------------
# daily_waves
# ---------------------------------------------------------------------------


class TestDailyWaves:
    def test_daily_resolution(self):
        ds = _hourly_ds(3, swh=np.ones((72, 2, 2)), mdts=np.zeros((72, 2, 2)))
        assert len(daily_waves(ds).time) == 3

    def test_both_variables_in_output(self):
        ds = _hourly_ds(2, swh=np.ones((48, 2, 2)), mdts=np.zeros((48, 2, 2)))
        r = daily_waves(ds)
        assert "swh" in r and "mdts" in r

    def test_raises_without_time_dim(self):
        with pytest.raises(ValueError, match="time"):
            daily_waves(xr.Dataset({"swh": 1.0, "mdts": 0.0}))


# ---------------------------------------------------------------------------
# Integration: process_atm_instante
# ---------------------------------------------------------------------------


class TestProcessAtmInstante:
    def test_output_has_all_expected_variables(self):
        from h2mare.processing.core.cds import process_atm_instante

        ds = _hourly_ds(
            2,
            u10=np.ones((48, 2, 2)),
            v10=np.ones((48, 2, 2)),
            tcc=np.ones((48, 2, 2)) * 0.5,
            msl=np.full((48, 2, 2), 101325.0),
        )
        result = process_atm_instante(ds)
        for v in ("wind_mean", "wind_std", "wind_max", "u10", "v10", "tcc", "msl"):
            assert v in result

    def test_lat_is_reversed(self):
        from h2mare.processing.core.cds import process_atm_instante

        ds = _hourly_ds(
            1,
            u10=np.ones((24, 2, 2)),
            v10=np.ones((24, 2, 2)),
            tcc=np.ones((24, 2, 2)),
            msl=np.ones((24, 2, 2)),
        )
        result = process_atm_instante(ds)
        # isel(lat=slice(None, None, -1)) reverses the lat order
        assert list(result.lat.values) == list(reversed(ds.lat.values))


# ---------------------------------------------------------------------------
# Integration: process_waves
# ---------------------------------------------------------------------------


class TestProcessWaves:
    def test_daily_output_with_reversed_lat(self):
        from h2mare.processing.core.cds import process_waves

        ds = _hourly_ds(2, swh=np.ones((48, 2, 2)), mdts=np.zeros((48, 2, 2)))
        result = process_waves(ds)
        assert len(result.time) == 2
        assert list(result.lat.values) == list(reversed(ds.lat.values))
