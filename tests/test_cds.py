"""Tests for processing/core/cds.py pure transformation functions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.models import TimeStep
from h2mare.processing.core import cds
from h2mare.processing.core.cds import (
    _get_ds_for_month,
    daily_cloud_cover,
    daily_sea_level_pressure,
    daily_total_rain,
    daily_waves,
    daily_wind,
    direction_to_uv,
    drop_scalar_dims,
    drop_valid_time,
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
# drop_scalar_dims / drop_valid_time
# ---------------------------------------------------------------------------


class TestDropScalarDims:
    def test_removes_the_grib_scalar_coords(self):
        ds = xr.Dataset({"swh": 1.0}).assign_coords(
            step=pd.Timedelta(0), number=0, surface=0.0
        )
        result = drop_scalar_dims(ds)
        assert set(result.coords) == set()
        assert "swh" in result

    def test_ignores_absent_names_without_error(self):
        ds = xr.Dataset({"swh": 1.0}).assign_coords(number=0)
        assert "number" not in drop_scalar_dims(ds).coords

    def test_leaves_valid_time_alone(self):
        """It varies along a dimension, so it is not this function's business."""
        times = pd.date_range("2020-01-01", periods=3, freq="h")
        ds = xr.Dataset(
            {"swh": (["time"], np.ones(3))},
            coords={"time": times, "valid_time": ("time", times.values)},
        )
        assert "valid_time" in drop_scalar_dims(ds).coords


class TestDropValidTime:
    """
    Dropping valid_time is safe only once `time` carries the axis. The unsafe
    case does not raise — it leaves the dimension in place with no labels — so
    the guard is what stands between a redundant coord and lost timestamps.
    """

    @staticmethod
    def _times():
        return pd.date_range("2020-01-01", periods=3, freq="h")

    def test_drops_it_when_time_carries_the_axis(self):
        times = self._times()
        ds = xr.Dataset(
            {"swh": (["time"], np.ones(3))},
            coords={"time": times, "valid_time": ("time", times.values)},
        )
        result = drop_valid_time(ds)

        assert "valid_time" not in result.coords
        assert list(result.time.values) == list(times.values)

    def test_keeps_it_when_it_is_the_axis(self):
        """Without the guard this silently strips every timestamp."""
        times = self._times()
        ds = xr.Dataset(
            {"swh": (["valid_time"], np.ones(3))}, coords={"valid_time": times}
        )
        result = drop_valid_time(ds)

        assert "valid_time" in result.coords, (
            "dropping the dimension coordinate leaves the dimension unlabelled "
            "rather than raising — the timestamps would just be gone"
        )

    def test_absent_valid_time_is_a_no_op(self):
        ds = xr.Dataset({"swh": (["time"], np.ones(3))}, coords={"time": self._times()})
        assert drop_valid_time(ds).equals(ds)


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
    """
    ERA5's hourly accumulations already cover only their own interval, so the
    conversion is a division and nothing else. The previous implementation
    differenced first, treating the field as a running total; these tests are
    written against ERA5's actual numbers rather than that assumption.
    """

    def test_accumulated_to_watt_rate(self):
        # 3600 J/m² accumulated over the hour → 1 W/m²
        da = _rad_da([3600.0, 3600.0, 3600.0, 3600.0])
        result = hourly_radiation(da)
        np.testing.assert_allclose(result.values, 1.0, rtol=1e-5)

    def test_every_timestep_survives(self):
        """Differencing silently dropped the first hour of every period."""
        da = _rad_da([3600.0, 7200.0, 10800.0, 14400.0])
        assert hourly_radiation(da).shape[0] == 4

    def test_a_falling_accumulation_is_not_a_reset(self):
        """
        A real tisr block rises and falls with the sun. Differencing read the
        falling limb as negative rates and the clip then zeroed them, which is
        what made stored tisr several times too small.
        """
        da = _rad_da([0.0, 905088.0, 2423552.0, 1144320.0, 0.0])
        result = hourly_radiation(da)

        np.testing.assert_allclose(
            result.values[:, 0, 0],
            np.array([0.0, 905088.0, 2423552.0, 1144320.0, 0.0]) / 3600.0,
            rtol=1e-5,
        )
        assert float(result.values[:, 0, 0].max()) == pytest.approx(673.2, rel=1e-3)

    def test_genuinely_negative_flux_is_preserved(self):
        """slhf is negative over essentially the whole ocean."""
        da = _rad_da([-360000.0, -360000.0, -360000.0])
        result = hourly_radiation(da, clip_small_negatives=True)

        np.testing.assert_allclose(result.values, -100.0, rtol=1e-5)

    def test_only_the_noise_sliver_is_clipped(self):
        da = _rad_da([-1e-4, -3600.0, 3600.0])  # → -2.8e-8, -1.0, +1.0 W/m²
        result = hourly_radiation(da, clip_small_negatives=True).values[:, 0, 0]

        assert float(result[0]) == 0.0, "rounding noise should flatten to zero"
        assert float(result[1]) == pytest.approx(-1.0, rel=1e-5), (
            "a real negative flux must survive the clip"
        )

    def test_preserves_negative_when_clip_disabled(self):
        da = _rad_da([-1e-4, -3600.0])
        result = hourly_radiation(da, clip_small_negatives=False).values[:, 0, 0]
        assert float(result[0]) < 0.0

    def test_period_is_read_from_the_axis(self):
        """A 3-hourly product divides by 10800, not 3600."""
        times = pd.date_range("2020-01-01", periods=3, freq="3h")
        da = xr.DataArray(
            np.full((3, 2, 2), 10800.0),
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
            name="ssrd",
        )
        np.testing.assert_allclose(hourly_radiation(da).values, 1.0, rtol=1e-5)

    def test_single_timestep_raises_rather_than_guessing(self):
        da = _rad_da([3600.0])
        with pytest.raises(ValueError, match="accumulation period"):
            hourly_radiation(da)

    def test_attrs_describe_the_rate_not_the_source(self):
        """
        The output is a rate, so every attribute that says otherwise has to be
        rewritten. GRIB_stepType matters most: left at 'accum' it invites the
        same second de-accumulation this function used to perform.
        """
        da = _rad_da([3600.0, 3600.0])
        da.attrs.update(
            {"units": "J m**-2", "GRIB_units": "J m**-2", "GRIB_stepType": "accum"}
        )
        attrs = hourly_radiation(da).attrs

        assert attrs["units"] == "W m-2"
        assert attrs["GRIB_units"] == "W m-2"
        assert attrs["GRIB_stepType"] == "avg", (
            "a rate must not still advertise itself as an accumulation"
        )

    def test_units_are_plain_ascii(self):
        """
        The units string travels to Parquet and on to CSV, where a superscript
        is mangled by any reader that does not assume UTF-8 — Windows' default
        codepage among them.
        """
        attrs = hourly_radiation(_rad_da([3600.0, 3600.0])).attrs

        for key in ("units", "GRIB_units"):
            assert attrs[key].isascii(), (
                f"{key}={attrs[key]!r} carries a non-ASCII character"
            )

    def test_source_provenance_is_otherwise_preserved(self):
        da = _rad_da([3600.0, 3600.0])
        da.attrs.update({"GRIB_paramId": 212, "GRIB_shortName": "tisr"})
        attrs = hourly_radiation(da).attrs

        assert attrs["GRIB_paramId"] == 212
        assert attrs["GRIB_shortName"] == "tisr"


# ---------------------------------------------------------------------------
# process_radiation — cadence selected by config
# ---------------------------------------------------------------------------


class TestProcessRadiationCadence:
    """
    time_step decides whether convert averages. Both cadences convert J/m²→W/m²
    here, so the store and h2ds agree on units either way — unlike the other
    hourly stores, which keep their source raw.
    """

    @staticmethod
    def _ds():
        times = pd.date_range("2020-01-01", periods=48, freq="h")
        shape = (48, 2, 2)
        return xr.Dataset(
            {
                "ssrd": (["time", "lat", "lon"], np.full(shape, 3600.0)),
                "slhf": (["time", "lat", "lon"], np.full(shape, -360000.0)),
            },
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )

    def test_daily_config_averages_as_before(self):
        cfg = SimpleNamespace(time_step=TimeStep.DAILY)
        out = cds.process_radiation(self._ds(), cfg, "radiation")

        assert out.sizes["time"] == 2, "daily store must still be one step per day"

    def test_hourly_config_keeps_the_native_axis(self):
        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)
        out = cds.process_radiation(self._ds(), cfg, "radiation")

        assert out.sizes["time"] == 48, "hourly store must not be averaged"

    def test_both_cadences_publish_the_same_units(self):
        """3600 J/m² per hour is 1 W/m² whether or not a daily mean follows."""
        daily = cds.process_radiation(
            self._ds(), SimpleNamespace(time_step=TimeStep.DAILY), "radiation"
        )
        hourly = cds.process_radiation(
            self._ds(), SimpleNamespace(time_step=TimeStep.HOURLY), "radiation"
        )

        np.testing.assert_allclose(daily["ssrd"].values, 1.0, rtol=1e-5)
        np.testing.assert_allclose(hourly["ssrd"].values, 1.0, rtol=1e-5)

    def test_hourly_store_keeps_negative_flux(self):
        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)
        out = cds.process_radiation(self._ds(), cfg, "radiation")

        np.testing.assert_allclose(out["slhf"].values, -100.0, rtol=1e-5)

    def test_hourly_lat_is_reversed_like_the_daily_path(self):
        ds = self._ds()
        out = cds.process_radiation(
            ds, SimpleNamespace(time_step=TimeStep.HOURLY), "radiation"
        )
        assert list(out.lat.values) == list(reversed(ds.lat.values))

    def test_grib_coords_never_reach_the_store(self):
        ds = self._ds().assign_coords(
            number=0, step=pd.Timedelta(0), surface=0.0, meanSea=0.0
        )
        for step in (TimeStep.HOURLY, TimeStep.DAILY):
            out = cds.process_radiation(
                ds, SimpleNamespace(time_step=step), "radiation"
            )
            assert set(out.coords) == {"time", "lat", "lon"}, (
                f"GRIB coords reached the {step} store: {sorted(out.coords)}"
            )


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
# process_atm_instante — cadence selected by config
# ---------------------------------------------------------------------------


class TestProcessAtmInstanteCadence:
    """
    time_step decides whether convert aggregates. An hourly store keeps ERA5's
    raw fields; the daily reductions move to compile time.
    """

    @staticmethod
    def _ds():
        shape = (48, 2, 2)
        return _hourly_ds(
            2,
            u10=np.full(shape, 3.0),
            v10=np.full(shape, 4.0),
            tcc=np.full(shape, 0.5),
            msl=np.full(shape, 101325.0),
        )

    def test_daily_config_aggregates_as_before(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_atm_instante

        cfg = SimpleNamespace(time_step=TimeStep.DAILY)
        out = process_atm_instante(self._ds(), cfg, "atm-instante")

        assert out.sizes["time"] == 2, "daily store must still be one step per day"
        assert "wind_mean" in out.data_vars

    def test_hourly_config_keeps_the_native_axis(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_atm_instante

        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)
        out = process_atm_instante(self._ds(), cfg, "atm-instante")

        assert out.sizes["time"] == 48, "hourly store must not be resampled"

    def test_hourly_store_holds_only_the_raw_fields(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_atm_instante

        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)
        out = process_atm_instante(self._ds(), cfg, "atm-instante")

        assert set(out.data_vars) == {"msl", "u10", "v10", "tcc"}, (
            "derived wind features belong to compile time, not the hourly store"
        )

    def test_hourly_msl_stays_in_native_pascals(self):
        """daily_sea_level_pressure does the Pa→hPa conversion at compile time."""
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_atm_instante

        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)
        out = process_atm_instante(self._ds(), cfg, "atm-instante")

        np.testing.assert_allclose(out["msl"].values, 101325.0)

    def test_hourly_lat_is_reversed_like_the_daily_path(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_atm_instante

        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)
        ds = self._ds()
        out = process_atm_instante(ds, cfg, "atm-instante")

        assert list(out.lat.values) == list(reversed(ds.lat.values))

    def test_missing_optional_field_is_tolerated(self):
        """A store without tcc must not blow up on the selection."""
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_atm_instante

        ds = self._ds().drop_vars("tcc")
        out = process_atm_instante(ds, SimpleNamespace(time_step=TimeStep.HOURLY))

        assert set(out.data_vars) == {"msl", "u10", "v10"}


class TestAtmInstanteDropsGribCoords:
    """
    cfgrib attaches number/step/surface/valid_time to every ERA5 field. The
    daily path sheds them twice over — drop_dims in each builder, and the
    resample discarding valid_time — so a cadence that does neither has to drop
    them itself or they reach the store. atm-accum-avg is clean without this
    only because merge_time_step already dropped them; atm-instante has no such
    step.
    """

    @staticmethod
    def _grib_shaped_ds():
        times = pd.date_range("2020-01-01", periods=48, freq="h")
        shape = (48, 2, 2)
        ds = xr.Dataset(
            {
                name: (["time", "lat", "lon"], np.full(shape, val, dtype="float32"))
                for name, val in [
                    ("u10", 3.0),
                    ("v10", 4.0),
                    ("tcc", 0.5),
                    ("msl", 101325.0),
                ]
            },
            coords={"time": times, "lat": [30.0, 30.25], "lon": [-10.0, -9.75]},
        )
        return ds.assign_coords(
            number=0,
            step=pd.Timedelta(0),
            surface=0.0,
            valid_time=("time", times.values),
        )

    def test_hourly_store_keeps_only_time_lat_lon(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_atm_instante

        out = process_atm_instante(
            self._grib_shaped_ds(), SimpleNamespace(time_step=TimeStep.HOURLY)
        )

        assert set(out.coords) == {"time", "lat", "lon"}, (
            f"GRIB coords reached the hourly store: {sorted(out.coords)}"
        )

    def test_daily_path_is_unchanged(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_atm_instante

        out = process_atm_instante(
            self._grib_shaped_ds(), SimpleNamespace(time_step=TimeStep.DAILY)
        )

        assert set(out.coords) == {"time", "lat", "lon"}


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


# ---------------------------------------------------------------------------
# process_waves — cadence selected by config
# ---------------------------------------------------------------------------


class TestProcessWavesCadence:
    """
    time_step decides whether convert aggregates. The signal reaches the
    processor through var_config, which every processor already receives.
    """

    @staticmethod
    def _ds():
        shape = (48, 2, 2)
        return _hourly_ds(
            2,
            swh=np.full(shape, 1.5),
            mdts=np.full(shape, 180.0),
        )

    def test_daily_config_aggregates_as_before(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_waves

        cfg = SimpleNamespace(time_step=TimeStep.DAILY)
        out = process_waves(self._ds(), cfg, "waves")

        assert out.sizes["time"] == 2, "daily store must still be one step per day"
        assert set(out.data_vars) == {"swh", "mdts"}

    def test_hourly_config_keeps_the_native_axis(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_waves

        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)
        out = process_waves(self._ds(), cfg, "waves")

        assert out.sizes["time"] == 48, "hourly store must keep every step"
        assert set(out.data_vars) == {"swh", "mdts"}

    def test_missing_config_defaults_to_daily(self):
        """A processor called without config keeps the historical behaviour."""
        from h2mare.processing.core.cds import process_waves

        out = process_waves(self._ds(), None, "waves")
        assert out.sizes["time"] == 2


# ---------------------------------------------------------------------------
# mdts is a direction — circular, not linear
# ---------------------------------------------------------------------------


class TestWaveDirectionIsCircular:
    """
    Averaging degrees arithmetically is wrong across the 0/360 wrap: 350° and
    10° are 20° apart and average to 0°, not 180°.
    """

    @staticmethod
    def _wrapping_ds():
        """One day whose directions straddle north: half at 350°, half at 10°."""
        shape = (24, 2, 2)
        mdts = np.full(shape, 350.0)
        mdts[12:] = 10.0
        return _hourly_ds(1, swh=np.full(shape, 2.0), mdts=mdts)

    def test_daily_mean_direction_does_not_flip_across_north(self):
        out = daily_waves(self._wrapping_ds())
        got = float(out["mdts"].isel(time=0, lat=0, lon=0))

        # 0/360 are the same heading, so accept either end.
        assert min(got, 360.0 - got) < 1e-6, (
            f"expected ~0/360 (north), got {got} — arithmetic mean would give 180"
        )

    def test_height_still_averages_arithmetically(self):
        out = daily_waves(self._wrapping_ds())
        assert float(out["swh"].isel(time=0, lat=0, lon=0)) == pytest.approx(2.0)

    def test_uv_round_trip_preserves_direction(self):
        from h2mare.processing.core.cds import uv_to_direction

        da = xr.DataArray([0.0, 10.0, 90.0, 180.0, 350.0], dims="time")
        comp = direction_to_uv(da)
        back = uv_to_direction(comp["u_ts"], comp["v_ts"])
        np.testing.assert_allclose(back.values, da.values, atol=1e-9)


# ---------------------------------------------------------------------------
# get_previous_dates_da — rolling-feature warm-up
# ---------------------------------------------------------------------------


class TestEkmanSeedWindow:
    """
    The seed prepended before a range decides whether the deep rolling features
    start warm. Nothing downstream can detect a short seed: ``min_periods=1``
    accepts a partial window and emits a plausible wrong number instead of
    raising, so these assertions are the only guard.
    """

    @staticmethod
    def _capture_requested_span(monkeypatch, t0: pd.Timestamp) -> dict:
        seen: dict[str, pd.Timestamp] = {}

        class _FakeCatalog:
            def __init__(self, var_key):
                pass

            def open_dataset(self, start_date, end_date):
                seen["start"] = pd.Timestamp(start_date)
                seen["end"] = pd.Timestamp(end_date)
                return None  # early-return path; only the request matters here

            def close(self):
                pass

        monkeypatch.setattr(cds, "ZarrCatalog", _FakeCatalog)

        da = xr.DataArray(
            np.zeros(5),
            dims=["time"],
            coords={"time": pd.date_range(t0, periods=5, freq="D")},
        )
        cds.get_previous_dates_da(da, "atm-accum-avg")
        return seen

    def test_seed_covers_the_deepest_lagged_feature(self, monkeypatch):
        """
        ekman_anom_lag14 reads the anomaly 14 days back, and that anomaly is
        itself a 7-day rolling mean — so the first output day needs 14 + 7 - 1
        days of real history behind it. The original 15-day seed was 5 short,
        which silently corrupted lag14 for the first weeks of every isolated
        range.
        """
        t0 = pd.Timestamp("2020-01-01")
        seen = self._capture_requested_span(monkeypatch, t0)

        needed = max(cds._EKMAN_LAGS) + cds._EKMAN_ROLL_DAYS - 1
        assert (t0 - seen["start"]).days >= needed, (
            f"seed spans {(t0 - seen['start']).days} days but lag"
            f"{max(cds._EKMAN_LAGS)} needs {needed}"
        )

    def test_seed_covers_the_deepest_event_window(self, monkeypatch):
        """n_upwell_events_14d sums exceedances back to t-13, each needing 7."""
        t0 = pd.Timestamp("2020-01-01")
        seen = self._capture_requested_span(monkeypatch, t0)

        needed = max(cds._EKMAN_EVENT_WINDOWS) - 1 + cds._EKMAN_ROLL_DAYS - 1
        assert (t0 - seen["start"]).days >= needed

    def test_seed_stops_the_day_before_the_range(self, monkeypatch):
        """The seed must abut the range, never overlap it."""
        t0 = pd.Timestamp("2020-01-01")
        seen = self._capture_requested_span(monkeypatch, t0)

        assert seen["end"] == t0 - pd.Timedelta(days=1)

    def test_hourly_store_keeps_the_native_axis_and_raw_fields_only(self):
        """
        With ``time_step: hourly`` the convert step stores raw ERA5 and derives
        nothing — the ekman chain is daily by construction and moves to compile.
        """
        n_days = 3
        ds = _hourly_ds(
            n_days=n_days,
            avg_iews=np.ones((n_days * 24, 2, 2)),
            avg_inss=np.ones((n_days * 24, 2, 2)),
            tp=np.ones((n_days * 24, 2, 2)),
        )
        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)

        out = cds.process_atm_accum_avg(ds, cfg, "atm-accum-avg")

        assert out.sizes["time"] == n_days * 24, "hourly axis must not be resampled"
        assert set(out.data_vars) == {"avg_iews", "avg_inss", "tp"}
        assert not [v for v in out.data_vars if "ekman" in v or "upwell" in v]

    def test_hourly_store_drops_fields_the_pipeline_does_not_publish(self):
        ds = _hourly_ds(
            n_days=1,
            avg_iews=np.ones((24, 2, 2)),
            avg_inss=np.ones((24, 2, 2)),
            tp=np.ones((24, 2, 2)),
            stray=np.ones((24, 2, 2)),
        )
        out = cds.process_atm_accum_avg(
            ds, SimpleNamespace(time_step=TimeStep.HOURLY), "atm-accum-avg"
        )

        assert "stray" not in out.data_vars

    def test_hourly_store_matches_the_daily_stores_lat_orientation(self):
        """Both cadences must write ascending lat, or compile regrids upside down."""
        ds = _hourly_ds(
            n_days=1,
            avg_iews=np.ones((24, 2, 2)),
            avg_inss=np.ones((24, 2, 2)),
            tp=np.ones((24, 2, 2)),
        )
        out = cds.process_atm_accum_avg(
            ds, SimpleNamespace(time_step=TimeStep.HOURLY), "atm-accum-avg"
        )

        assert list(out.lat.values) == list(ds.lat.values)[::-1]

    def test_warmup_is_derived_from_the_declared_depths(self):
        """
        Adding a deeper lag must widen the seed automatically — the constant is
        computed from the tuples, so this fails if someone reverts it to a
        literal and then adds lag21.
        """
        assert cds._EKMAN_WARMUP_DAYS >= max(cds._EKMAN_LAGS) + cds._EKMAN_ROLL_DAYS - 1
        assert (
            cds._EKMAN_WARMUP_DAYS
            >= max(cds._EKMAN_EVENT_WINDOWS) - 1 + cds._EKMAN_ROLL_DAYS - 1
        )


# ---------------------------------------------------------------------------
# compute_curl_and_ekman — spatial stencil chunking
# ---------------------------------------------------------------------------


class TestCurlStencilChunking:
    """The curl rolls along lat/lon, so those axes must not be tiled."""

    def test_curl_runs_on_lat_lon_contiguous_chunks(self):
        """
        The curl is a spatial stencil, so lat/lon must sit in one chunk each.

        A store written with the "timeseries" layout is spatially tiled; rolling
        across those tiles shuffles every chunk and the circular wrap couples
        opposite ends of the axis, which is what stalled a real compile. Values
        are unaffected either way, so only the chunking can catch it.
        """
        n_lat, n_lon = 12, 16
        times = pd.date_range("2020-01-01", periods=48, freq="h")
        shape = (len(times), n_lat, n_lon)
        ds = xr.Dataset(
            {
                "avg_iews": (["time", "lat", "lon"], np.ones(shape)),
                "avg_inss": (["time", "lat", "lon"], np.ones(shape)),
            },
            coords={
                "time": times,
                "lat": np.linspace(30.0, 41.0, n_lat),
                "lon": np.linspace(-10.0, 5.0, n_lon),
            },
        )
        # Tile the spatial dims the way the on-disk timeseries layout does.
        tiled = ds.chunk({"time": 24, "lat": n_lat // 2, "lon": n_lon // 2})
        assert len(tiled["avg_iews"].chunks[1]) > 1, "fixture must be tiled"

        out = cds.compute_curl_and_ekman(tiled)

        lat_chunks = out["ekman_pumping"].chunks[out["ekman_pumping"].dims.index("lat")]
        lon_chunks = out["ekman_pumping"].chunks[out["ekman_pumping"].dims.index("lon")]
        assert len(lat_chunks) == 1, f"lat still tiled: {lat_chunks}"
        assert len(lon_chunks) == 1, f"lon still tiled: {lon_chunks}"

    def test_curl_leaves_unchunked_input_unchunked(self):
        """A numpy-backed dataset must not be forced into dask by the rechunk."""
        n_lat, n_lon = 12, 16
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        shape = (len(times), n_lat, n_lon)
        ds = xr.Dataset(
            {
                "avg_iews": (["time", "lat", "lon"], np.ones(shape)),
                "avg_inss": (["time", "lat", "lon"], np.ones(shape)),
            },
            coords={
                "time": times,
                "lat": np.linspace(30.0, 41.0, n_lat),
                "lon": np.linspace(-10.0, 5.0, n_lon),
            },
        )

        out = cds.compute_curl_and_ekman(ds)

        assert out["ekman_pumping"].chunks is None


# ---------------------------------------------------------------------------
# add_engineered_ekman — partial windows at the start of the archive
# ---------------------------------------------------------------------------


class TestUpwellEventPartialWindow:
    """
    An N-day event count must not answer before N days of history exist.

    The lagged anomalies already report missing history as NaN, because shift
    fills its leading positions. The counts used min_periods=1, so they emitted
    a count over however many days happened to be there — a silent under-count
    indistinguishable from a genuinely quiet period.
    """

    # Mid-Atlantic, so clip_land_data does not blank the fixture.
    _LATS = [30.0, 30.25]
    _LONS = [-40.0, -39.75]

    def _write_climatology(self, tmp_path):
        """Zero climatology, so every day counts as an exceedance."""
        shape2d = (len(self._LATS), len(self._LONS))
        xr.Dataset(
            {
                "ekman_pumping_anom": (
                    ["month", "lat", "lon"],
                    np.zeros((12, *shape2d)),
                )
            },
            coords={
                "month": np.arange(1, 13),
                "lat": self._LATS,
                "lon": self._LONS,
            },
        ).to_netcdf(tmp_path / cds._EKMAN_P90_FILE)

        xr.Dataset(
            {
                "ekman_pumping": (
                    ["dayofyear", "lat", "lon"],
                    np.zeros((cds._EKMAN_DOY_BUCKETS, *shape2d)),
                )
            },
            coords={
                "dayofyear": np.arange(1, cds._EKMAN_DOY_BUCKETS + 1),
                "lat": self._LATS,
                "lon": self._LONS,
            },
        ).to_netcdf(tmp_path / cds._EKMAN_DOY_FILE)

    def _features(self, tmp_path, monkeypatch, n_days: int = 30):
        self._write_climatology(tmp_path)
        monkeypatch.setattr(
            cds, "get_settings", lambda: SimpleNamespace(CLIMATOLOGY_DIR=tmp_path)
        )

        times = pd.date_range("2020-01-01", periods=n_days, freq="D")
        da = xr.DataArray(
            np.ones((n_days, len(self._LATS), len(self._LONS))),
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": self._LATS, "lon": self._LONS},
            name="ekman_pumping",
        )
        # seed_from_store=False: no store here, and it is the compile path.
        return cds.add_engineered_ekman(da, "atm-accum-avg", seed_from_store=False)

    def test_count_is_nan_until_its_window_is_full(self, tmp_path, monkeypatch):
        out = self._features(tmp_path, monkeypatch)

        for w in cds._EKMAN_EVENT_WINDOWS:
            series = out[f"n_upwell_events_{w}d"].isel(lat=0, lon=0).values
            assert np.isnan(series[: w - 1]).all(), (
                f"n_upwell_events_{w}d answered before {w} days existed: "
                f"{series[: w - 1]}"
            )

    def test_count_reports_the_full_window_once_warm(self, tmp_path, monkeypatch):
        out = self._features(tmp_path, monkeypatch)

        for w in cds._EKMAN_EVENT_WINDOWS:
            series = out[f"n_upwell_events_{w}d"].isel(lat=0, lon=0).values
            assert series[w - 1] == w, f"n_upwell_events_{w}d should saturate at {w}"
            assert (series[w - 1 :] == w).all()

    def test_counts_no_longer_ramp_from_one(self, tmp_path, monkeypatch):
        """
        The old signature: 1, 2, 3 … w. Each of those was a count over fewer
        than w days wearing the name of a w-day count.
        """
        out = self._features(tmp_path, monkeypatch)
        series = out["n_upwell_events_14d"].isel(lat=0, lon=0).values

        assert not np.array_equal(series[:3], np.array([1.0, 2.0, 3.0]))

    def test_lagged_anomaly_still_nan_over_the_same_gap(self, tmp_path, monkeypatch):
        """The two families should now agree about what is missing."""
        out = self._features(tmp_path, monkeypatch)

        for lag in cds._EKMAN_LAGS:
            series = out[f"ekman_anom_lag{lag}"].isel(lat=0, lon=0).values
            assert np.isnan(series[:lag]).all()


# ---------------------------------------------------------------------------
# calendar_doy — the leap-year alignment of the climatology
# ---------------------------------------------------------------------------


def _doy(dates) -> list[int]:
    time = xr.DataArray(pd.to_datetime(dates), dims="time", name="time")
    return [int(v) for v in cds.calendar_doy(time).values]


class TestCalendarDoy:
    def test_a_calendar_date_keeps_its_index_across_leap_years(self):
        """
        The whole point: 1 March must be the same bucket every year. Raw
        dayofyear gives 60 in a common year and 61 in a leap year, so the
        climatology averaged 1 March with 2 March and the anomaly subtracted
        the wrong day for ten months of every fourth year.
        """
        assert _doy(["2019-03-01", "2020-03-01", "2021-03-01"]) == [60, 60, 60]
        assert _doy(["2019-12-31", "2020-12-31"]) == [365, 365]

    def test_before_march_is_untouched(self):
        assert _doy(["2019-01-01", "2020-01-01", "2020-02-28"]) == [1, 1, 59]

    def test_leap_day_borrows_the_28th(self):
        """Five samples in a twenty-year baseline does not make a bucket."""
        assert _doy(["2020-02-29"]) == [59]

    def test_never_leaves_the_365_day_calendar(self):
        every_day = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        values = cds.calendar_doy(
            xr.DataArray(every_day, dims="time", name="time")
        ).values
        assert values.min() == 1
        assert values.max() == cds._EKMAN_DOY_BUCKETS


class TestClimatologyGrid:
    """
    The anomaly and the event counts subtract and compare against the
    climatology with xarray's default inner join, so a climatology on less
    than the data's grid shrank them to the overlap, silently.
    """

    _LATS = [30.0, 30.25, 30.5]
    _LONS = [-40.0, -39.75]

    def _write(self, tmp_path, lats, lons):
        shape2d = (len(lats), len(lons))
        coords = {"lat": lats, "lon": lons}
        xr.Dataset(
            {"ekman_pumping_anom": (["month", "lat", "lon"], np.zeros((12, *shape2d)))},
            coords={"month": np.arange(1, 13), **coords},
        ).to_netcdf(tmp_path / cds._EKMAN_P90_FILE)
        n = cds._EKMAN_DOY_BUCKETS
        xr.Dataset(
            {"ekman_pumping": (["dayofyear", "lat", "lon"], np.zeros((n, *shape2d)))},
            coords={"dayofyear": np.arange(1, n + 1), **coords},
        ).to_netcdf(tmp_path / cds._EKMAN_DOY_FILE)

    def _run(self, tmp_path, monkeypatch, lats, lons):
        self._write(tmp_path, lats, lons)
        monkeypatch.setattr(
            cds, "get_settings", lambda: SimpleNamespace(CLIMATOLOGY_DIR=tmp_path)
        )
        times = pd.date_range("2020-01-01", periods=40, freq="D")
        da = xr.DataArray(
            np.ones((len(times), len(self._LATS), len(self._LONS))),
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": self._LATS, "lon": self._LONS},
            name="ekman_pumping",
        )
        return cds.add_engineered_ekman(da, "atm-accum-avg", seed_from_store=False)

    def test_a_climatology_short_of_the_grid_is_refused(self, tmp_path, monkeypatch):
        """Regression: the anomaly came back on 2 of the data's 3 latitudes."""
        with pytest.raises(ValueError, match=r"1 of 3 lat values are not on it"):
            self._run(tmp_path, monkeypatch, self._LATS[:2], self._LONS)

    def test_labels_a_float_apart_still_cover_every_cell(self, tmp_path, monkeypatch):
        """Regression: an inner join on values a float apart kept no cell at all."""
        lats = [v + 1e-9 for v in self._LATS]
        out = self._run(tmp_path, monkeypatch, lats, self._LONS)

        assert out.sizes["lat"] == 3 and out.sizes["lon"] == 2
        np.testing.assert_array_equal(out["lat"].values, self._LATS)
        assert np.isfinite(out["ekman_anom"].values).all()

    def test_a_wider_climatology_is_cut_to_the_data(self, tmp_path, monkeypatch):
        out = self._run(tmp_path, monkeypatch, [29.75, *self._LATS], self._LONS)
        np.testing.assert_array_equal(out["lat"].values, self._LATS)


class TestClimatologyAlignment:
    """The consumer must read the climatology on the same calendar it was built on."""

    _LATS = [30.0, 30.25]
    _LONS = [-40.0, -39.75]

    def _write_climatology(self, tmp_path, n_buckets: int):
        shape2d = (len(self._LATS), len(self._LONS))
        coords = {"lat": self._LATS, "lon": self._LONS}
        xr.Dataset(
            {"ekman_pumping_anom": (["month", "lat", "lon"], np.zeros((12, *shape2d)))},
            coords={"month": np.arange(1, 13), **coords},
        ).to_netcdf(tmp_path / cds._EKMAN_P90_FILE)

        # Each bucket holds its own index, so an anomaly reveals which bucket
        # was selected: anomaly = value - bucket_index.
        buckets = np.arange(1, n_buckets + 1, dtype="float64")
        xr.Dataset(
            {
                "ekman_pumping": (
                    ["dayofyear", "lat", "lon"],
                    np.broadcast_to(buckets[:, None, None], (n_buckets, *shape2d)),
                )
            },
            coords={"dayofyear": np.arange(1, n_buckets + 1), **coords},
        ).to_netcdf(tmp_path / cds._EKMAN_DOY_FILE)

    def _run(self, tmp_path, monkeypatch, start: str, n_buckets: int = 365):
        self._write_climatology(tmp_path, n_buckets)
        monkeypatch.setattr(
            cds, "get_settings", lambda: SimpleNamespace(CLIMATOLOGY_DIR=tmp_path)
        )
        # Long enough to outrun the deepest lag; shift() past the end of a
        # shorter axis leaves dask with a zero-length chunk.
        times = pd.date_range(start, periods=40, freq="D")
        da = xr.DataArray(
            np.zeros((len(times), len(self._LATS), len(self._LONS))),
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": self._LATS, "lon": self._LONS},
            name="ekman_pumping",
        )
        return cds.add_engineered_ekman(da, "atm-accum-avg", seed_from_store=False)

    @staticmethod
    def _bucket_on(out, date: str) -> float:
        """Which climatology bucket was subtracted on *date*.

        The input is zero and each bucket holds its own index, so the anomaly
        is the negated index of whichever bucket the selection landed on.
        """
        return float(-out["ekman_anom"].sel(time=date).isel(lat=0, lon=0))

    def test_leap_year_march_reads_the_calendar_aligned_bucket(
        self, tmp_path, monkeypatch
    ):
        """
        1 March must subtract the same bucket whatever the year. On raw
        dayofyear a leap year takes 61 — the day after — for every day from
        March to December.
        """
        leap = self._run(tmp_path, monkeypatch, "2020-02-20")

        assert self._bucket_on(leap, "2020-03-01") == 60.0
        assert self._bucket_on(leap, "2020-03-15") == 74.0

    def test_common_year_agrees_with_the_leap_year(self, tmp_path, monkeypatch):
        common = self._run(tmp_path, monkeypatch, "2019-02-20")

        assert self._bucket_on(common, "2019-03-01") == 60.0
        assert self._bucket_on(common, "2019-03-15") == 74.0

    def test_leap_day_takes_the_28th_bucket(self, tmp_path, monkeypatch):
        out = self._run(tmp_path, monkeypatch, "2020-02-20")

        assert self._bucket_on(out, "2020-02-29") == 59.0
        assert self._bucket_on(out, "2020-02-28") == 59.0

    def test_a_366_bucket_file_is_refused(self, tmp_path, monkeypatch):
        """
        It would select happily — every index asked for exists — and return a
        climatology a day out. Nothing downstream could tell.
        """
        with pytest.raises(ValueError, match="366 dayofyear buckets"):
            self._run(tmp_path, monkeypatch, "2020-02-20", n_buckets=366)
