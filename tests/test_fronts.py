"""Tests for processing/core/fronts.py and the boa_fronts config entry."""

import os
from pathlib import Path
from unittest.mock import patch

import msgspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from h2mare.config import get_settings
from h2mare.models import AppConfig, BOAFrontSpec
from h2mare.processing.core.fronts import (
    DEFAULT_N_WORKERS,
    BOA_application,
    FrontProcessor,
    _detect_day,
    apply_boa_fronts,
    boa,
    clear_staging,
    create_base_grid,
    filt3,
    filt5,
    stage_path,
)
from h2mare.utils.spatial import haversine_min_distance_kdtree

REPO = Path(__file__).resolve().parent.parent

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
# BOA_application
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
        result_xr = BOA_application(da, threshold=0.1)
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
        with patch(
            "h2mare.processing.core.fronts.globe.is_land", return_value=all_ocean
        ):
            latlon_arr, sea_mask = create_base_grid(lat, lon)
        assert sea_mask.shape == (3, 3)
        assert latlon_arr.ndim == 2
        assert latlon_arr.shape[1] == 2

    def test_all_ocean_includes_all_pixels(self):
        lat = np.array([30.0, 35.0])
        lon = np.array([-10.0, -5.0])
        all_ocean = np.zeros((2, 2), dtype=bool)
        with patch(
            "h2mare.processing.core.fronts.globe.is_land", return_value=all_ocean
        ):
            latlon_arr, sea_mask = create_base_grid(lat, lon)
        assert latlon_arr.shape[0] == 4

    def test_all_land_excludes_all_pixels(self):
        lat = np.array([30.0, 35.0])
        lon = np.array([-10.0, -5.0])
        all_land = np.ones((2, 2), dtype=bool)
        with patch(
            "h2mare.processing.core.fronts.globe.is_land", return_value=all_land
        ):
            latlon_arr, sea_mask = create_base_grid(lat, lon)
        assert latlon_arr.shape[0] == 0


# ---------------------------------------------------------------------------
# _detect_day
# ---------------------------------------------------------------------------


class TestDetectDay:
    def _day(self, date="2020-01-15"):
        lat = np.array([30.0, 35.0])
        lon = np.array([-10.0, -5.0])
        da = xr.DataArray(
            np.random.rand(1, 2, 2).astype("float32"),
            dims=["time", "lat", "lon"],
            coords={
                "time": pd.date_range(date, periods=1, freq="D"),
                "lat": lat,
                "lon": lon,
            },
        )
        return da, lat, lon

    def _run(self, da, lat, lon, date, fronts=np.array([[32.0, -7.5]])):
        with patch(
            "h2mare.processing.core.fronts.BOA_application", return_value=fronts
        ):
            return _detect_day(
                pd.Timestamp(date),
                da=da,
                threshold=0.4,
                name="sst_fdist",
                latlon1_arr=np.array(
                    [[30.0, -10.0], [30.0, -5.0], [35.0, -10.0], [35.0, -5.0]]
                ),
                lat=lat,
                lon=lon,
                sea_mask=np.ones((2, 2), dtype=bool),
            )

    def test_returns_dataarray_with_correct_name_and_time(self):
        da, lat, lon = self._day()
        result = self._run(da, lat, lon, "2020-01-15")

        assert isinstance(result, xr.DataArray)
        assert result.name == "sst_fdist"
        assert "time" in result.dims
        assert result.sel(time=pd.Timestamp("2020-01-15")).shape == (2, 2)

    def test_sea_pixels_have_finite_distances(self):
        da, lat, lon = self._day("2020-06-01")
        result = self._run(da, lat, lon, "2020-06-01")
        assert np.isfinite(result.values).all()

    def test_distances_are_float32(self):
        """A distance in km has far fewer digits than float64 offers, and the
        staged month is half the size for it."""
        da, lat, lon = self._day()
        assert self._run(da, lat, lon, "2020-01-15").dtype == np.float32

    def test_land_stays_nan(self):
        da, lat, lon = self._day()
        with patch(
            "h2mare.processing.core.fronts.BOA_application",
            return_value=np.array([[32.0, -7.5]]),
        ):
            result = _detect_day(
                pd.Timestamp("2020-01-15"),
                da=da,
                threshold=0.4,
                name="sst_fdist",
                latlon1_arr=np.array([[30.0, -10.0]]),
                lat=lat,
                lon=lon,
                sea_mask=np.array([[True, False], [False, False]]),
            )
        assert np.isfinite(result.values).sum() == 1


# ---------------------------------------------------------------------------
# FrontProcessor
# ---------------------------------------------------------------------------


def _spec(**kw) -> BOAFrontSpec:
    return msgspec.convert({"source": "sst", "threshold": 0.4, **kw}, BOAFrontSpec)


def _ds(dates: list[str] | None = None, name: str = "sst", size: int = 6) -> xr.Dataset:
    """A step-shaped field over open ocean, so every cell is a sea cell."""
    times = pd.DatetimeIndex(pd.to_datetime(dates or ["2020-01-30", "2020-01-31"]))
    lat = np.linspace(30.0, 32.0, size)
    lon = np.linspace(-40.0, -38.0, size)
    field = np.zeros((len(times), size, size), dtype="float32")
    field[:, :, size // 2 :] = 5.0
    return xr.Dataset(
        {name: (["time", "lat", "lon"], field)},
        coords={"time": times, "lat": lat, "lon": lon},
    )


@pytest.mark.usefixtures("serial_pool")
class TestFrontProcessor:
    def _run(self, ds, spec=None, name="sst_fdist", var_key="sst"):
        return FrontProcessor(var_key, name, spec or _spec()).from_dataset(ds)

    def test_returns_a_lazy_layer_over_the_source_grid(self, interim_dir):
        ds = _ds()
        layer = self._run(ds)
        assert layer.name == "sst_fdist"
        assert layer.dims == ("time", "lat", "lon")
        assert layer.shape == ds["sst"].shape
        # Backed by the staging store rather than held in memory: a year of
        # sst at 0.05° is ~3.7 GB of distances.
        assert layer.chunks is not None

    def test_coordinates_come_from_the_source(self, interim_dir):
        """The layer is merged back into the dataset it was detected in, and
        that merge aligns on coordinate values — a time axis decoded back from
        Zarr at another resolution would align to nothing."""
        ds = _ds()
        layer = self._run(ds)
        for coord in ("time", "lat", "lon"):
            assert layer[coord].dtype == ds[coord].dtype
            np.testing.assert_array_equal(layer[coord].values, ds[coord].values)

    def test_distances_are_zero_on_the_front_and_positive_away_from_it(
        self, interim_dir
    ):
        layer = self._run(_ds(size=8)).values
        assert np.isfinite(layer).all()
        assert layer.min() == pytest.approx(0.0)
        assert layer.max() > 0.0

    def test_matches_the_formerly_hardcoded_path(self, interim_dir):
        """The move to config and to staging changed where the threshold comes
        from and where the days are held, not what is computed — so stores
        written before and after it agree, to float32."""
        ds = _ds(["2020-01-31", "2020-02-01"], size=8)
        layer = self._run(ds)

        lat, lon = ds["lat"].values, ds["lon"].values
        latlon1_arr, sea_mask = create_base_grid(lat, lon)
        for i, date in enumerate(pd.DatetimeIndex(ds["time"].values)):
            fronts = BOA_application(ds["sst"].sel(time=date), 0.4)
            expected = np.full((len(lat), len(lon)), np.nan)
            expected[sea_mask] = haversine_min_distance_kdtree(latlon1_arr, fronts)
            np.testing.assert_allclose(layer.isel(time=i).values, expected, rtol=1e-6)

    def test_days_come_from_the_axis_not_a_calendar(self, interim_dir):
        """A source that skipped a day used to be asked for it anyway — the
        days were rebuilt as a date_range between the axis ends — and the
        KeyError surfaced from inside a pool worker."""
        ds = _ds(["2020-01-01", "2020-01-02", "2020-01-04"])
        layer = self._run(ds)
        assert len(layer.time) == 3
        np.testing.assert_array_equal(layer.time.values, ds.time.values)

    def test_months_are_staged_one_at_a_time(self, interim_dir):
        """Peak memory is one month, not one period."""
        ds = _ds(["2020-01-30", "2020-01-31", "2020-02-01", "2020-03-02"])
        with patch.object(
            FrontProcessor,
            "_stage_batch",
            autospec=True,
            side_effect=FrontProcessor._stage_batch,
        ) as staged:
            layer = self._run(ds)
        assert [len(call.args[3]) for call in staged.call_args_list] == [2, 1, 1]
        np.testing.assert_array_equal(layer.time.values, ds.time.values)

    def test_every_day_survives_the_month_boundary(self, interim_dir):
        ds = _ds(["2020-01-31", "2020-02-01"])
        layer = self._run(ds)
        assert np.isfinite(layer.values).all()
        assert len(layer.time) == 2

    def test_source_staging_is_removed_and_the_layer_staging_kept(self, interim_dir):
        """The layer is read from its staging store by whoever writes it, so
        clearing it is the caller's call; the source copy is spent."""
        self._run(_ds())
        assert not stage_path("sst", "sst_fdist_source").exists()
        assert stage_path("sst", "sst_fdist").exists()

    def test_clear_staging_removes_what_a_run_left(self, interim_dir):
        self._run(_ds())
        assert clear_staging("sst") == 1
        assert list(interim_dir.glob(".sst_*")) == []

    def test_clear_staging_leaves_another_var_key_alone(self, interim_dir):
        self._run(_ds())
        assert clear_staging("chl") == 0
        assert stage_path("sst", "sst_fdist").exists()

    def test_unsorted_time_axis_is_refused(self, interim_dir):
        ds = _ds(["2020-01-02", "2020-01-01"])
        with pytest.raises(ValueError, match="unsorted time axis"):
            self._run(ds)

    def test_empty_time_axis_is_refused(self, interim_dir):
        with pytest.raises(ValueError, match="no time steps"):
            self._run(_ds().isel(time=slice(0, 0)))

    def test_worker_count_comes_from_the_spec(self, monkeypatch):
        """On a machine with cores to spare — the cap is the next test."""
        monkeypatch.setattr(os, "cpu_count", lambda: 64)
        assert FrontProcessor("sst", "sst_fdist", _spec()).n_workers == (
            DEFAULT_N_WORKERS
        )
        assert FrontProcessor("sst", "sst_fdist", _spec(n_workers=2)).n_workers == 2

    def test_the_pool_is_capped_to_the_machine(self, monkeypatch):
        """DEFAULT_N_WORKERS is 10; a smaller box gets its own core count, and
        a CI runner with 4 cores no longer starts 10 spawn workers."""
        monkeypatch.setattr(os, "cpu_count", lambda: 4)
        assert FrontProcessor("sst", "sst_fdist", _spec()).n_workers == 4
        assert FrontProcessor("sst", "sst_fdist", _spec(n_workers=2)).n_workers == 2

    def test_the_env_ceiling_caps_the_pool(self, monkeypatch):
        monkeypatch.setattr(os, "cpu_count", lambda: 64)
        monkeypatch.setattr(get_settings(), "MAX_WORKERS", 2)
        assert FrontProcessor("sst", "sst_fdist", _spec(n_workers=8)).n_workers == 2


class TestRealPool:
    """One pass through the actual process pool, which the rest stub out."""

    def test_detects_through_a_worker_process(self, interim_dir):
        ds = _ds(size=6)
        layer = FrontProcessor("sst", "sst_fdist", _spec(n_workers=1)).from_dataset(ds)
        assert np.isfinite(layer.values).all()
        assert len(layer.time) == 2


# ---------------------------------------------------------------------------
# apply_boa_fronts
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("serial_pool")
class TestApplyBOAFronts:
    @pytest.mark.parametrize("fronts", [None, {}])
    def test_nothing_declared_leaves_the_dataset_alone(self, fronts, interim_dir):
        ds = _ds()
        out = apply_boa_fronts(ds.copy(), fronts, "sst")
        xr.testing.assert_identical(out, ds)

    def test_layer_lands_under_its_declared_name(self, interim_dir):
        out = apply_boa_fronts(_ds(), {"sst_fdist": _spec()}, "sst")
        assert "sst_fdist" in out.data_vars

    def test_one_var_key_can_declare_a_layer_per_variable(self, interim_dir):
        """A store holding several fields publishes a front layer for each —
        which is why the entries are keyed by output name."""
        ds = _ds()
        ds["adt"] = ds["sst"] * 2.0
        out = apply_boa_fronts(
            ds,
            {
                "sst_fdist": _spec(source="sst"),
                "adt_fdist": _spec(source="adt", threshold=0.05),
            },
            "sst",
        )
        assert {"sst_fdist", "adt_fdist"} <= set(map(str, out.data_vars))

    def test_missing_source_names_the_entry_and_what_is_there(self, interim_dir):
        with pytest.raises(ValueError, match=r"boa_fronts\.adt_fdist reads 'adt'.*sst"):
            apply_boa_fronts(_ds(), {"adt_fdist": _spec(source="adt")}, "sst")

    def test_source_with_a_depth_axis_is_refused(self, interim_dir):
        """BOA reads one lat×lon field per day; a 3-D slab would be filtered
        along whichever two axes came first."""
        ds = _ds().expand_dims(depth=[0.0, 50.0])
        with pytest.raises(ValueError, match=r"also has \['depth'\]"):
            apply_boa_fronts(ds, {"sst_fdist": _spec()}, "sst")

    def test_source_without_lat_lon_is_refused(self, interim_dir):
        ds = _ds().mean("lat")
        with pytest.raises(ValueError, match="over time/lat/lon"):
            apply_boa_fronts(ds, {"sst_fdist": _spec()}, "sst")


# ---------------------------------------------------------------------------
# BOAFrontSpec
# ---------------------------------------------------------------------------


class TestBOAFrontSpec:
    def test_n_workers_defaults_to_none(self):
        assert _spec().n_workers is None

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"threshold": 0}, "positive"),
            ({"threshold": -0.4}, "positive"),
            ({"n_workers": 0}, "at least 1"),
        ],
    )
    def test_refuses_malformed_entries(self, kw, match):
        with pytest.raises(msgspec.ValidationError, match=match):
            _spec(**kw)

    def test_refuses_misspelt_field(self):
        with pytest.raises(msgspec.ValidationError, match="treshold"):
            msgspec.convert(
                {"source": "sst", "threshold": 0.4, "treshold": 0.5}, BOAFrontSpec
            )

    def test_decodes_inside_a_variable_entry(self):
        cfg = msgspec.convert(
            {
                "variables": {
                    "sst": {
                        "local_folder": "CMEMS_SST",
                        "source_vars": ["analysed_sst"],
                        "dataset_id_rep": "x",
                        "source": "cmems",
                        "boa_fronts": {
                            "sst_fdist": {"source": "sst", "threshold": 0.4}
                        },
                    }
                },
                "secrets": {},
            },
            AppConfig,
        )
        spec = (cfg.variables["sst"].boa_fronts or {})["sst_fdist"]
        assert (spec.source, spec.threshold) == ("sst", 0.4)


# ---------------------------------------------------------------------------
# Config-load checks
# ---------------------------------------------------------------------------


def _entry(**kw) -> dict:
    return {
        "local_folder": "CMEMS_SST",
        "source_vars": ["analysed_sst"],
        "dataset_id_rep": "x",
        "source": "cmems",
        **kw,
    }


def _load(**kw):
    return msgspec.convert(
        {"variables": {"sst": _entry(**kw)}, "secrets": {}}, AppConfig
    )


class TestFrontEntriesAtLoad:
    def test_a_layer_may_not_overwrite_its_own_source(self):
        with pytest.raises(msgspec.ValidationError, match="writes over its own source"):
            _load(boa_fronts={"sst": {"source": "sst", "threshold": 0.4}})

    def test_a_name_derived_vars_writes_is_refused(self):
        with pytest.raises(msgspec.ValidationError, match="both by boa_fronts"):
            _load(
                derived_vars={"sst_fdist": {"op": "rolling_std", "source": "sst"}},
                boa_fronts={"sst_fdist": {"source": "sst", "threshold": 0.4}},
            )

    def test_a_depth_column_name_is_refused(self):
        with pytest.raises(msgspec.ValidationError, match="both by boa_fronts"):
            _load(
                depth_levels={"thetao": [0]},
                boa_fronts={"thetao_0": {"source": "thetao", "threshold": 0.4}},
            )

    def test_an_hourly_store_is_refused(self):
        """Detection reads one field per day; an hourly axis offers 24."""
        with pytest.raises(msgspec.ValidationError, match="needs a daily store"):
            _load(
                time_step="hourly",
                boa_fronts={"sst_fdist": {"source": "sst", "threshold": 0.4}},
            )

    def test_a_daily_store_loads(self):
        cfg = _load(boa_fronts={"sst_fdist": {"source": "sst", "threshold": 0.4}})
        assert cfg.variables["sst"].boa_fronts is not None


class TestRepoConfig:
    """Guard the tracked config.yaml. Both thresholds used to be hardcoded in
    fronts.py, so dropping an entry would silently stop a layer being written,
    and changing one would silently rewrite the store's meaning."""

    @pytest.fixture(scope="class")
    def config(self) -> AppConfig:
        raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
        return msgspec.convert(
            {"variables": raw["variables"], "secrets": {}}, AppConfig
        )

    @pytest.mark.parametrize(
        "var_key, expected",
        [
            ("sst", {"sst_fdist": ("sst", 0.4)}),
            ("chl", {"chl_fdist": ("chl", 0.06)}),
        ],
    )
    def test_declares_the_formerly_hardcoded_thresholds(
        self, config, var_key, expected
    ):
        fronts = config.variables[var_key].boa_fronts or {}
        assert {n: (s.source, s.threshold) for n, s in fronts.items()} == expected

    @pytest.mark.parametrize("var_key", ["sst", "chl"])
    def test_the_layer_is_published(self, config, var_key):
        """A layer absent from compiled_vars never reaches h2ds or Parquet."""
        entry = config.variables[var_key]
        assert set(entry.boa_fronts or {}) <= set(entry.compiled_vars or [])
