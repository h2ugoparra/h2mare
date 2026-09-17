"""Tests for storage/xarray_helpers.py."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.storage.xarray_helpers import (
    apply_cf_attrs,
    chunk_dataset,
    convert360_180,
    drop_source_encoding_attrs,
    get_dataset_encoding,
    rename_dims,
    snap_grid_coords,
    unified_time_chunk,
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
    def test_spatial_dims_below_tile_stay_full_size(self):
        """lat/lon smaller than spatial_chunk are kept whole (single chunk)."""
        ds = _make_ds(n_time=10, n_lat=50, n_lon=60)
        result = chunk_dataset(ds, target_mb=32, spatial_chunk=256)
        assert result.chunks["lat"] == (50,)
        assert result.chunks["lon"] == (60,)

    def test_spatial_dims_above_tile_are_tiled(self):
        """lat/lon larger than spatial_chunk are split into tiles of that size."""
        ds = _make_ds(n_time=10, n_lat=50, n_lon=60)
        result = chunk_dataset(ds, target_mb=32, spatial_chunk=20)
        # 50 -> 20,20,10 ; 60 -> 20,20,20
        assert result.chunks["lat"] == (20, 20, 10)
        assert result.chunks["lon"] == (20, 20, 20)

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

    def test_depth_chunked_to_1_even_when_payload_fits_target(self):
        """Levels are indexed into, not read together: one per chunk even when
        the whole column would fit. A 23-level store written with depth in one
        chunk decompressed every level to read one, 10x slower."""
        times = pd.date_range("2020-01-01", periods=10, freq="D")
        # 3 × 10 × 10 × 4 bytes = 1 200 bytes ≪ 32 MB
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
        assert result.chunks["depth"] == (1,) * n_depth
        # The freed budget goes to time, as it would with depth split for size.
        assert result.chunks["time"] == (10,)

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


class TestChunkDatasetMapLayout:
    def test_pins_time_to_one_and_keeps_space_full(self):
        """map layout with map_time_chunk=1: one timestep per chunk, space full."""
        ds = _make_ds(n_time=10, n_lat=50, n_lon=60)
        result = chunk_dataset(ds, layout="map", map_time_chunk=1)
        assert result.chunks["time"] == (1,) * 10
        assert result.chunks["lat"] == (50,)
        assert result.chunks["lon"] == (60,)

    def test_default_map_time_chunk_is_14(self):
        """The map layout defaults to a 14-day time block."""
        ds = _make_ds(n_time=30, n_lat=50, n_lon=60)
        result = chunk_dataset(ds, layout="map")
        # 30 -> 14,14,2
        assert result.chunks["time"] == (14, 14, 2)
        assert result.chunks["lat"] == (50,)
        assert result.chunks["lon"] == (60,)

    def test_map_time_chunk_sets_time_block(self):
        """map_time_chunk > 1 produces a small time block, space still full."""
        ds = _make_ds(n_time=10, n_lat=50, n_lon=60)
        result = chunk_dataset(ds, layout="map", map_time_chunk=4)
        # 10 -> 4,4,2
        assert result.chunks["time"] == (4, 4, 2)
        assert result.chunks["lat"] == (50,)
        assert result.chunks["lon"] == (60,)

    def test_map_time_chunk_capped_at_time_size(self):
        """A map_time_chunk larger than the time axis is capped to it."""
        ds = _make_ds(n_time=5, n_lat=4, n_lon=4)
        result = chunk_dataset(ds, layout="map", map_time_chunk=100)
        assert result.chunks["time"] == (5,)

    def test_collapses_depth_to_one(self):
        """Non-spatial, non-time dims (depth) collapse to 1 in map layout."""
        times = pd.date_range("2020-01-01", periods=6, freq="D")
        n_depth = 5
        data = np.ones((6, n_depth, 10, 10), dtype=np.float32)
        ds = xr.Dataset(
            {"thetao": (["time", "depth", "lat", "lon"], data)},
            coords={
                "time": times,
                "depth": np.arange(n_depth, dtype=np.float32),
                "lat": np.arange(10, dtype=np.float32),
                "lon": np.arange(10, dtype=np.float32),
            },
        )
        result = chunk_dataset(ds, layout="map", map_time_chunk=1)
        assert result.chunks["depth"] == (1,) * n_depth
        assert result.chunks["lat"] == (10,)
        assert result.chunks["lon"] == (10,)
        assert result.chunks["time"] == (1,) * 6

    def test_tiles_space_when_single_timestep_exceeds_target(self):
        """A single full-grid timestep above target_mb is tiled down to fit."""
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        # one timestep = 1000*1000*4 = ~3.8 MB > target_mb=1 → spatial must tile
        data = np.ones((3, 1000, 1000), dtype=np.float32)
        ds = xr.Dataset(
            {"sst": (["time", "lat", "lon"], data)},
            coords={
                "time": times,
                "lat": np.linspace(0, 10, 1000),
                "lon": np.linspace(0, 10, 1000),
            },
        )
        result = chunk_dataset(ds, layout="map", map_time_chunk=1, target_mb=1)
        assert result.chunks["time"] == (1,) * 3
        # spatial tiled below the full 1000 so the per-chunk payload fits target
        assert max(result.chunks["lat"]) < 1000
        assert max(result.chunks["lon"]) < 1000

    def test_warns_when_no_spatial_dims(self):
        """map layout on non-gridded data warns and only sets the time chunk."""
        from loguru import logger

        times = pd.date_range("2020-01-01", periods=4, freq="D")
        ds = xr.Dataset(
            {"profile": (["time", "band"], np.ones((4, 3), dtype=np.float32))},
            coords={"time": times, "band": np.arange(3)},
        )

        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            result = chunk_dataset(ds, layout="map", map_time_chunk=1)
        finally:
            logger.remove(sink_id)

        assert any("no spatial dims" in m for m in messages)
        # Non-spatial, non-time dim collapses to 1; time still pinned.
        assert result.chunks["band"] == (1,) * 3
        assert result.chunks["time"] == (1,) * 4

    def test_default_layout_is_timeseries(self):
        """Default (no layout arg) keeps the time-contiguous extraction layout."""
        ds = _make_ds(n_time=365, n_lat=50, n_lon=60)
        result = chunk_dataset(ds)
        # time-contiguous: a single time chunk spans many days, not 1.
        assert result.chunks["time"][0] > 1

    def test_logs_layout_and_chunk_size(self):
        """chunk_dataset logs the layout, chunk shape and size for any dataset."""
        from loguru import logger

        ds = _make_ds(n_time=30, n_lat=50, n_lon=60)
        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
        try:
            chunk_dataset(ds, layout="map")
        finally:
            logger.remove(sink_id)

        assert any("layout='map'" in m and "MB" in m for m in messages)


class TestSnapGridCoords:
    def test_noise_drift_collapses_to_rounded_grid(self):
        """Labels off the grid by sub-4dp float noise snap to the rounded grid."""
        ds = xr.Dataset(
            coords={"lon": [-14.91671, -14.83329], "lat": [40.00001, 40.08334]}
        )
        out = snap_grid_coords(ds)
        np.testing.assert_array_equal(out.lon.values, [-14.9167, -14.8333])
        np.testing.assert_array_equal(out.lat.values, [40.0, 40.0833])

    def test_two_noise_drifted_grids_become_identical(self):
        """The core fix: two grids 1.5e-5 apart snap to bit-identical labels."""
        a = xr.Dataset(coords={"lon": [10.000004, 10.083337]})
        b = xr.Dataset(coords={"lon": [10.000019, 10.083322]})
        assert not np.array_equal(a.lon.values, b.lon.values)
        assert np.array_equal(
            snap_grid_coords(a).lon.values, snap_grid_coords(b).lon.values
        )

    def test_finer_than_4dp_grid_left_unchanged(self):
        """A genuinely finer grid (cells <1e-4 apart) is refused, never merged."""
        ds = xr.Dataset(coords={"lon": [1.00001, 1.00002]})
        out = snap_grid_coords(ds)
        np.testing.assert_array_equal(out.lon.values, ds.lon.values)

    def test_clean_grid_is_noop(self):
        """A grid already on the rounded grid (e.g. 0.25°) is returned unchanged."""
        ds = xr.Dataset(
            coords={"lon": [-10.0, -9.75, -9.5], "lat": [30.0, 30.25, 30.5]}
        )
        out = snap_grid_coords(ds)
        np.testing.assert_array_equal(out.lon.values, ds.lon.values)
        np.testing.assert_array_equal(out.lat.values, ds.lat.values)

    def test_missing_coords_are_ignored(self):
        ds = xr.Dataset({"x": (["a"], np.ones(3))}, coords={"a": [0, 1, 2]})
        out = snap_grid_coords(ds)
        assert "lon" not in out.coords and "lat" not in out.coords


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


class TestDropSourceEncodingAttrs:
    """
    These attributes were true of the source file and stopped being true when
    the data was regridded and re-encoded, but nothing removed them — so they
    reached h2ds asserting the wrong grid, the wrong units and packing bounds
    read as physical limits.
    """

    @staticmethod
    def _ds() -> xr.Dataset:
        da = xr.DataArray(
            np.ones((2, 2)),
            dims=["lat", "lon"],
            coords={"lat": [0.0, 0.25], "lon": [0.0, 0.25]},
            attrs={
                # source grid, wrong once regridded
                "GRIB_Nx": 181,
                "GRIB_Ny": 141,
                "GRIB_iDirectionIncrementInDegrees": 0.5,
                "GRIB_missingValue": 3.4028234663852886e38,
                "GRIB_units": "N m**-2",
                # source packing, reads as a physical range and is not
                "valid_min": -32766,
                "valid_max": 21306,
                # genuinely about the quantity
                "standard_name": "sea_water_potential_temperature",
                "long_name": "Potential temperature",
                "units": "degrees_C",
                "cell_methods": "area: mean",
                "unit_long": "Degrees Celsius",
            },
        )
        return xr.Dataset({"thetao_100": da})

    def test_grib_attrs_are_gone(self):
        out = drop_source_encoding_attrs(self._ds())
        assert not [k for k in out["thetao_100"].attrs if k.startswith("GRIB_")]

    def test_packing_bounds_are_gone(self):
        """[-32766, 21306] 'degrees_C' against real values of [-2.3, 28.2]."""
        attrs = drop_source_encoding_attrs(self._ds())["thetao_100"].attrs
        assert "valid_min" not in attrs
        assert "valid_max" not in attrs

    def test_attributes_describing_the_quantity_survive(self):
        attrs = drop_source_encoding_attrs(self._ds())["thetao_100"].attrs
        assert set(attrs) == {
            "standard_name",
            "long_name",
            "units",
            "cell_methods",
            "unit_long",
        }

    def test_a_contradicting_grib_units_cannot_outlive_units(self):
        """n_upwell_events_7d carried GRIB_units='N m**-2' beside units='count'."""
        attrs = drop_source_encoding_attrs(self._ds())["thetao_100"].attrs
        assert "GRIB_units" not in attrs
        assert attrs["units"] == "degrees_C"

    def test_coordinates_are_left_alone(self):
        ds = self._ds()
        ds["lat"].attrs["valid_min"] = -90
        assert drop_source_encoding_attrs(ds)["lat"].attrs["valid_min"] == -90

    def test_a_dataset_with_nothing_to_drop_is_unchanged(self):
        ds = xr.Dataset({"sst": xr.DataArray([1.0], dims="x", attrs={"units": "degC"})})
        assert drop_source_encoding_attrs(ds)["sst"].attrs == {"units": "degC"}

    def test_grib_attrs_survive_when_drop_grib_is_off(self):
        """
        The native path keeps them: a CDS store is written at ERA5's own grid,
        so GRIB_Nx/Ny still describe it, and hourly_radiation deliberately
        rewrites GRIB_units/GRIB_stepType there to stop the accumulation being
        differenced twice. The packing bounds go either way.
        """
        attrs = drop_source_encoding_attrs(self._ds(), drop_grib=False)[
            "thetao_100"
        ].attrs
        assert attrs["GRIB_Nx"] == 181
        assert attrs["GRIB_units"] == "N m**-2"
        assert "valid_min" not in attrs
        assert "valid_max" not in attrs


class TestApplyCfAttrs:
    """
    The single place both write paths take their metadata from.

    Without the coordinate half, ``rio.clip`` cannot resolve lon/lat — it falls
    back to CF attributes when the dims are not named x/y, and finding none it
    clipped geometry extraction to nothing but NaN.

    Run against a stub table rather than the repo's config.yaml: settings
    resolve through ``H2MARE_ROOT``, so what ``get_settings`` returns depends on
    the machine, and on CI there is no deployed config at all. That the real
    table is CF-valid is a question about the table, and belongs to
    ``test_cf_compliance.py``; these are about the mechanics.
    """

    @pytest.fixture(autouse=True)
    def _stub_settings(self, monkeypatch):
        table = {
            "sst": {
                "long_name": "Analysed sea surface temperature",
                "standard_name": "sea_surface_foundation_temperature",
                "units": "degree_C",
            },
            "gke": {"long_name": "Geostrophic Kinetic Energy", "units": "m2.s-2"},
            "msl": {"units": "hPa", "cell_methods": "time: mean"},
            "swh": {"units": "m"},
            "thetao": {"units": "degree_C"},
        }
        overrides = {"atm-instante": {"msl": {"units": "Pa", "cell_methods": None}}}

        class _Stub:
            @staticmethod
            def get_var_info(name):
                return table.get(name, {})

            native_attr_overrides = overrides

        monkeypatch.setattr("h2mare.config.get_settings", lambda: _Stub())

    @staticmethod
    def _ds(var: str = "sst", **attrs) -> xr.Dataset:
        times = pd.date_range("2020-01-01", periods=2, freq="D")
        return xr.Dataset(
            {
                var: (
                    ["time", "lat", "lon"],
                    np.ones((2, 2, 2), dtype=np.float32),
                    attrs,
                )
            },
            coords={"time": times, "lat": [0.0, 0.25], "lon": [0.0, 0.25]},
        )

    def test_spatial_coords_get_cf_attributes(self):
        ds = apply_cf_attrs(self._ds())
        assert ds["lon"].attrs["standard_name"] == "longitude"
        assert ds["lon"].attrs["units"] == "degrees_east"
        assert ds["lon"].attrs["axis"] == "X"
        assert ds["lat"].attrs["standard_name"] == "latitude"
        assert ds["lat"].attrs["units"] == "degrees_north"
        assert ds["lat"].attrs["axis"] == "Y"

    def test_time_is_labelled_but_gets_no_units(self):
        """
        Regression: xarray writes time units/calendar into .encoding at to_zarr,
        and a units in .attrs as well makes that write raise rather than merely
        disagree. The axis label is still worth setting.
        """
        ds = apply_cf_attrs(self._ds())
        assert ds["time"].attrs["standard_name"] == "time"
        assert ds["time"].attrs["axis"] == "T"
        assert "units" not in ds["time"].attrs
        assert "calendar" not in ds["time"].attrs

    def test_depth_is_positive_down_when_present(self):
        ds = self._ds("thetao").expand_dims(depth=[0.0, 100.0])
        out = apply_cf_attrs(ds)
        assert out["depth"].attrs["positive"] == "down"
        assert out["depth"].attrs["axis"] == "Z"

    def test_variable_attrs_come_from_config(self):
        ds = apply_cf_attrs(self._ds("sst"))
        assert ds["sst"].attrs["units"] == "degree_C"
        assert ds["sst"].attrs["standard_name"] == "sea_surface_foundation_temperature"

    def test_a_derived_variable_that_lost_its_attrs_gets_them_back(self):
        """
        gke is (ugos**2 + vgos**2)/2 and sst is a subtraction; xarray drops attrs
        on arithmetic, so both reached the native store carrying none at all.
        """
        ds = apply_cf_attrs(self._ds("gke"), native_var_key="ssh")
        assert ds["gke"].attrs["units"] == "m2.s-2"
        assert ds["gke"].attrs["long_name"] == "Geostrophic Kinetic Energy"

    def test_native_override_replaces_the_published_units(self):
        """msl is Pa in its own store and hPa only after the compile converts."""
        compiled = apply_cf_attrs(self._ds("msl"))
        native = apply_cf_attrs(self._ds("msl"), native_var_key="atm-instante")
        assert compiled["msl"].attrs["units"] == "hPa"
        assert native["msl"].attrs["units"] == "Pa"

    def test_a_null_override_removes_the_attribute(self):
        """
        An hourly store holds instantaneous values, so the cell_methods naming a
        daily mean does not describe it and has to come off rather than change.
        """
        compiled = apply_cf_attrs(self._ds("msl"))
        native = apply_cf_attrs(self._ds("msl"), native_var_key="atm-instante")
        assert compiled["msl"].attrs["cell_methods"] == "time: mean"
        assert "cell_methods" not in native["msl"].attrs

    def test_overrides_apply_only_to_their_own_var_key(self):
        """msl's Pa must not leak into a store that is not atm-instante."""
        ds = apply_cf_attrs(self._ds("msl"), native_var_key="waves")
        assert ds["msl"].attrs["units"] == "hPa"

    def test_compiled_path_drops_grib_attrs(self):
        ds = apply_cf_attrs(self._ds("swh", GRIB_Nx=181, GRIB_units="m"))
        assert not [k for k in ds["swh"].attrs if k.startswith("GRIB_")]

    def test_native_path_keeps_grib_attrs(self):
        ds = apply_cf_attrs(
            self._ds("swh", GRIB_paramId=140229), native_var_key="waves"
        )
        assert ds["swh"].attrs["GRIB_paramId"] == 140229

    def test_an_unconfigured_variable_is_left_alone(self):
        ds = apply_cf_attrs(self._ds("not_in_config", units="widgets"))
        assert ds["not_in_config"].attrs["units"] == "widgets"

    def test_rioxarray_can_resolve_lon_lat_once_they_are_labelled(self):
        """
        The reason the coordinate half exists. rio resolves spatial dims by name
        and falls back to CF attributes when they are not x/y — so on a store
        whose lon/lat carry nothing, it cannot find them at all, which is what
        made geometry extraction clip to NaN against h2ds and the CDS stores.
        """
        import rioxarray  # noqa: F401  (registers the .rio accessor)
        from rioxarray.exceptions import MissingSpatialDimensionError

        with pytest.raises(MissingSpatialDimensionError):
            _ = self._ds().rio.x_dim

        labelled = apply_cf_attrs(self._ds())
        assert (labelled.rio.x_dim, labelled.rio.y_dim) == ("lon", "lat")

    def test_a_written_store_keeps_time_units_in_encoding_only(self, tmp_path):
        """
        End-to-end guard on the .attrs/.encoding split: xarray raises on write if
        both carry time units, and the append path is where that would surface
        rather than the create.
        """
        ds = apply_cf_attrs(self._ds())
        path = tmp_path / "t.zarr"
        ds.to_zarr(path)
        ds.assign_coords(time=pd.date_range("2020-01-03", periods=2, freq="D")).to_zarr(
            path, append_dim="time"
        )

        back = xr.open_zarr(path, consolidated=False)
        assert back.sizes["time"] == 4
        assert "units" not in back["time"].attrs
        assert back["time"].encoding["calendar"] == "proleptic_gregorian"
        assert back["lat"].attrs["units"] == "degrees_north"
