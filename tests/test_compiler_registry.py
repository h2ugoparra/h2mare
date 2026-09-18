"""Tests for processing/compiler_registry.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.models import TimeStep
from h2mare.processing import compiler_registry
from h2mare.processing.compiler_registry import (
    COMPILE_PROCESSORS,
    _compile_atm_accum_avg,
    _compile_bathy,
    _compile_depth_var,
    _compile_moon,
    _compile_sst,
    compile_default,
)
from h2mare.processing.core.cds import _EKMAN_WARMUP_DAYS
from h2mare.types import BBox, DateRange

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def _make_grid() -> xr.Dataset:
    """Tiny 2×2 lat/lon grid that matches what Compiler.base_grid looks like."""
    return xr.Dataset(
        coords={
            "lat": xr.DataArray([30.0, 30.25], dims="lat"),
            "lon": xr.DataArray([-10.0, -9.75], dims="lon"),
        }
    )


def _make_compiler(tmp_path: Path, *, bbox=(-10, 25, 15, 55)) -> MagicMock:
    """Build a minimal Compiler-like mock with the attributes processors need."""
    compiler = MagicMock()
    compiler.base_grid = _make_grid()
    compiler.bbox = BBox(*bbox)
    compiler.var_config.bbox = bbox
    compiler.remote_store_root = tmp_path
    compiler.app_config.variables = {}
    return compiler


def _make_catalog(ds: xr.Dataset | None = None) -> MagicMock:
    """Build a ZarrCatalog mock that returns *ds* from open_dataset."""
    catalog = MagicMock()
    if ds is None:
        catalog.open_dataset.side_effect = FileNotFoundError("no data")
    else:
        catalog.open_dataset.return_value = ds
    return catalog


def _daily_ds(var: str, dates: pd.DatetimeIndex) -> xr.Dataset:
    """Minimal (time, lat, lon) dataset for testing interp_like paths."""
    data = np.ones((len(dates), 2, 2), dtype="float32")
    return xr.Dataset(
        {
            var: xr.DataArray(
                data,
                dims=["time", "lat", "lon"],
                coords={"time": dates, "lat": [30.0, 30.25], "lon": [-10.0, -9.75]},
            )
        },
    )


_DR = DateRange("2020-01-01", "2020-01-03")
_DATES = pd.date_range("2020-01-01", "2020-01-03", freq="D")


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------


class TestCompileProcessorsRegistry:
    def test_contains_bathy(self):
        assert "bathy" in COMPILE_PROCESSORS

    def test_contains_moon(self):
        assert "moon" in COMPILE_PROCESSORS

    def test_contains_atm_accum_avg(self):
        assert "atm-accum-avg" in COMPILE_PROCESSORS

    def test_contains_sst(self):
        assert "sst" in COMPILE_PROCESSORS

    def test_all_values_are_callable(self):
        for key, fn in COMPILE_PROCESSORS.items():
            assert callable(fn), f"COMPILE_PROCESSORS[{key!r}] is not callable"


# ---------------------------------------------------------------------------
# _compile_bathy
# ---------------------------------------------------------------------------


class TestCompileBathy:
    def test_raises_when_data_file_is_none(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables["bathy"] = MagicMock(data_file=None)

        with pytest.raises(ValueError, match="data_file"):
            _compile_bathy(compiler, None, _DR)

    def test_opens_file_at_expected_path(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        bathy_cfg = MagicMock()
        bathy_cfg.data_file = "bathy.nc"
        bathy_cfg.local_folder = "bathy"
        # A MagicMock invents any attribute asked of it, so this has to be set
        # explicitly: without it bathy would appear to declare a store_root of
        # its own and resolve under a mock instead of remote_store_root.
        bathy_cfg.store_root = None
        compiler.app_config.variables["bathy"] = bathy_cfg

        fake_ds = _daily_ds("elevation", _DATES).isel(time=0).drop_vars("time")
        fake_ds = fake_ds.rename({"lat": "lat", "lon": "lon"})

        with patch(
            "h2mare.processing.compiler_registry.xr.open_dataset", return_value=fake_ds
        ) as mock_open:
            _compile_bathy(compiler, None, _DR)

        called_path = mock_open.call_args[0][0]
        assert called_path == tmp_path / "bathy" / "bathy.nc"


# ---------------------------------------------------------------------------
# _compile_moon
# ---------------------------------------------------------------------------


class TestCompileMoon:
    def test_returns_dataset_with_moon_phase(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        result = _compile_moon(compiler, None, _DR)
        assert isinstance(result, xr.Dataset)
        assert "moon_phase" in result.data_vars

    def test_time_dimension_matches_date_range(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        result = _compile_moon(compiler, None, _DR)
        assert len(result.time) == 3  # 2020-01-01 to 2020-01-03

    def test_spatial_dimensions_match_base_grid(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        result = _compile_moon(compiler, None, _DR)
        assert list(result.lat.values) == list(compiler.base_grid.lat.values)
        assert list(result.lon.values) == list(compiler.base_grid.lon.values)

    def test_phase_values_are_in_valid_range(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        result = _compile_moon(compiler, None, _DR)
        vals = result["moon_phase"].values
        assert float(vals.min()) >= 0.0
        assert float(vals.max()) <= 100.0


# ---------------------------------------------------------------------------
# _compile_depth_var (o2)
# ---------------------------------------------------------------------------


class TestCompileO2:
    _depths = [0, 100, 500, 1000]

    def _make_o2_ds(self) -> xr.Dataset:
        data = np.ones((3, len(self._depths), 2, 2), dtype="float32")
        return xr.Dataset(
            {
                "o2": xr.DataArray(
                    data,
                    dims=["time", "depth", "lat", "lon"],
                    coords={
                        "time": _DATES,
                        "depth": self._depths,
                        "lat": [30.0, 30.25],
                        "lon": [-10.0, -9.75],
                    },
                )
            }
        )

    def _make_o2_compiler(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables["o2"] = SimpleNamespace(
            compile_depth_slices=self._depths
        )
        return compiler

    def _make_o2_catalog(self, ds=None):
        catalog = _make_catalog(ds)
        catalog.var_key = "o2"
        return catalog

    def test_returns_none_when_data_missing(self, tmp_path):
        result = _compile_depth_var(
            self._make_o2_compiler(tmp_path), self._make_o2_catalog(None), _DR
        )
        assert result is None

    def test_missing_compile_depth_slices_raises_a_config_error(self, tmp_path):
        """
        A 3-D variable declaring no depth levels is a config error and must say
        so. This was an `assert`, which `python -O` strips — with it gone, None
        reached ds.sel(depth=None) and surfaced as a TypeError somewhere
        unrelated to the config key that actually needed setting.
        """
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables["o2"] = SimpleNamespace(compile_depth_slices=None)

        with pytest.raises(ValueError) as excinfo:
            _compile_depth_var(compiler, self._make_o2_catalog(self._make_o2_ds()), _DR)

        msg = str(excinfo.value)
        assert "o2" in msg
        assert "compile_depth_slices" in msg, "the message does not name the config key"

    def test_returns_one_variable_per_depth(self, tmp_path):
        result = _compile_depth_var(
            self._make_o2_compiler(tmp_path),
            self._make_o2_catalog(self._make_o2_ds()),
            _DR,
        )
        assert result is not None
        assert "o2_0" in result.data_vars
        assert "o2_100" in result.data_vars
        assert "o2_500" in result.data_vars
        assert "o2_1000" in result.data_vars

    def test_depth_dim_dropped_in_output(self, tmp_path):
        result = _compile_depth_var(
            self._make_o2_compiler(tmp_path),
            self._make_o2_catalog(self._make_o2_ds()),
            _DR,
        )
        assert result is not None
        for var in result.data_vars:
            assert "depth" not in result[var].dims


# ---------------------------------------------------------------------------
# _compile_depth_var (thetao)
# ---------------------------------------------------------------------------


class TestCompileThetao:
    _depths = [100, 200, 500, 1000]

    def _make_thetao_ds(self) -> xr.Dataset:
        data = np.ones((3, len(self._depths), 2, 2), dtype="float32")
        return xr.Dataset(
            {
                "thetao": xr.DataArray(
                    data,
                    dims=["time", "depth", "lat", "lon"],
                    coords={
                        "time": _DATES,
                        "depth": self._depths,
                        "lat": [30.0, 30.25],
                        "lon": [-10.0, -9.75],
                    },
                )
            }
        )

    def _make_thetao_compiler(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables["thetao"] = SimpleNamespace(
            compile_depth_slices=self._depths
        )
        return compiler

    def _make_thetao_catalog(self, ds=None):
        catalog = _make_catalog(ds)
        catalog.var_key = "thetao"
        return catalog

    def test_returns_none_when_data_missing(self, tmp_path):
        result = _compile_depth_var(
            self._make_thetao_compiler(tmp_path), self._make_thetao_catalog(None), _DR
        )
        assert result is None

    def test_returns_one_variable_per_depth(self, tmp_path):
        result = _compile_depth_var(
            self._make_thetao_compiler(tmp_path),
            self._make_thetao_catalog(self._make_thetao_ds()),
            _DR,
        )
        assert result is not None
        assert "thetao_100" in result.data_vars
        assert "thetao_200" in result.data_vars
        assert "thetao_500" in result.data_vars
        assert "thetao_1000" in result.data_vars

    def test_depth_dim_dropped_in_output(self, tmp_path):
        result = _compile_depth_var(
            self._make_thetao_compiler(tmp_path),
            self._make_thetao_catalog(self._make_thetao_ds()),
            _DR,
        )
        assert result is not None
        for var in result.data_vars:
            assert "depth" not in result[var].dims


# ---------------------------------------------------------------------------
# _compile_depth_var — per-variable depth_levels
# ---------------------------------------------------------------------------


_STORE_DEPTHS = [0.5, 47.4, 109.7]


def _mixed_store() -> xr.Dataset:
    """dyn_rep-like store: 3-D thetao/uo beside 2-D zos, none named like the key."""
    coords = {
        "time": _DATES,
        "depth": _STORE_DEPTHS,
        "lat": [30.0, 30.25],
        "lon": [-10.0, -9.75],
    }
    by_depth = np.arange(len(_STORE_DEPTHS), dtype="float32")[None, :, None, None]
    cube = np.broadcast_to(by_depth, (3, len(_STORE_DEPTHS), 2, 2))
    dims4 = ["time", "depth", "lat", "lon"]
    return xr.Dataset(
        {
            "thetao": xr.DataArray(cube.copy(), dims=dims4, coords=coords),
            "uo": xr.DataArray(cube.copy() * 10, dims=dims4, coords=coords),
            "zos": xr.DataArray(
                np.ones((3, 2, 2), dtype="float32"),
                dims=["time", "lat", "lon"],
                coords={k: coords[k] for k in ("time", "lat", "lon")},
            ),
        }
    )


class TestCompileDepthLevels:
    def _run(self, tmp_path, ds, **cfg):
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables["dyn_rep"] = SimpleNamespace(**cfg)
        catalog = _make_catalog(ds)
        catalog.var_key = "dyn_rep"
        return _compile_depth_var(compiler, catalog, _DR)

    def test_mixed_store_slices_each_variable_at_its_own_levels(self, tmp_path):
        result = self._run(
            tmp_path, _mixed_store(), depth_levels={"thetao": [0, 50], "uo": [0]}
        )
        assert sorted(result.data_vars) == ["thetao_0", "thetao_50", "uo_0", "zos"]
        assert "depth" not in result.dims and "depth" not in result.coords

    def test_levels_match_nearest_and_keep_the_requested_name(self, tmp_path):
        result = self._run(
            tmp_path, _mixed_store(), depth_levels={"thetao": [0, 50, 1000], "uo": [0]}
        )
        # Values encode the store's depth index: 0 → 0.5 m, 50 → 47.4 m,
        # 1000 → the deepest level the store has.
        assert float(result["thetao_0"].isel(time=0, lat=0, lon=0)) == 0
        assert float(result["thetao_50"].isel(time=0, lat=0, lon=0)) == 1
        assert float(result["thetao_1000"].isel(time=0, lat=0, lon=0)) == 2
        assert float(result["uo_0"].isel(time=0, lat=0, lon=0)) == 0

    def test_three_d_variable_without_levels_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="'uo' has a depth axis"):
            self._run(tmp_path, _mixed_store(), depth_levels={"thetao": [0]})

    def test_levels_for_a_variable_the_store_lacks_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match=r"\['so'\]"):
            self._run(
                tmp_path,
                _mixed_store(),
                depth_levels={"thetao": [0], "uo": [0], "so": [0]},
            )

    def test_levels_for_a_two_d_variable_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="'zos', which has no depth axis"):
            self._run(
                tmp_path,
                _mixed_store(),
                depth_levels={"thetao": [0], "uo": [0], "zos": [0]},
            )

    def test_extract_only_levels_do_not_reach_compile(self, tmp_path):
        result = self._run(
            tmp_path,
            _mixed_store(),
            depth_levels={"thetao": [0], "uo": [0]},
            extract_depth_levels={"thetao": [50]},
        )
        assert "thetao_0" in result.data_vars
        assert "thetao_50" not in result.data_vars

    def test_hourly_store_with_levels_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="hourly"):
            self._run(
                tmp_path,
                _mixed_store(),
                depth_levels={"thetao": [0], "uo": [0]},
                time_step=TimeStep.HOURLY,
            )


class TestDepthDispatch:
    """Depth handling is chosen by config, so a new 3-D var_key needs no entry."""

    def _dispatch(self, monkeypatch, var_config) -> str:
        from h2mare.processing.compiler import Compiler

        monkeypatch.setattr(compiler_registry, "_compile_depth_var", lambda *a: "depth")
        monkeypatch.setattr(compiler_registry, "compile_default", lambda *a: "default")
        fake = SimpleNamespace(
            app_config=SimpleNamespace(variables={"dyn_rep": var_config}),
            _catalog_cache={},
            _catalog_for=lambda var_key, auto_refresh: MagicMock(),
            _has_overlap=lambda *a: True,
        )
        return Compiler._process_variable(fake, "dyn_rep", _DR)

    def test_depth_levels_route_to_the_depth_processor(self, monkeypatch):
        cfg = SimpleNamespace(depth_levels={"thetao": [0]})
        assert self._dispatch(monkeypatch, cfg) == "depth"

    def test_older_list_form_routes_to_the_depth_processor(self, monkeypatch):
        cfg = SimpleNamespace(compile_depth_slices=[100])
        assert self._dispatch(monkeypatch, cfg) == "depth"

    def test_no_levels_route_to_the_default(self, monkeypatch):
        assert self._dispatch(monkeypatch, SimpleNamespace()) == "default"

    def test_depth_var_keys_are_not_registered_by_name(self):
        assert "o2" not in COMPILE_PROCESSORS
        assert "thetao" not in COMPILE_PROCESSORS


class TestCompileDefaultRefusesDepth:
    def test_store_with_a_depth_axis_is_refused(self, tmp_path):
        """
        Regression: an unregistered 3-D var_key went through compile_default,
        which kept the depth axis and carried it into the 2-D h2ds.
        """
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables["dyn_rep"] = SimpleNamespace()
        catalog = _make_catalog(_mixed_store())
        catalog.var_key = "dyn_rep"

        with pytest.raises(ValueError, match="depth axis but declares no depth_levels"):
            compile_default(compiler, catalog, _DR)


# ---------------------------------------------------------------------------
# _compile_atm_accum_avg
# ---------------------------------------------------------------------------


class TestCompileAtmAccumAvg:
    def _make_atm_ds(self) -> xr.Dataset:
        base = _daily_ds("precip", _DATES)
        base["dayofyear"] = xr.DataArray([1, 2, 3], dims="time")
        base["month"] = xr.DataArray([1, 1, 1], dims="time")
        base["quantile"] = xr.DataArray([0.5, 0.5, 0.5], dims="time")
        return base

    def test_returns_none_when_data_missing(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(None)
        result = _compile_atm_accum_avg(compiler, catalog, _DR)
        assert result is None

    def test_drops_auxiliary_variables(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(self._make_atm_ds())
        result = _compile_atm_accum_avg(compiler, catalog, _DR)
        assert result is not None
        assert "dayofyear" not in result
        assert "month" not in result
        assert "quantile" not in result

    def test_retains_data_variable(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(self._make_atm_ds())
        result = _compile_atm_accum_avg(compiler, catalog, _DR)
        assert result is not None
        assert "precip" in result.data_vars

    def test_daily_store_reads_exactly_the_requested_window(self, tmp_path):
        """The daily store already holds the features, so no warm-up is needed."""
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(self._make_atm_ds())
        _compile_atm_accum_avg(compiler, catalog, _DR)

        kwargs = catalog.open_dataset.call_args.kwargs
        assert kwargs["start_date"] == _DR.start


class TestCompileAtmAccumAvgHourly:
    """
    With an hourly store the ekman chain is computed at compile time, so the
    read has to reach back far enough to start the rolling features warm.
    """

    @staticmethod
    def _capture_window(monkeypatch) -> dict:
        seen: dict = {}

        def _fake_open(catalog, var_key, date_range, bbox, **kwargs):
            seen["range"] = date_range
            return None  # short-circuits before any climatology is touched

        monkeypatch.setattr(compiler_registry, "_open_or_warn", _fake_open)
        return seen

    def test_widens_the_read_by_the_ekman_warmup(self, tmp_path, monkeypatch):
        seen = self._capture_window(monkeypatch)
        compiler = _make_compiler(tmp_path)

        result = compiler_registry._atm_accum_from_hourly(
            compiler, _make_catalog(None), _DR
        )

        assert result is None
        assert (_DR.start - seen["range"].start).days == _EKMAN_WARMUP_DAYS, (
            "hourly compile must reach back far enough to warm ekman_anom_lag14"
        )

    def test_widening_extends_backwards_only(self, tmp_path, monkeypatch):
        """The end must not move, or compile would emit days it was not asked for."""
        seen = self._capture_window(monkeypatch)
        compiler_registry._atm_accum_from_hourly(
            _make_compiler(tmp_path), _make_catalog(None), _DR
        )

        assert seen["range"].end == _DR.end

    def test_hourly_config_dispatches_to_the_hourly_path(self, tmp_path, monkeypatch):
        called = {}

        def _fake_hourly(compiler, catalog, date_range):
            called["yes"] = True
            return None

        monkeypatch.setattr(compiler_registry, "_atm_accum_from_hourly", _fake_hourly)
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables = {
            "atm-accum-avg": SimpleNamespace(time_step=TimeStep.HOURLY)
        }

        _compile_atm_accum_avg(compiler, _make_catalog(None), _DR)

        assert called.get("yes"), "time_step: hourly must not read the daily path"

    def test_daily_config_does_not_dispatch_to_the_hourly_path(
        self, tmp_path, monkeypatch
    ):
        def _boom(compiler, catalog, date_range):
            raise AssertionError("daily store must not take the hourly path")

        monkeypatch.setattr(compiler_registry, "_atm_accum_from_hourly", _boom)
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables = {
            "atm-accum-avg": SimpleNamespace(time_step=TimeStep.DAILY)
        }

        _compile_atm_accum_avg(compiler, _make_catalog(None), _DR)


# ---------------------------------------------------------------------------
# _compile_atm_instante
# ---------------------------------------------------------------------------


def _fake_hourly_atm_instante(start: str, end: str) -> xr.Dataset:
    """Hourly dataset shaped like the atm-instante store, on the base grid."""
    times = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    lat, lon = [30.0, 30.25], [-10.0, -9.75]
    shape = (len(times), len(lat), len(lon))
    values = {
        "u10": np.full(shape, 3.0, dtype="float32"),
        "v10": np.full(shape, 4.0, dtype="float32"),
        "tcc": np.full(shape, 0.5, dtype="float32"),
        "msl": np.full(shape, 101325.0, dtype="float32"),
    }
    return xr.Dataset(
        {k: (["time", "lat", "lon"], v) for k, v in values.items()},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def _hourly_instante_compiler(tmp_path) -> MagicMock:
    compiler = _make_compiler(tmp_path)
    compiler.app_config.variables = {
        "atm-instante": SimpleNamespace(time_step=TimeStep.HOURLY)
    }
    return compiler


class TestCompileAtmInstanteHourly:
    """
    With an hourly store the daily aggregates are computed at compile time, so
    h2ds must come out with the same columns and cadence as from a daily store.
    """

    def test_hourly_config_dispatches_to_the_hourly_path(self, tmp_path, monkeypatch):
        called = {}

        def _spy(catalog, var_key, bbox, date_range, reduce_slab, warmup_days=0):
            called["var_key"] = var_key
            return None

        monkeypatch.setattr(compiler_registry, "_reduce_hourly_in_slabs", _spy)

        compiler_registry._compile_atm_instante(
            _hourly_instante_compiler(tmp_path), _make_catalog(None), _DR
        )

        assert called.get("var_key") == "atm-instante"

    def test_daily_config_does_not_dispatch_to_the_hourly_path(
        self, tmp_path, monkeypatch
    ):
        def _boom(*args, **kwargs):
            raise AssertionError("daily store must not take the hourly path")

        monkeypatch.setattr(compiler_registry, "_reduce_hourly_in_slabs", _boom)
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables = {
            "atm-instante": SimpleNamespace(time_step=TimeStep.DAILY)
        }

        compiler_registry._compile_atm_instante(compiler, _make_catalog(None), _DR)

    def test_hourly_store_yields_the_daily_h2ds_columns(self, tmp_path):
        ds = _fake_hourly_atm_instante("2020-01-01", "2020-01-03")
        out = compiler_registry._compile_atm_instante(
            _hourly_instante_compiler(tmp_path), _make_catalog(ds), _DR
        )

        assert out is not None
        assert {"wind_mean", "wind_std", "wind_max", "u10", "v10", "tcc", "msl"} <= set(
            out.data_vars
        )
        assert out.sizes["time"] == 3, "h2ds stays daily whatever the store's cadence"

    def test_reduction_matches_the_daily_convert_path(self, tmp_path):
        """u=3, v=4 → wind speed 5 m/s; msl converted Pa→hPa exactly once."""
        ds = _fake_hourly_atm_instante("2020-01-01", "2020-01-03")
        out = compiler_registry._compile_atm_instante(
            _hourly_instante_compiler(tmp_path), _make_catalog(ds), _DR
        )

        assert out is not None
        np.testing.assert_allclose(out["wind_mean"].values, 5.0, rtol=1e-6)
        np.testing.assert_allclose(out["wind_max"].values, 5.0, rtol=1e-6)
        np.testing.assert_allclose(out["msl"].values, 1013.25, rtol=1e-6)

    def test_returns_none_when_data_missing(self, tmp_path):
        result = compiler_registry._compile_atm_instante(
            _hourly_instante_compiler(tmp_path), _make_catalog(None), _DR
        )
        assert result is None

    def test_last_day_keeps_all_of_its_hours(self, tmp_path):
        """A midnight-stamped slab end would drop the final day's other 23 steps."""
        ds = _fake_hourly_atm_instante("2020-01-01", "2020-01-03")
        # Make the last day's wind differ so a truncated read is visible.
        ds["u10"].loc[{"time": slice("2020-01-03 01:00", None)}] = 30.0

        out = compiler_registry._compile_atm_instante(
            _hourly_instante_compiler(tmp_path), _make_catalog(ds), _DR
        )

        assert out is not None
        last = float(out["wind_mean"].isel(time=-1).mean())
        assert last > 6.0, "reading only 00:00 of the last day would give 5 m/s"


class TestAtmInstanteSlabbing:
    """The reduction runs slab by slab, with no warm-up to trim."""

    def test_long_range_is_reduced_in_multiple_slabs(self, tmp_path, monkeypatch):
        ds = _fake_hourly_atm_instante("2020-01-01", "2020-01-31")
        cells = ds.sizes["lat"] * ds.sizes["lon"]
        monkeypatch.setattr(
            compiler_registry,
            "_SLAB_SOURCE_BUDGET_BYTES",
            cells * 4 * 4 * 24 * 10,  # 4 vars × float32 × 24 steps × 10 days
        )
        seen: list[DateRange] = []

        real = compiler_registry._daily_atm_instante_for_slab
        monkeypatch.setattr(
            compiler_registry,
            "_daily_atm_instante_for_slab",
            lambda hourly, slab: (seen.append(slab), real(hourly, slab))[1],
        )

        dr = DateRange("2020-01-01", "2020-01-31")
        out = compiler_registry._compile_atm_instante(
            _hourly_instante_compiler(tmp_path), _make_catalog(ds), dr
        )

        assert len(seen) > 1, "a long range must not be reduced in one graph"
        assert seen[0].start == dr.start and seen[-1].end == dr.end
        assert out is not None
        assert out.sizes["time"] == 31, "slabs must concatenate to the full range"

    def test_read_is_not_widened_backwards(self, tmp_path, monkeypatch):
        """Nothing here is a rolling feature, so warming would only cost reads."""
        seen: dict = {}

        def _fake_open(catalog, var_key, date_range, bbox, **kwargs):
            seen["range"] = date_range
            return None

        monkeypatch.setattr(compiler_registry, "_open_or_warn", _fake_open)

        compiler_registry._compile_atm_instante(
            _hourly_instante_compiler(tmp_path), _make_catalog(None), _DR
        )

        assert seen["range"].start == _DR.start


# ---------------------------------------------------------------------------
# radiation through compile_default (it has no processor of its own)
# ---------------------------------------------------------------------------


def _fake_hourly_radiation(start: str, end: str) -> xr.Dataset:
    """Hourly W/m² dataset shaped like the radiation store, on the base grid."""
    times = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    lat, lon = [30.0, 30.25], [-10.0, -9.75]
    shape = (len(times), len(lat), len(lon))
    return xr.Dataset(
        {
            "ssrd": (["time", "lat", "lon"], np.full(shape, 200.0, dtype="float32")),
            "tisr": (["time", "lat", "lon"], np.full(shape, 400.0, dtype="float32")),
            "slhf": (["time", "lat", "lon"], np.full(shape, -100.0, dtype="float32")),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )


def _hourly_radiation_compiler(tmp_path) -> MagicMock:
    compiler = _make_compiler(tmp_path)
    compiler.app_config.variables = {
        "radiation": SimpleNamespace(time_step=TimeStep.HOURLY)
    }
    return compiler


def _radiation_catalog(ds: xr.Dataset | None = None) -> MagicMock:
    catalog = _make_catalog(ds)
    catalog.var_key = "radiation"
    return catalog


class TestCompileRadiationHourly:
    """
    An hourly store needs only the daily mean — the units were settled at
    convert. That is exactly what ``compile_default`` does, so radiation is
    deliberately unregistered and these exercise it through the default.
    """

    def test_not_registered_so_it_rides_the_default(self):
        assert "radiation" not in COMPILE_PROCESSORS

    def test_hourly_config_dispatches_to_the_hourly_path(self, tmp_path, monkeypatch):
        called = {}

        def _spy(catalog, var_key, bbox, date_range, reduce_slab, warmup_days=0):
            called["var_key"] = var_key
            called["warmup"] = warmup_days
            return None

        monkeypatch.setattr(compiler_registry, "_reduce_hourly_in_slabs", _spy)

        compile_default(
            _hourly_radiation_compiler(tmp_path), _radiation_catalog(None), _DR
        )

        assert called.get("var_key") == "radiation"
        assert called.get("warmup") == 0, "a daily mean needs nothing from earlier days"

    def test_daily_config_does_not_dispatch_to_the_hourly_path(
        self, tmp_path, monkeypatch
    ):
        def _boom(*args, **kwargs):
            raise AssertionError("daily store must not take the hourly path")

        monkeypatch.setattr(compiler_registry, "_reduce_hourly_in_slabs", _boom)
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables = {
            "radiation": SimpleNamespace(time_step=TimeStep.DAILY)
        }

        compile_default(compiler, _radiation_catalog(None), _DR)

    def test_hourly_store_yields_daily_columns_unchanged_in_units(self, tmp_path):
        ds = _fake_hourly_radiation("2020-01-01", "2020-01-03")
        out = compile_default(
            _hourly_radiation_compiler(tmp_path), _radiation_catalog(ds), _DR
        )

        assert out is not None
        assert set(out.data_vars) == {"ssrd", "tisr", "slhf"}
        assert out.sizes["time"] == 3, "h2ds stays daily whatever the store's cadence"
        # A mean of constants is that constant — W/m² in, W/m² out, no rescaling.
        np.testing.assert_allclose(out["ssrd"].values, 200.0, rtol=1e-5)
        np.testing.assert_allclose(out["slhf"].values, -100.0, rtol=1e-5)

    def test_returns_none_when_data_missing(self, tmp_path):
        result = compile_default(
            _hourly_radiation_compiler(tmp_path), _radiation_catalog(None), _DR
        )
        assert result is None

    def test_last_day_keeps_all_of_its_hours(self, tmp_path):
        """A midnight-stamped slab end would average only 00:00 of the last day."""
        ds = _fake_hourly_radiation("2020-01-01", "2020-01-03")
        ds["ssrd"].loc[{"time": slice("2020-01-03 01:00", None)}] = 800.0

        out = compile_default(
            _hourly_radiation_compiler(tmp_path), _radiation_catalog(ds), _DR
        )

        assert out is not None
        last = float(out["ssrd"].isel(time=-1).mean())
        assert last > 700.0, "reading only 00:00 of the last day would give 200"


# ---------------------------------------------------------------------------
# _compile_sst
# ---------------------------------------------------------------------------


class TestCompileSst:
    def test_returns_none_when_data_missing(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(None)
        result = _compile_sst(compiler, catalog, _DR)
        assert result is None

    def test_calls_postprocess_sst_fdist(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(_daily_ds("sst", _DATES))

        with patch(
            "h2mare.processing.compiler_registry._compile_sst.__module__",
            create=True,
        ):
            with patch(
                "h2mare.processing.compiler.postprocess_sst_fdist",
                wraps=lambda ds, **kw: ds,
            ) as mock_post:
                _compile_sst(compiler, catalog, _DR)

        mock_post.assert_called_once()

    def test_clips_negative_sst_fdist_values(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        ds = _daily_ds("sst", _DATES)
        ds["sst_fdist"] = xr.DataArray(
            np.array([-1.0, 0.5, -0.2, 1.0]).reshape(1, 2, 2),
            dims=["time", "lat", "lon"],
            coords={"time": _DATES[:1], "lat": [30.0, 30.25], "lon": [-10.0, -9.75]},
        )
        # Use only 1 date to keep shape consistent
        ds = ds.isel(time=slice(0, 1))
        catalog = _make_catalog(ds)
        result = _compile_sst(compiler, catalog, _DR)
        assert result is not None
        assert float(result["sst_fdist"].min()) >= 0.0


# ---------------------------------------------------------------------------
# compile_default
# ---------------------------------------------------------------------------


class TestCompileDefault:
    def test_returns_none_when_data_missing(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(None)
        result = compile_default(compiler, catalog, _DR)
        assert result is None

    def test_passes_bbox_and_dates_to_catalog(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        ds = _daily_ds("ssh", _DATES)
        catalog = _make_catalog(ds)
        compile_default(compiler, catalog, _DR)
        catalog.open_dataset.assert_called_once_with(
            start_date=_DR.start,
            end_date=_DR.end,
            bbox=compiler.var_config.bbox,
        )

    def test_returns_dataset_on_success(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(_daily_ds("ssh", _DATES))
        result = compile_default(compiler, catalog, _DR)
        assert isinstance(result, xr.Dataset)

    def test_finer_store_is_aggregated_not_sampled(self, tmp_path):
        """
        Regression: a store finer than the base grid was put on it with
        ``interp_like``, which samples the target cell centre and ignores the
        rest of the cell — 1 of 25 source cells going 0.05° → 0.25°.
        """
        compiler = _make_compiler(tmp_path)
        # Two base-grid cells' worth of a 4x-finer store. The signal sits in one
        # corner of the first cell, away from its centre: a linear ramp would
        # not tell the two methods apart, since bilinear interpolation of a
        # ramp already equals its mean.
        step = 0.0625
        lat = 30.0 - step * 2 + (np.arange(8) + 0.5) * step
        lon = -10.0 - step * 2 + (np.arange(8) + 0.5) * step
        values = np.zeros((1, 8, 8), dtype="float32")
        values[0, 0, 0] = 16.0
        ds = xr.Dataset(
            {"ssh": xr.DataArray(values, dims=["time", "lat", "lon"])},
            coords={"time": _DATES[:1], "lat": lat, "lon": lon},
        )

        result = compile_default(compiler, _make_catalog(ds), _DR)

        assert result is not None
        # Mean of the 4x4 block behind the first cell, not the centre sample.
        expected = values[0, :4, :4].mean()
        assert float(result["ssh"][0, 0, 0]) == pytest.approx(expected, rel=1e-3)


class TestCompileDefaultHourly:
    """
    An unregistered hourly variable must still reach h2ds on the daily axis.
    Handed over unreduced it would not fail — the compiler's outer join would
    union 24 stamps into every day, leaving the daily variables null in 23.
    """

    @staticmethod
    def _setup(tmp_path, ds, *, time_step):
        compiler = _make_compiler(tmp_path)
        compiler.app_config.variables["newvar"] = SimpleNamespace(time_step=time_step)
        catalog = _make_catalog(ds)
        catalog.var_key = "newvar"
        return compiler, catalog

    def test_hourly_store_is_reduced_to_a_daily_axis(self, tmp_path):
        compiler, catalog = self._setup(
            tmp_path,
            _fake_hourly("2020-01-01", "2020-01-03"),
            time_step=TimeStep.HOURLY,
        )

        result = compile_default(compiler, catalog, _DR)

        assert result is not None
        assert list(pd.DatetimeIndex(result.time.values)) == list(_DATES)

    def test_reduction_is_a_mean_over_the_whole_day(self, tmp_path):
        """Every hour of the day must contribute — a midnight-stamped slab end
        would average the single 00:00 step instead of all 24."""
        ds = _fake_hourly("2020-01-01", "2020-01-01")
        # Hour-of-day as the value: the day's mean is 11.5, its first step 0.
        ds["v0"][:] = np.arange(24, dtype="float32")[:, None, None]
        compiler, catalog = self._setup(tmp_path, ds, time_step=TimeStep.HOURLY)

        result = compile_default(
            compiler, catalog, DateRange("2020-01-01", "2020-01-01")
        )

        assert result is not None
        assert float(result["v0"].isel(time=0).mean()) == pytest.approx(11.5)

    def test_daily_store_keeps_the_unreduced_path(self, tmp_path):
        compiler, catalog = self._setup(
            tmp_path, _daily_ds("newvar", _DATES), time_step=TimeStep.DAILY
        )

        result = compile_default(compiler, catalog, _DR)

        assert result is not None
        assert list(pd.DatetimeIndex(result.time.values)) == list(_DATES)
        # One plain read, not the slabbed hourly path.
        catalog.open_dataset.assert_called_once_with(
            start_date=_DR.start, end_date=_DR.end, bbox=compiler.var_config.bbox
        )

    def test_var_key_absent_from_config_is_treated_as_daily(self, tmp_path):
        """A stand-in config without the entry must keep the path every existing
        store uses, rather than reducing an axis that is already daily."""
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(_daily_ds("newvar", _DATES))
        catalog.var_key = "not-in-config"

        result = compile_default(compiler, catalog, _DR)

        assert result is not None
        assert list(pd.DatetimeIndex(result.time.values)) == list(_DATES)


# ---------------------------------------------------------------------------
# _process_variable dispatch (via Compiler)
# ---------------------------------------------------------------------------


class TestProcessVariableDispatch:
    """Verify _process_variable routes to the registry and falls back to default."""

    def _make_compiler_instance(self, tmp_path):
        import msgspec

        from h2mare.models import AppConfig
        from h2mare.processing.compiler import Compiler

        config = msgspec.convert(
            {
                "variables": {
                    "h2ds": {
                        "local_folder": "h2ds",
                        "source_vars": ["sst"],
                        "dataset_id_rep": "h2ds",
                        "source": "compiled",
                        "archive_raw": False,
                        "pattern": r"(\d{4})",
                        "subset": False,
                        "bbox": (-10, 25, 15, 55),
                    },
                    "ssh": {
                        "local_folder": "ssh",
                        "source_vars": ["adt"],
                        "dataset_id_rep": "cmems-ssh",
                        "source": "cmems",
                        "archive_raw": False,
                        "pattern": r".*\.nc",
                        "subset": False,
                        "bbox": (-10, 25, 15, 55),
                    },
                },
                "secrets": {},
            },
            AppConfig,
        )
        with patch("h2mare.processing.compiler.ZarrCatalog"):
            return Compiler(
                var_key="h2ds",
                app_config=config,
                remote_store_root=tmp_path / "remote",
                local_store_root=tmp_path / "local",
            )

    def test_registered_processor_is_called(self, tmp_path):
        compiler = self._make_compiler_instance(tmp_path)
        compiler.base_grid = _make_grid()

        sentinel = xr.Dataset({"moon_phase": xr.DataArray([1.0])})
        with (
            patch(
                "h2mare.processing.compiler_registry.COMPILE_PROCESSORS",
                {"moon": lambda *a: sentinel},
            ),
            patch("h2mare.processing.compiler.ZarrCatalog"),
        ):
            result = compiler._process_variable("moon", _DR)

        assert result is sentinel

    def test_compile_default_used_for_unregistered_variable(self, tmp_path):
        compiler = self._make_compiler_instance(tmp_path)
        compiler.base_grid = _make_grid()

        ds = _daily_ds("ssh", _DATES)
        catalog_mock = _make_catalog(ds)

        with (
            patch("h2mare.processing.compiler_registry.COMPILE_PROCESSORS", {}),
            patch(
                "h2mare.processing.compiler.ZarrCatalog",
                return_value=catalog_mock,
            ),
            patch.object(compiler, "_has_overlap", return_value=True),
        ):
            result = compiler._process_variable("ssh", _DR)

        assert isinstance(result, xr.Dataset)

    def test_returns_none_when_no_overlap(self, tmp_path):
        compiler = self._make_compiler_instance(tmp_path)
        compiler.base_grid = _make_grid()

        with (
            patch("h2mare.processing.compiler.ZarrCatalog"),
            patch.object(compiler, "_has_overlap", return_value=False),
        ):
            result = compiler._process_variable("ssh", _DR)

        assert result is None


# ---------------------------------------------------------------------------
# Hourly reduction slabbing
# ---------------------------------------------------------------------------


def _fake_hourly(start: str, end: str, n_vars: int = 1) -> xr.Dataset:
    """Hourly dataset over a whole-day span, shaped like the atm-accum store."""
    times = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    lat, lon = [30.0, 31.0, 32.0, 33.0], [-10.0, -9.0, -8.0, -7.0, -6.0]
    shape = (len(times), len(lat), len(lon))
    return xr.Dataset(
        {
            f"v{i}": (["time", "lat", "lon"], np.ones(shape, dtype="float32"))
            for i in range(n_vars)
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )


class TestSlabs:
    def test_slabs_tile_the_range_without_gap_or_overlap(self):
        dr = DateRange("2020-01-01", "2020-01-31")
        slabs = compiler_registry._slabs(dr, 10)

        assert slabs[0].start == dr.start
        assert slabs[-1].end == dr.end
        for prev, nxt in zip(slabs, slabs[1:]):
            assert nxt.start == prev.end + pd.Timedelta(days=1)

    def test_no_slab_exceeds_the_requested_length(self):
        slabs = compiler_registry._slabs(DateRange("2020-01-01", "2020-01-31"), 10)
        assert all((s.end - s.start).days + 1 <= 10 for s in slabs)

    def test_range_shorter_than_a_slab_stays_whole(self):
        dr = DateRange("2020-01-01", "2020-01-05")
        assert compiler_registry._slabs(dr, 30) == [dr]

    def test_absurd_slab_length_yields_one_slab_rather_than_overflowing(self):
        """
        A small store divides the byte budget into a day count far past what
        pd.Timedelta can hold. Clamping to the range is what keeps that an
        ordinary one-slab reduction instead of an OutOfBoundsTimedelta.
        """
        dr = DateRange("2020-01-01", "2020-01-05")
        assert compiler_registry._slabs(dr, 10**12) == [dr]


class TestSlabsAgainstStoreCoverage:
    """
    A compile window routinely outruns an hourly store: the CDS variables lag
    their provider by different amounts, so a range ending today reaches past
    whichever one is furthest behind. Slabs past the last stamp select nothing,
    and a reducer with no warm-up resamples an empty axis and raises — which is
    how one lagging variable took a whole 29-chunk compile down with it.
    """

    @staticmethod
    def _short_slabs(monkeypatch, ds, days: int = 5):
        cells = ds.sizes["lat"] * ds.sizes["lon"]
        per_day = cells * ds["v0"].dtype.itemsize * 24
        monkeypatch.setattr(
            compiler_registry, "_SLAB_SOURCE_BUDGET_BYTES", per_day * days
        )

    def test_a_range_past_the_end_of_the_store_still_reduces(self, monkeypatch):
        """The reducer is the real one, so an empty slab would raise as it did."""
        ds = _fake_hourly("2020-01-01", "2020-01-10")
        self._short_slabs(monkeypatch, ds)

        out = compiler_registry._reduce_hourly_in_slabs(
            _make_catalog(ds),
            "lagging-var",
            None,
            DateRange("2020-01-01", "2020-01-31"),
            compiler_registry._daily_mean_for_slab,
        )

        assert out is not None
        assert len(out.time) == 10, "only the days the store holds"

    def test_a_range_starting_before_the_store_still_reduces(self, monkeypatch):
        ds = _fake_hourly("2020-01-20", "2020-01-31")
        self._short_slabs(monkeypatch, ds)

        out = compiler_registry._reduce_hourly_in_slabs(
            _make_catalog(ds),
            "late-starting-var",
            None,
            DateRange("2020-01-01", "2020-01-31"),
            compiler_registry._daily_mean_for_slab,
        )

        assert out is not None
        assert len(out.time) == 12

    def test_no_overlap_at_all_returns_none(self):
        ds = _fake_hourly("2020-01-01", "2020-01-10")

        out = compiler_registry._reduce_hourly_in_slabs(
            _make_catalog(ds),
            "stale-var",
            None,
            DateRange("2021-01-01", "2021-01-31"),
            compiler_registry._daily_mean_for_slab,
        )

        assert out is None

    def test_the_reducer_never_sees_an_empty_slab(self, monkeypatch):
        ds = _fake_hourly("2020-01-01", "2020-01-10")
        self._short_slabs(monkeypatch, ds)
        seen: list[int] = []

        def _record(hourly, slab):
            sub = hourly.sel(
                time=slice(slab.start, compiler_registry._end_of_day(slab.end))
            )
            seen.append(sub.sizes.get("time", 0))
            return compiler_registry._daily_mean_for_slab(hourly, slab)

        compiler_registry._reduce_hourly_in_slabs(
            _make_catalog(ds),
            "lagging-var",
            None,
            DateRange("2020-01-01", "2020-01-31"),
            _record,
        )

        assert seen and all(n > 0 for n in seen)


class TestSlabDays:
    def test_derived_from_data_volume_not_machine(self, monkeypatch):
        ds = _fake_hourly("2020-01-01", "2020-01-31")
        cells = ds.sizes["lat"] * ds.sizes["lon"]
        per_day = cells * 4 * 24  # one float32 var, hourly

        monkeypatch.setattr(
            compiler_registry, "_SLAB_SOURCE_BUDGET_BYTES", per_day * 10
        )
        assert compiler_registry._slab_days(ds) == 10

    def test_more_variables_means_shorter_slabs(self, monkeypatch):
        monkeypatch.setattr(compiler_registry, "_SLAB_SOURCE_BUDGET_BYTES", 10**7)
        one = compiler_registry._slab_days(_fake_hourly("2020-01-01", "2020-01-31", 1))
        four = compiler_registry._slab_days(_fake_hourly("2020-01-01", "2020-01-31", 4))

        assert four < one, "slab length must shrink as h2ds grows"

    def test_never_returns_zero_days(self, monkeypatch):
        """A budget smaller than a single day must still make progress."""
        monkeypatch.setattr(compiler_registry, "_SLAB_SOURCE_BUDGET_BYTES", 1)
        assert (
            compiler_registry._slab_days(_fake_hourly("2020-01-01", "2020-01-05")) == 1
        )


class TestEndOfDay:
    def test_covers_the_final_hourly_step(self):
        """A midnight end bound would drop that day's other 23 steps."""
        got = compiler_registry._end_of_day("2020-01-31")

        assert got > pd.Timestamp("2020-01-31 23:00")
        assert got < pd.Timestamp("2020-02-01")


class TestAtmAccumSlabbing:
    """The reduction must run slab by slab, each warmed independently."""

    @staticmethod
    def _capture_slabs(monkeypatch, ds: xr.Dataset) -> list[DateRange]:
        seen: list[DateRange] = []

        def _fake_open(catalog, var_key, date_range, bbox, **kwargs):
            return ds

        def _fake_slab(ds_hourly, slab):
            seen.append(slab)
            times = pd.date_range(slab.start, slab.end, freq="D")
            return xr.Dataset(
                {"ekman_anom": (["time"], np.ones(len(times), dtype="float32"))},
                coords={"time": times},
            )

        monkeypatch.setattr(compiler_registry, "_open_or_warn", _fake_open)
        monkeypatch.setattr(compiler_registry, "_daily_features_for_slab", _fake_slab)
        return seen

    def test_range_is_processed_in_multiple_slabs(self, tmp_path, monkeypatch):
        ds = _fake_hourly("2019-12-12", "2020-01-31")
        seen = self._capture_slabs(monkeypatch, ds)
        cells = ds.sizes["lat"] * ds.sizes["lon"]
        monkeypatch.setattr(
            compiler_registry, "_SLAB_SOURCE_BUDGET_BYTES", cells * 4 * 24 * 10
        )

        dr = DateRange("2020-01-01", "2020-01-31")
        out = compiler_registry._atm_accum_from_hourly(
            _make_compiler(tmp_path), _make_catalog(None), dr
        )

        assert len(seen) > 1, "a long range must not be reduced in one graph"
        assert seen[0].start == dr.start and seen[-1].end == dr.end
        assert out is not None
        assert out.sizes["time"] == 31, "slabs must concatenate to the full range"

    def test_result_covers_every_requested_day_exactly_once(
        self, tmp_path, monkeypatch
    ):
        ds = _fake_hourly("2019-12-12", "2020-01-31")
        self._capture_slabs(monkeypatch, ds)
        cells = ds.sizes["lat"] * ds.sizes["lon"]
        monkeypatch.setattr(
            compiler_registry, "_SLAB_SOURCE_BUDGET_BYTES", cells * 4 * 24 * 7
        )

        out = compiler_registry._atm_accum_from_hourly(
            _make_compiler(tmp_path),
            _make_catalog(None),
            DateRange("2020-01-01", "2020-01-31"),
        )

        assert out is not None
        times = pd.DatetimeIndex(out.time.values)
        assert times.is_monotonic_increasing
        assert not times.duplicated().any()

    def test_missing_source_still_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            compiler_registry,
            "_open_or_warn",
            lambda catalog, var_key, date_range, bbox, **kw: None,
        )
        out = compiler_registry._atm_accum_from_hourly(
            _make_compiler(tmp_path), _make_catalog(None), _DR
        )
        assert out is None
