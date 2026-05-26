"""Tests for processing/compiler_registry.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.processing.compiler_registry import (
    COMPILE_PROCESSORS,
    _compile_atm_accum_avg,
    _compile_bathy,
    _compile_moon,
    _compile_o2,
    _compile_sst,
    _compile_thetao,
    compile_default,
)
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


def _make_compiler(tmp_path: Path, *, bbox=(- 10, 25, 15, 55)) -> MagicMock:
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
        {var: xr.DataArray(data, dims=["time", "lat", "lon"],
                           coords={"time": dates,
                                   "lat": [30.0, 30.25],
                                   "lon": [-10.0, -9.75]})},
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

    def test_contains_o2(self):
        assert "o2" in COMPILE_PROCESSORS

    def test_contains_thetao(self):
        assert "thetao" in COMPILE_PROCESSORS

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
        compiler.app_config.variables["bathy"] = bathy_cfg

        fake_ds = _daily_ds("elevation", _DATES).isel(time=0).drop_vars("time")
        fake_ds = fake_ds.rename({"lat": "lat", "lon": "lon"})

        with patch("h2mare.processing.compiler_registry.xr.open_dataset",
                   return_value=fake_ds) as mock_open:
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
# _compile_o2
# ---------------------------------------------------------------------------


class TestCompileO2:
    def _make_o2_ds(self) -> xr.Dataset:
        depths = [0, 100, 500, 1000]
        data = np.ones((3, len(depths), 2, 2), dtype="float32")
        return xr.Dataset(
            {"o2": xr.DataArray(
                data,
                dims=["time", "depth", "lat", "lon"],
                coords={"time": _DATES, "depth": depths,
                        "lat": [30.0, 30.25], "lon": [-10.0, -9.75]},
            )}
        )

    def test_returns_none_when_data_missing(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(None)
        result = _compile_o2(compiler, catalog, _DR)
        assert result is None

    def test_returns_one_variable_per_depth(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(self._make_o2_ds())
        result = _compile_o2(compiler, catalog, _DR)
        assert result is not None
        assert "o2_0" in result.data_vars
        assert "o2_100" in result.data_vars
        assert "o2_500" in result.data_vars
        assert "o2_1000" in result.data_vars

    def test_depth_dim_dropped_in_output(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(self._make_o2_ds())
        result = _compile_o2(compiler, catalog, _DR)
        assert result is not None
        for var in result.data_vars:
            assert "depth" not in result[var].dims


# ---------------------------------------------------------------------------
# _compile_thetao
# ---------------------------------------------------------------------------


class TestCompileThetao:
    def _make_thetao_ds(self) -> xr.Dataset:
        depths = [100, 200, 500, 1000]
        data = np.ones((3, len(depths), 2, 2), dtype="float32")
        return xr.Dataset(
            {"thetao": xr.DataArray(
                data,
                dims=["time", "depth", "lat", "lon"],
                coords={"time": _DATES, "depth": depths,
                        "lat": [30.0, 30.25], "lon": [-10.0, -9.75]},
            )}
        )

    def test_returns_none_when_data_missing(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(None)
        result = _compile_thetao(compiler, catalog, _DR)
        assert result is None

    def test_returns_one_variable_per_depth(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(self._make_thetao_ds())
        result = _compile_thetao(compiler, catalog, _DR)
        assert result is not None
        assert "thetao_100" in result.data_vars
        assert "thetao_200" in result.data_vars
        assert "thetao_500" in result.data_vars
        assert "thetao_1000" in result.data_vars

    def test_depth_dim_dropped_in_output(self, tmp_path):
        compiler = _make_compiler(tmp_path)
        catalog = _make_catalog(self._make_thetao_ds())
        result = _compile_thetao(compiler, catalog, _DR)
        assert result is not None
        for var in result.data_vars:
            assert "depth" not in result[var].dims


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
                        "variables": ["sst"],
                        "dataset_id_rep": "h2ds",
                        "source": "compiled",
                        "pattern": r"(\d{4})",
                        "subset": False,
                        "bbox": (-10, 25, 15, 55),
                    },
                    "ssh": {
                        "local_folder": "ssh",
                        "variables": ["adt"],
                        "dataset_id_rep": "cmems-ssh",
                        "source": "cmems",
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
            patch(
                "h2mare.processing.compiler_registry.COMPILE_PROCESSORS", {}
            ),
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
