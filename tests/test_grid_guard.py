"""Refusing to write one grid into a store that holds another."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import check_grid_compatible


def _grid_ds(
    start: float = 30.0,
    step: float = 0.25,
    n: int = 8,
    dates: str = "2020-01-01",
    days: int = 3,
) -> xr.Dataset:
    """A (time, lat, lon) dataset on a cell-centred grid of *n* cells."""
    centres = start + (np.arange(n) + 0.5) * step
    times = pd.date_range(dates, periods=days, freq="D")
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], np.ones((days, n, n), dtype="float32"))},
        coords={"time": times, "lat": centres, "lon": centres.copy()},
    )


class TestCheckGridCompatible:
    def test_identical_grids_pass(self):
        check_grid_compatible(_grid_ds(), _grid_ds())

    def test_wider_extent_on_the_same_lattice_passes(self):
        """Widening a bbox adds cells at the ends; that merges correctly."""
        check_grid_compatible(_grid_ds(n=8), _grid_ds(start=28.0, n=16))

    def test_rounded_labels_pass(self):
        """Stored labels are rounded to 4 dp on write; that is not a new grid."""
        stored = _grid_ds(step=1 / 12, n=60)
        stored = stored.assign_coords(
            lat=np.round(stored["lat"].values, 4),
            lon=np.round(stored["lon"].values, 4),
        )
        check_grid_compatible(stored, _grid_ds(step=1 / 12, n=60))

    def test_finer_step_is_refused(self):
        with pytest.raises(ValueError, match=r"'lat' step differs"):
            check_grid_compatible(_grid_ds(step=0.25), _grid_ds(step=1 / 12, n=24))

    def test_half_cell_offset_is_refused(self):
        """Same step, shifted phase — node-registered against cell-centred."""
        with pytest.raises(ValueError, match="out of phase"):
            check_grid_compatible(_grid_ds(), _grid_ds(start=30.125))

    def test_irregular_stored_axis_is_refused(self):
        stored = _grid_ds()
        stored = stored.assign_coords(
            lat=[30.0, 30.25, 30.5, 31.5, 31.75, 32.0, 33.0, 34.0]
        )
        with pytest.raises(ValueError, match="not a regular axis"):
            check_grid_compatible(stored, _grid_ds())

    def test_single_cell_axis_has_no_step_to_compare(self):
        check_grid_compatible(_grid_ds(n=1), _grid_ds(n=8))

    def test_message_names_the_remedy(self):
        with pytest.raises(ValueError, match="its own store"):
            check_grid_compatible(_grid_ds(step=0.25), _grid_ds(step=0.5))


class TestWritePathRefuses:
    def test_append_on_another_grid_raises(self, tmp_path):
        """
        Regression: the append merged the two with an outer join, so the store
        came back holding both grids on one axis with every variable NaN at the
        other grid's cells — silently, and worse on each run.
        """
        path = tmp_path / "h2ds.zarr"
        write_append_zarr("sst", _grid_ds(step=0.25, n=8), path)

        with pytest.raises(ValueError, match="different grid"):
            write_append_zarr(
                "sst", _grid_ds(step=1 / 12, n=24, dates="2020-01-04"), path
            )

    def test_the_store_is_left_on_its_own_grid(self, tmp_path):
        path = tmp_path / "h2ds.zarr"
        write_append_zarr("sst", _grid_ds(step=0.25, n=8), path)

        with pytest.raises(ValueError):
            write_append_zarr(
                "sst", _grid_ds(step=1 / 12, n=24, dates="2020-01-04"), path
            )

        with xr.open_zarr(path, consolidated=False) as ds:
            assert ds.sizes["lat"] == 8
            assert ds.sizes["lon"] == 8

    def test_same_grid_still_appends(self, tmp_path):
        path = tmp_path / "h2ds.zarr"
        write_append_zarr("sst", _grid_ds(), path)
        write_append_zarr("sst", _grid_ds(dates="2020-01-04"), path)

        with xr.open_zarr(path, consolidated=False) as ds:
            assert ds.sizes["time"] == 6


class TestCompilerChecksEarly:
    """A wrong ``dx`` must fail before a chunk is computed, not after."""

    @staticmethod
    def _compiler(tmp_path: Path, grid: xr.Dataset) -> MagicMock:
        from h2mare.processing.compiler import Compiler

        compiler = MagicMock()
        compiler.catalog.store_root = tmp_path
        compiler.base_grid = grid
        compiler._check_store_grid = lambda: Compiler._check_store_grid(compiler)
        return compiler

    def test_raises_when_the_store_is_on_another_grid(self, tmp_path):
        _grid_ds(step=0.25, n=8).to_zarr(tmp_path / "h2ds_2020.zarr")
        compiler = self._compiler(tmp_path, _grid_ds(step=1 / 12, n=24))

        with pytest.raises(ValueError, match="different grid"):
            compiler._check_store_grid()

    def test_passes_on_the_store_s_own_grid(self, tmp_path):
        _grid_ds(step=0.25, n=8).to_zarr(tmp_path / "h2ds_2020.zarr")
        compiler = self._compiler(tmp_path, _grid_ds(step=0.25, n=8))

        compiler._check_store_grid()

    def test_empty_store_has_nothing_to_check(self, tmp_path):
        compiler = self._compiler(tmp_path, _grid_ds(step=1 / 12, n=24))

        compiler._check_store_grid()
