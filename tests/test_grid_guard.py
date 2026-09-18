"""Refusing to write one grid, or one set of depth levels, into a store that
holds another."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import (
    check_depth_compatible,
    check_grid_compatible,
)


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


def _depth_ds(
    levels: list[float],
    dates: str = "2020-01-01",
    days: int = 3,
    fill: float = 1.0,
) -> xr.Dataset:
    """A (time, depth, lat, lon) dataset on the standard 8-cell grid."""
    centres = 30.0 + (np.arange(8) + 0.5) * 0.25
    times = pd.date_range(dates, periods=days, freq="D")
    shape = (days, len(levels), 8, 8)
    return xr.Dataset(
        {"chl": (["time", "depth", "lat", "lon"], np.full(shape, fill, "float32"))},
        coords={
            "time": times,
            "depth": np.array(levels, dtype="float32"),
            "lat": centres,
            "lon": centres.copy(),
        },
    )


class TestCheckDepthCompatible:
    def test_identical_levels_pass(self):
        check_depth_compatible(_depth_ds([0.5, 5.0]), _depth_ds([0.5, 5.0]))

    def test_added_levels_are_refused(self):
        with pytest.raises(ValueError, match="different depth levels"):
            check_depth_compatible(_depth_ds([0.5]), _depth_ds([0.5, 5.0, 110.0]))

    def test_dropped_levels_are_refused(self):
        with pytest.raises(ValueError, match="different depth levels"):
            check_depth_compatible(_depth_ds([0.5, 5.0, 110.0]), _depth_ds([0.5]))

    def test_same_count_different_levels_is_refused(self):
        with pytest.raises(ValueError, match="different depth levels"):
            check_depth_compatible(_depth_ds([0.5, 5.0]), _depth_ds([0.5, 50.0]))

    def test_float32_drift_within_tolerance_passes(self):
        """o2's 2023+ files sit an ULP off the older ones; that is not a change."""
        stored = _depth_ds([0.494, 109.7])
        incoming = _depth_ds([0.494, 109.7])
        incoming = incoming.assign_coords(
            depth=incoming["depth"].values + np.float32(6e-5)
        )
        check_depth_compatible(stored, incoming)

    def test_a_dataset_without_depth_is_not_compared(self):
        """Adding a 2-D variable alongside 3-D ones changes no shared axis."""
        check_depth_compatible(_depth_ds([0.5, 5.0]), _grid_ds())
        check_depth_compatible(_grid_ds(), _depth_ds([0.5, 5.0]))

    def test_message_names_the_levels_and_the_remedy(self):
        with pytest.raises(ValueError, match=r"stored \[0.5\], incoming \[0.5, 5\]"):
            check_depth_compatible(_depth_ds([0.5]), _depth_ds([0.5, 5.0]))
        with pytest.raises(ValueError, match="--start-date"):
            check_depth_compatible(_depth_ds([0.5]), _depth_ds([0.5, 5.0]))

    def test_many_levels_are_abbreviated(self):
        with pytest.raises(ValueError, match=r"… \(23 levels\)"):
            check_depth_compatible(
                _depth_ds([0.5]), _depth_ds([float(i) for i in range(23)])
            )


class TestWritePathRefusesDepthChange:
    def test_trailing_append_with_new_levels_raises(self, tmp_path):
        """
        Regression: the append unioned the depth axes, so the store came back
        holding every level with each pre-existing date NaN at the new ones —
        silently, and worse on each run.
        """
        path = tmp_path / "bio_rep.zarr"
        write_append_zarr("bio_rep", _depth_ds([0.5]), path)

        with pytest.raises(ValueError, match="different depth levels"):
            write_append_zarr(
                "bio_rep", _depth_ds([0.5, 5.0, 110.0], dates="2020-01-04"), path
            )

    def test_the_store_is_left_on_its_own_levels(self, tmp_path):
        path = tmp_path / "bio_rep.zarr"
        write_append_zarr("bio_rep", _depth_ds([0.5]), path)

        with pytest.raises(ValueError):
            write_append_zarr(
                "bio_rep", _depth_ds([0.5, 5.0, 110.0], dates="2020-01-04"), path
            )

        with xr.open_zarr(path, consolidated=False) as ds:
            assert ds.sizes["depth"] == 1
            assert ds.sizes["time"] == 3

    def test_same_levels_still_append(self, tmp_path):
        path = tmp_path / "bio_rep.zarr"
        write_append_zarr("bio_rep", _depth_ds([0.5, 5.0]), path)
        write_append_zarr("bio_rep", _depth_ds([0.5, 5.0], dates="2020-01-04"), path)

        with xr.open_zarr(path, consolidated=False) as ds:
            assert ds.sizes["time"] == 6
            assert ds.sizes["depth"] == 2

    def test_a_full_rewrite_may_change_the_levels(self, tmp_path):
        """
        Re-running over the whole period is how a store moves onto new levels:
        nothing of the old axis survives, so there is no union to refuse.
        """
        path = tmp_path / "bio_rep.zarr"
        write_append_zarr("bio_rep", _depth_ds([0.5], fill=1.0), path)
        write_append_zarr("bio_rep", _depth_ds([0.5, 5.0, 110.0], fill=2.0), path)

        with xr.open_zarr(path, consolidated=False) as ds:
            assert ds.sizes["depth"] == 3
            assert ds.sizes["time"] == 3
            assert not np.isnan(ds["chl"].values).any()

    def test_a_full_rewrite_warns_that_levels_are_dropped(self, tmp_path):
        path = tmp_path / "bio_rep.zarr"
        write_append_zarr("bio_rep", _depth_ds([0.5, 5.0, 110.0]), path)

        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            write_append_zarr("bio_rep", _depth_ds([5.0, 110.0]), path)
        finally:
            logger.remove(sink)

        assert any("different depth levels" in m for m in messages), messages

        with xr.open_zarr(path, consolidated=False) as ds:
            assert ds.sizes["depth"] == 2
