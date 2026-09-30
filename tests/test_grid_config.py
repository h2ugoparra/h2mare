"""The compile grid declared in config: resolution, values_at, and phase."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import numpy as np
import pytest
import xarray as xr
from loguru import logger

from h2mare.models import AppConfig
from h2mare.types import BBox
from h2mare.utils.spatial import GridBuilder, regrid_to

VALID_ENTRY = {
    "local_folder": "x",
    "source_vars": ["a"],
    "dataset_id_rep": "d",
    "source": "cmems",
    "archive_raw": False,
}


def _config(**entry):
    return {"variables": {"dyn": {**VALID_ENTRY, **entry}}, "secrets": {}}


class TestCellsPerDegree:
    def test_a_whole_count_is_accepted(self):
        raw = _config(cells_per_degree=12, bbox=[-80, 0, 10, 70])
        assert msgspec.convert(raw, AppConfig).variables["dyn"].cells_per_degree == 12

    def test_zero_or_negative_is_refused(self):
        with pytest.raises(msgspec.ValidationError, match="positive whole number"):
            msgspec.convert(_config(cells_per_degree=0), AppConfig)

    def test_a_count_that_does_not_tile_the_bbox_is_refused(self):
        """4 cells per degree leaves 0.2 of a cell on a 10.3° span."""
        raw = _config(cells_per_degree=4, bbox=[-10, 30, 10, 40.3])
        with pytest.raises(msgspec.ValidationError, match="does not divide"):
            msgspec.convert(raw, AppConfig)

    def test_the_message_names_the_axis_and_the_span(self):
        raw = _config(cells_per_degree=4, bbox=[-10, 30, 10, 40.3])
        with pytest.raises(msgspec.ValidationError) as err:
            msgspec.convert(raw, AppConfig)
        assert "lat span (10.3" in str(err.value)

    def test_values_at_defaults_to_center(self):
        cfg = msgspec.convert(_config(), AppConfig)
        assert cfg.variables["dyn"].values_at == "cell_center"

    def test_an_unknown_values_at_is_refused(self):
        with pytest.raises(msgspec.ValidationError, match="Invalid enum value"):
            msgspec.convert(_config(values_at="corner"), AppConfig)


class TestGridBuilder:
    BOX = BBox(-80, 0, 10, 70)

    def test_center_is_unchanged_from_the_shipped_grid(self):
        """The 0.25° h2ds axis must not move: every store on disk is on it."""
        grid = GridBuilder(self.BOX, 0.25, 0.25).generate_grid()
        expected = np.arange(0 + 0.125, 70 + 0.125, 0.25)
        np.testing.assert_array_equal(grid["lat"].values, expected)
        assert grid.sizes == {"lat": 280, "lon": 360}

    def test_grid_line_puts_values_on_the_bbox_edges(self):
        grid = GridBuilder(self.BOX, 0.25, 0.25, values_at="grid_line").generate_grid()
        assert grid["lat"].values[0] == 0.0
        assert grid["lat"].values[-1] == 70.0
        assert grid.sizes == {"lat": 281, "lon": 361}

    def test_a_twelfth_degree_grid_has_the_right_cell_count(self):
        grid = GridBuilder(self.BOX, 1 / 12, 1 / 12).generate_grid()
        assert grid.sizes == {"lat": 840, "lon": 1080}

    def test_the_count_does_not_depend_on_the_bbox(self):
        """
        Regression: ``np.arange`` decides its length in floating point, so a
        step that is not exactly representable gives 91 cells over [30, 45] at
        1/6° where 90 belong — while the same step elsewhere is right.
        """
        grid = GridBuilder(BBox(-10, 30, 10, 45), 1 / 6, 1 / 6).generate_grid()
        assert grid.sizes["lat"] == 90
        assert np.arange(30 + (1 / 6) / 2, 45 + (1 / 6) / 2, 1 / 6).size == 91

    def test_the_two_are_half_a_cell_apart(self):
        centred = GridBuilder(self.BOX, 0.25, 0.25).generate_grid()
        node = GridBuilder(self.BOX, 0.25, 0.25, values_at="grid_line").generate_grid()
        assert centred["lat"].values[0] - node["lat"].values[0] == pytest.approx(0.125)


class TestCompilerReadsTheConfig:
    @staticmethod
    def _grid(cells_per_degree, values_at):
        from h2mare.processing.compiler import Compiler

        compiler = MagicMock()
        compiler.bbox = BBox(-80, 0, 10, 70)
        compiler.var_config = SimpleNamespace(
            cells_per_degree=cells_per_degree, values_at=values_at
        )
        return Compiler._build_base_grid(compiler)

    def test_the_declared_resolution_is_used(self):
        grid = self._grid(12, "cell_center")
        assert grid.sizes == {"lat": 840, "lon": 1080}

    def test_the_declared_values_at_is_used(self):
        assert self._grid(4, "grid_line")["lat"].values[0] == 0.0

    def test_an_entry_declaring_neither_gets_the_shipped_grid(self):
        grid = self._grid(None, "cell_center")
        assert grid.sizes == {"lat": 280, "lon": 360}
        assert grid["lat"].values[0] == pytest.approx(0.125)


class TestPhaseWarning:
    """A source at the target's resolution but off its phase loses detail."""

    @staticmethod
    def _source(start: float, step: float = 0.25, n: int = 8) -> xr.Dataset:
        axis = start + np.arange(n) * step
        return xr.Dataset(
            {"v": (("lat", "lon"), np.ones((n, n), dtype="float32"))},
            coords={"lat": axis, "lon": axis.copy()},
        )

    @staticmethod
    def _target(start: float, step: float = 0.25, n: int = 7) -> xr.Dataset:
        axis = start + np.arange(n) * step
        return xr.Dataset(coords={"lat": axis, "lon": axis.copy()})

    @staticmethod
    def _warnings(*calls) -> list[str]:
        """Messages logged while regridding each (source, target) pair."""
        from h2mare.utils import spatial

        spatial._phase_warned.clear()
        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            for source, target in calls:
                regrid_to(source, target)
        finally:
            logger.remove(sink)
        return [m for m in messages if "out of phase" in m]

    def test_an_offset_source_at_the_same_resolution_warns(self):
        msgs = self._warnings((self._source(0.0), self._target(0.125)))
        assert len(msgs) == 1
        assert "0.50 cells out of phase" in msgs[0]

    def test_an_aligned_source_is_quiet(self):
        assert self._warnings((self._source(0.0), self._target(0.0))) == []

    def test_a_coarsened_source_is_quiet_whatever_its_phase(self):
        """The area mean integrates over the target cell; phase cannot bite."""
        fine = self._source(0.0, step=0.05, n=40)
        assert self._warnings((fine, self._target(0.125))) == []

    def test_it_is_said_once_per_grid_pair(self):
        pair = (self._source(0.0), self._target(0.125))
        assert len(self._warnings(pair, pair)) == 1
