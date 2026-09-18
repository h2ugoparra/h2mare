"""Resolution-aware regridding onto the compile base grid."""

import numpy as np
import pytest
import xarray as xr

from h2mare.types import BBox
from h2mare.utils.spatial import GridBuilder, axis_step, regrid_to


def _grid(step: float, start: float = 0.0, n: int = 4) -> xr.Dataset:
    """Cell-centred grid of *n* cells of width *step* from *start*."""
    centres = start + (np.arange(n) + 0.5) * step
    return xr.Dataset(coords={"lat": centres, "lon": centres.copy()})


def _field(grid: xr.Dataset, values: np.ndarray | None = None) -> xr.Dataset:
    """Dataset of one variable on *grid*, filled with a ramp unless given."""
    shape = (grid.sizes["lat"], grid.sizes["lon"])
    if values is None:
        values = np.arange(shape[0] * shape[1], dtype="float32").reshape(shape)
    return xr.Dataset(
        {"v": xr.DataArray(values.astype("float32"), dims=("lat", "lon"))},
        coords={"lat": grid["lat"], "lon": grid["lon"]},
    )


class TestAxisStep:
    def test_recovers_step_from_rounded_labels(self):
        # What a 1/12 store looks like on disk: snap_grid_coords rounds every
        # label to 4 dp, so consecutive diffs alternate 0.0833/0.0834.
        exact = 1 / 24 + np.arange(840) / 12
        # The median of np.diff reads 0.083302 on this axis; the endpoints carry
        # the same rounding but divide it by 839 cells.
        assert axis_step(np.round(exact, 4)) == pytest.approx(1 / 12, abs=1e-6)

    def test_recovers_fine_step(self):
        exact = 1 / 480 + np.arange(2000) / 240
        assert axis_step(np.round(exact, 4)) == pytest.approx(1 / 240, abs=1e-6)

    def test_rejects_irregular_axis(self):
        with pytest.raises(ValueError, match="not regularly spaced"):
            axis_step(np.array([0.0, 1.0, 5.0, 6.0]))

    def test_rejects_descending_axis(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            axis_step(np.array([3.0, 2.0, 1.0]))


class TestMethodChoice:
    def test_coarsening_uses_conservative(self):
        source = _field(_grid(0.05, n=20))
        target = _grid(0.25, n=4)
        # Every source cell in the footprint contributes, so the corner cell is
        # the mean of its 5x5 block rather than the centre sample.
        result = regrid_to(source, target)
        block = source["v"].values[:5, :5]
        assert float(result["v"][0, 0]) == pytest.approx(block.mean(), rel=1e-4)

    def test_refining_matches_interp_like(self):
        """A finer target must go on interpolating exactly as it always has."""
        source = _field(_grid(0.25, n=4))
        target = _grid(0.05, n=20)
        expected = source.interp_like(target, method="linear", assume_sorted=True)
        result = regrid_to(source, target)
        np.testing.assert_allclose(result["v"].values, expected["v"].values)

    def test_rounded_same_resolution_stays_linear(self):
        """4 dp label rounding must not read as 'slightly coarser'."""
        source = _field(_grid(0.25, n=8))
        source = source.assign_coords(
            lat=np.round(source["lat"].values + 1e-5, 4),
            lon=np.round(source["lon"].values + 1e-5, 4),
        )
        target = _grid(0.25, n=8)
        expected = source.interp_like(target, method="linear", assume_sorted=True)
        np.testing.assert_allclose(
            regrid_to(source, target)["v"].values, expected["v"].values
        )


class TestConservativeCorrectness:
    def test_matches_block_mean_when_aligned(self):
        source = _field(_grid(0.05, n=20))
        target = _grid(0.25, n=4)
        expected = source["v"].coarsen(lat=5, lon=5).mean()
        result = regrid_to(source, target)
        # Latitude weighting is by cell area, so the two agree to the curvature
        # across 0.25 deg rather than exactly.
        np.testing.assert_allclose(result["v"].values, expected.values, rtol=1e-3)

    def test_constant_field_is_preserved(self):
        source = _field(_grid(1 / 24, n=48), np.full((48, 48), 7.5))
        result = regrid_to(source, _grid(0.25, n=8))
        np.testing.assert_allclose(result["v"].values, 7.5, rtol=1e-6)

    def test_non_integer_ratio(self):
        """0.05 -> 1/12 is ratio 1.67: no whole number of cells per target."""
        source = _field(_grid(0.05, n=20), np.full((20, 20), 3.0))
        result = regrid_to(source, _grid(1 / 12, n=12))
        np.testing.assert_allclose(result["v"].values, 3.0, rtol=1e-6)


class TestNaNHandling:
    def test_keeps_cell_that_interp_drops(self):
        """The 0 x NaN = NaN case: a land neighbour of weight ~0 nulls the cell."""
        # A target centre coincides with source cell 7. Interpolating there
        # still spans the interval to its left, so cell 6 takes part at weight
        # exactly 0 — and 0 * NaN is NaN, which is how a land pixel that
        # contributes nothing still empties an all-water cell.
        values = np.ones((10, 10), dtype="float32")
        values[6, 6] = np.nan
        source = _field(_grid(0.05, n=10), values)
        target = _grid(0.25, n=2)

        interped = source.interp_like(target, method="linear", assume_sorted=True)
        result = regrid_to(source, target)

        assert bool(np.isnan(interped["v"].values).any())
        assert not bool(np.isnan(result["v"].values).any())

    def test_averages_only_valid_cells(self):
        values = np.ones((10, 10), dtype="float32")
        values[:5, :5] = np.nan  # the whole first target cell is land
        values[5:, :5] = 4.0
        source = _field(_grid(0.05, n=10), values)
        result = regrid_to(source, _grid(0.25, n=2))

        assert bool(np.isnan(result["v"].values[0, 0]))
        assert float(result["v"].values[1, 0]) == pytest.approx(4.0, rel=1e-4)

    def test_min_coverage_masks_thin_cells(self):
        values = np.full((10, 10), np.nan, dtype="float32")
        values[0, 0] = 2.0  # 1 of 25 cells valid -> 4% coverage
        values[5:, 5:] = 2.0  # a full target cell
        source = _field(_grid(0.05, n=10), values)
        target = _grid(0.25, n=2)

        assert float(regrid_to(source, target)["v"].values[0, 0]) == pytest.approx(2.0)
        masked = regrid_to(source, target, min_coverage=0.5)
        assert bool(np.isnan(masked["v"].values[0, 0]))
        assert float(masked["v"].values[1, 1]) == pytest.approx(2.0)


class TestNearest:
    def test_never_chosen_automatically(self):
        """Averaging an identifier is wrong, but only the caller knows that."""
        # Eddy track IDs: unrelated integers, so their mean is not one of them.
        rng = np.random.default_rng(0)
        ids = rng.integers(1, 100_000, size=(10, 10)).astype("float32")
        source = _field(_grid(0.05, n=10), ids)
        target = _grid(0.25, n=2)

        averaged = regrid_to(source, target)["v"].values
        picked = regrid_to(source, target, method="nearest")["v"].values

        assert np.abs(averaged - np.round(averaged)).max() > 1e-3
        assert set(np.unique(picked)).issubset(set(np.unique(ids)))

    def test_carries_target_labels(self):
        source = _field(_grid(0.05, n=10))
        target = _grid(0.25, n=2)
        result = regrid_to(source, target, method="nearest")
        np.testing.assert_array_equal(result["lat"].values, target["lat"].values)


class TestGridIdentity:
    @pytest.mark.parametrize("step", [0.05, 0.25, 0.5])
    def test_coords_are_bit_identical_to_target(self, step):
        """Compiler.run merges with join='outer': near-equal axes would union."""
        target = GridBuilder(BBox(-80, 0, 10, 70), 0.25, 0.25).generate_grid()
        source = _field(_grid(step, start=-80.0, n=int(10 / step)))
        result = regrid_to(source, target)

        for axis in ("lat", "lon"):
            assert np.array_equal(result[axis].values, target[axis].values)

    def test_merge_does_not_double_the_axis(self):
        target = GridBuilder(BBox(-80, 0, 10, 70), 0.25, 0.25).generate_grid()
        coarse = _field(_grid(0.05, start=-80.0, n=200)).rename({"v": "a"})
        fine = _field(_grid(0.5, start=-80.0, n=20)).rename({"v": "b"})

        merged = xr.merge(
            [regrid_to(coarse, target), regrid_to(fine, target)], join="outer"
        )
        assert merged.sizes["lat"] == target.sizes["lat"]
        assert merged.sizes["lon"] == target.sizes["lon"]


class TestDatasetPlumbing:
    def test_preserves_extra_dims_and_dtype(self):
        values = np.ones((3, 10, 10), dtype="float32")
        source = xr.Dataset(
            {"v": xr.DataArray(values, dims=("time", "lat", "lon"))},
            coords={
                "time": np.arange(3),
                "lat": (np.arange(10) + 0.5) * 0.05,
                "lon": (np.arange(10) + 0.5) * 0.05,
            },
        )
        result = regrid_to(source, _grid(0.25, n=2))

        assert result["v"].dims == ("time", "lat", "lon")
        assert result["v"].dtype == np.dtype("float32")
        assert result.sizes["time"] == 3

    def test_preserves_variable_attrs(self):
        source = _field(_grid(0.05, n=10))
        source["v"].attrs = {"units": "degrees_C"}
        result = regrid_to(source, _grid(0.25, n=2))
        assert result["v"].attrs["units"] == "degrees_C"

    def test_works_on_dask_backed_data(self):
        source = _field(_grid(0.05, n=20)).chunk({"lat": 5, "lon": 5})
        result = regrid_to(source, _grid(0.25, n=4))
        assert result["v"].chunks is not None
        np.testing.assert_allclose(
            result["v"].compute().values,
            source["v"].compute().coarsen(lat=5, lon=5).mean().values,
            rtol=1e-3,
        )

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="unknown regrid method"):
            regrid_to(
                _field(_grid(0.05, n=10)),
                _grid(0.25, n=2),
                method="bilinear",  # type: ignore[arg-type]
            )
