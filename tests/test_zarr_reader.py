"""Unit tests for ZarrReader — the opening half of the ZarrCatalog split.

The suite drives ZarrReader only through the ``ZarrCatalog`` facade, leaving its
own contract untested: the two-mode argument guard, time normalisation, and the
preprocessing applied per-file before ``open_mfdataset`` combines them. Those
last two run inside a callback during the open, so they are exercised directly
here rather than inferred from a combined dataset.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.storage.zarr_reader import (
    AXIS_SNAP_TOL,
    ZarrReader,
    snap_axes_to_reference,
)
from h2mare.types import BBox


def _reader(coverage=None) -> ZarrReader:
    """Reader over a stubbed index — these paths never touch the catalog."""
    index = MagicMock()
    index.get_time_coverage.return_value = coverage
    index.var_key = "sst"
    return ZarrReader(index)


def _ds(times, lats=(30.0, 35.0, 40.0), lons=(-10.0, -5.0, 0.0), variables=("sst",)):
    """Small time × lat × lon dataset, one distinct value per cell."""
    index = pd.DatetimeIndex(times)
    shape = (len(index), len(lats), len(lons))
    size = int(np.prod(shape))
    return xr.Dataset(
        {
            v: (("time", "lat", "lon"), np.arange(size, dtype=float).reshape(shape))
            for v in variables
        },
        coords={"time": index, "lat": list(lats), "lon": list(lons)},
    )


class TestOpenDatasetGuards:
    def test_both_modes_at_once_raises(self):
        reader = _reader()
        with pytest.raises(ValueError, match="Cannot use both"):
            reader.open_dataset(dates="2021-01-01", start_date="2021-01-01")

    def test_no_dates_and_no_coverage_raises(self):
        """With nothing requested the reader falls back to full coverage; an
        empty store leaves it with no range to open."""
        reader = _reader(coverage=None)
        with pytest.raises(ValueError, match="sparse 'dates' or"):
            reader.open_dataset()


class TestNormalizeTime:
    def test_times_are_truncated_to_midnight(self):
        reader = _reader()
        ds = _ds(["2021-01-01T13:30:00", "2021-01-02T06:15:00"])

        out = reader._normalize_time(ds)

        assert list(pd.DatetimeIndex(out.time.values)) == [
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-02"),
        ]

    def test_dataset_without_time_passes_through(self):
        """Time-less statics (bathy) go through the same open path."""
        reader = _reader()
        ds = xr.Dataset(
            {"elevation": (("lat",), [1.0, 2.0])}, coords={"lat": [0.0, 1.0]}
        )

        assert reader._normalize_time(ds) is ds


class TestPreprocessDataset:
    def test_selects_requested_variables(self):
        reader = _reader()
        ds = _ds(["2021-01-01"], variables=("sst", "chl"))

        out = reader._preprocess_dataset(ds, bbox=None, variables=["sst"])

        assert set(out.data_vars) == {"sst"}

    def test_accepts_a_bare_string_variable(self):
        reader = _reader()
        ds = _ds(["2021-01-01"], variables=("sst", "chl"))

        out = reader._preprocess_dataset(ds, bbox=None, variables="sst")

        assert set(out.data_vars) == {"sst"}

    def test_unknown_variable_keeps_everything(self):
        """Selecting nothing would silently empty the dataset, so a fully
        unmatched request warns and leaves the file untouched instead."""
        reader = _reader()
        ds = _ds(["2021-01-01"], variables=("sst",))

        out = reader._preprocess_dataset(ds, bbox=None, variables=["nope"])

        assert set(out.data_vars) == {"sst"}

    def test_descending_latitude_is_sorted_ascending(self):
        """ERA5 ships north-to-south; leaving it that way makes every later
        slice(lat) selection silently empty."""
        reader = _reader()
        ds = _ds(["2021-01-01"], lats=(40.0, 35.0, 30.0))

        out = reader._preprocess_dataset(ds, bbox=None, variables=None)

        assert list(out.lat.values) == [30.0, 35.0, 40.0]

    def test_bbox_subsets_the_grid_keeping_one_cell_of_padding(self):
        """The subset is padded by one cell on each side, so interpolating onto
        the requested extent later has neighbours at the edges instead of NaN.
        A grid only as wide as the request would hide the padding entirely."""
        reader = _reader()
        ds = _ds(
            ["2021-01-01"],
            lats=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0),
            lons=(-20.0, -10.0, 0.0, 10.0, 20.0),
        )

        out = reader._preprocess_dataset(
            ds, bbox=BBox.from_tuple((-10.0, 10.0, 0.0, 20.0)), variables=None
        )

        # requested lats 10-20, plus one cell either side
        assert list(out.lat.values) == [0.0, 10.0, 20.0, 30.0]
        # requested lons -10-0, plus one cell either side
        assert list(out.lon.values) == [-20.0, -10.0, 0.0, 10.0]


class TestApplyBbox:
    def test_missing_coordinates_returns_dataset_unchanged(self):
        """A store without lat/lon (or y/x) cannot be subset; that is a warning,
        not a failure, so the caller still gets its data."""
        reader = _reader()
        ds = xr.Dataset({"v": (("depth",), [1.0, 2.0])}, coords={"depth": [0.0, 10.0]})

        out = reader._apply_bbox(ds, BBox.from_tuple((-10.0, 30.0, 0.0, 40.0)))

        assert out is ds

    def test_a_failed_selection_raises_rather_than_returning_everything(self):
        """
        Regression: a selection that raised was logged and the unclipped
        dataset returned, so asking for a region silently gave the whole domain.
        """
        reader = _reader()
        ds = xr.Dataset(
            {"v": (("lat", "lon"), np.ones((3, 4)))},
            coords={"lat": [30.0, 35.0, 40.0], "lon": [0.0, -10.0, 10.0, -5.0]},
        )

        with pytest.raises(ValueError, match="could not subset to"):
            reader._apply_bbox(ds, BBox.from_tuple((-6.0, 30.0, 1.0, 40.0)))


class TestSnapAxesToReference:
    """
    The same grid written on different occasions can disagree in the last
    float bits. combine="by_coords" compares coordinate arrays exactly, so it
    stops treating the axis as shared and starts treating it as one to
    concatenate along — surfacing as "does not have monotonic global indexes
    along dimension lon", which names neither the cause nor the file.
    """

    @staticmethod
    def _drifted(values, scale: float = 1e-12) -> np.ndarray:
        """The same axis as it might be rebuilt: equal to within float noise."""
        arr = np.asarray(values, dtype=float)
        return arr + np.linspace(0.0, scale, arr.size)

    def test_float_noise_is_snapped_to_the_reference(self):
        ref = np.array([-10.0, -5.0, 0.0])
        ds = _ds(["2021-01-01"], lons=self._drifted(ref))

        out, snapped = snap_axes_to_reference(ds, {"lon": ref})

        assert set(snapped) == {"lon"}
        assert snapped["lon"] == pytest.approx(1e-12, rel=0.1)
        np.testing.assert_array_equal(out.lon.values, ref)

    def test_an_identical_axis_is_left_alone(self):
        ref = np.array([-10.0, -5.0, 0.0])
        ds = _ds(["2021-01-01"], lons=ref)

        out, snapped = snap_axes_to_reference(ds, {"lon": ref})

        assert snapped == {}
        np.testing.assert_array_equal(out.lon.values, ref)

    def test_a_genuinely_different_grid_is_refused(self):
        """0.25 vs 0.1 spacing is not drift; it must still reach xarray."""
        ref = np.array([-10.0, -5.0, 0.0])
        shifted = ref + 0.25
        ds = _ds(["2021-01-01"], lons=shifted)

        out, snapped = snap_axes_to_reference(ds, {"lon": ref})

        assert snapped == {}
        np.testing.assert_array_equal(out.lon.values, shifted)

    def test_a_different_length_is_refused(self):
        ds = _ds(["2021-01-01"], lons=(-10.0, -5.0, 0.0))

        out, snapped = snap_axes_to_reference(ds, {"lon": np.array([-10.0, 0.0])})

        assert snapped == {}
        assert out.lon.size == 3

    def test_both_axes_snap_independently(self):
        ref_lat = np.array([30.0, 35.0, 40.0])
        ref_lon = np.array([-10.0, -5.0, 0.0])
        ds = _ds(["2021-01-01"], lats=self._drifted(ref_lat), lons=ref_lon)

        _, snapped = snap_axes_to_reference(ds, {"lat": ref_lat, "lon": ref_lon})

        assert set(snapped) == {"lat"}  # lon already matched

    def test_a_coord_absent_from_the_dataset_is_skipped(self):
        ds = _ds(["2021-01-01"]).drop_vars("lat")

        _, snapped = snap_axes_to_reference(ds, {"lat": np.array([1.0, 2.0, 3.0])})

        assert snapped == {}

    def test_tolerance_is_the_boundary(self):
        ref = np.array([0.0, 1.0, 2.0])
        just_over = ref + np.array([0.0, 0.0, 2e-9])

        _, snapped = snap_axes_to_reference(
            _ds(["2021-01-01"], lons=just_over), {"lon": ref}, tol=1e-9
        )
        assert snapped == {}

        _, snapped = snap_axes_to_reference(
            _ds(["2021-01-01"], lons=just_over), {"lon": ref}, tol=1e-8
        )
        assert set(snapped) == {"lon"}


class TestDepthIsSnappedToo:
    """
    The o2 store's 2023+ files label the same 46 CMEMS levels up to 6.1e-05 m
    away from the older files — one float32 ULP at ~900 m, a relabel and not a
    regrid. Reading the store whole failed with "cannot align objects with
    join='exact' ... 'depth'", because depth is not the axis being concatenated
    along, so the mismatch surfaced as an alignment failure rather than as the
    monotonicity complaint lat/lon give.

    depth is metres while lat/lon are degrees, which is why the tolerance is
    per coordinate: the degrees limit (1e-9) rejects this by six orders of
    magnitude.
    """

    #: Levels 36-38 of the real o2 axis, in both of the store's spellings.
    OLD = np.array([846.7606201171875, 894.9822387695312, 947.4478759765625])
    NEW = np.array([846.7606201171875, 894.9822998046875, 947.4478149414062])

    @staticmethod
    def _ds_with_depth(depths) -> xr.Dataset:
        return xr.Dataset(
            {"o2": (("depth",), np.arange(len(depths), dtype=float))},
            coords={"depth": np.asarray(depths, dtype=float)},
        )

    def test_the_real_o2_drift_is_snapped(self):
        out, snapped = snap_axes_to_reference(
            self._ds_with_depth(self.NEW), {"depth": self.OLD}
        )

        assert set(snapped) == {"depth"}
        assert snapped["depth"] == pytest.approx(6.1e-05, rel=0.05)
        np.testing.assert_array_equal(out.depth.values, self.OLD)

    def test_a_genuinely_different_level_is_refused(self):
        """A metre is a different level, not a relabel — it must still raise."""
        shifted = self.OLD + 1.0

        out, snapped = snap_axes_to_reference(
            self._ds_with_depth(shifted), {"depth": self.OLD}
        )

        assert snapped == {}
        np.testing.assert_array_equal(out.depth.values, shifted)

    def test_the_degrees_tolerance_would_have_refused_it(self):
        """Pins why the tolerance had to become per coordinate."""
        _, snapped = snap_axes_to_reference(
            self._ds_with_depth(self.NEW), {"depth": self.OLD}, tol=AXIS_SNAP_TOL["lon"]
        )

        assert snapped == {}

    def test_depth_is_one_of_the_axes_the_reader_collects(self):
        assert "depth" in AXIS_SNAP_TOL


class TestDriftedStoreIsReadable:
    """
    Regression, end to end through xarray: two files whose axes differ only by
    float noise must combine. Without the snap this raises "Resulting object
    does not have monotonic global indexes along dimension lon".
    """

    @staticmethod
    def _write_pair(tmp_path, drift: float):
        lons = np.array([-10.0, -5.0, 0.0])
        a = _ds(["2021-01-01", "2021-01-02"], lons=lons)
        b = _ds(["2021-01-03", "2021-01-04"], lons=lons + np.linspace(0, drift, 3))
        pa, pb = tmp_path / "a.zarr", tmp_path / "b.zarr"
        # Written the way the pipeline writes: to_zarr's default consolidates
        # metadata. Passing consolidated=False here made the read below fall
        # back to non-consolidated metadata and warn about it — a warning about
        # the fixture, not about the drift these tests are pinning.
        a.to_zarr(pa)
        b.to_zarr(pb)
        return [str(pa), str(pb)]

    _DATES = ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]

    def _open(self, paths):
        """
        Open through ``ZarrReader.open_dataset`` — the real production call.

        This used to re-implement the ``open_mfdataset`` invocation inline,
        which meant it asserted against a copy of the reader's arguments rather
        than the reader's. Reverting ``join`` in zarr_reader.py left every test
        in this class green. Driving the reader keeps them honest; the index is
        stubbed only to supply the paths, which is all this path needs.
        """
        index = MagicMock()
        index.var_key = "sst"
        index.map_dates_to_paths.return_value = {
            p: [pd.Timestamp(d) for d in self._DATES] for p in paths
        }
        return ZarrReader(index).open_dataset(dates=self._DATES)

    def test_float_drifted_files_combine(self, tmp_path):
        paths = self._write_pair(tmp_path, drift=1e-12)

        with self._open(paths) as ds:
            assert ds.sizes["time"] == 4
            assert ds.sizes["lon"] == 3  # not doubled by an outer join

    def test_the_same_files_are_refused_without_the_snap(self, tmp_path):
        """
        Pins what the snap buys. Without it the two axes are not equal, and
        under join='exact' xarray refuses to align them rather than inventing a
        combined axis. Across a real store of many files the same mismatch used
        to surface as "Resulting object does not have monotonic global indexes
        along dimension lon", which names neither the cause nor the file.

        Was asserted the other way round while the reader relied on xarray's
        old join='outer' default: the read succeeded and handed back a doubled
        axis of near-duplicate points. Failing is the better half of that pair.
        """
        paths = self._write_pair(tmp_path, drift=1e-12)

        # The reader always computes a reference, so the snap is disabled here
        # rather than skipped — the point is what the *rest* of the read does
        # when the axes are left as written.
        with patch.object(ZarrReader, "_reference_axes", staticmethod(lambda _: {})):
            with pytest.raises(RuntimeError, match="(?i)align|monotonic"):
                self._open(paths).close()

    def test_a_genuinely_different_grid_is_refused_even_with_the_snap(self, tmp_path):
        """
        The promise ``snap_axes_to_reference`` makes in its own docstring — "a
        genuinely different grid ... still reaches xarray and still raises" —
        and which nothing verified.

        It was not true under join='outer': two grids 5 degrees apart (far
        outside the snap tolerance, so the snap correctly declines) unioned
        into a longer axis of interleaved points and the read *succeeded*,
        handing back a coordinate axis that was not either file's grid.
        """
        paths = self._write_pair(tmp_path, drift=5.0)

        with pytest.raises(RuntimeError, match="(?i)align|monotonic"):
            self._open(paths).close()

    def test_reference_axes_needs_more_than_one_file(self, tmp_path):
        """A lone file is never combined against anything; it defines its axes."""
        paths = self._write_pair(tmp_path, drift=1e-12)

        assert ZarrReader._reference_axes(paths[:1]) == {}
        assert set(ZarrReader._reference_axes(paths)) == {"lat", "lon"}

    def test_reference_axes_survives_an_unreadable_path(self):
        """Advisory only — a failure here must fall back, not break the read."""
        assert ZarrReader._reference_axes(["/no/such/a.zarr", "/no/such/b.zarr"]) == {}


def _ds_depth(times, depths, variables=("o2",)) -> xr.Dataset:
    """time × depth dataset, standing in for a depth-resolved store."""
    index = pd.DatetimeIndex(times)
    shape = (len(index), len(depths))
    size = int(np.prod(shape))
    return xr.Dataset(
        {
            v: (("time", "depth"), np.arange(size, dtype=float).reshape(shape))
            for v in variables
        },
        coords={"time": index, "depth": np.asarray(depths, dtype=float)},
    )


class TestDepthDriftedStoreIsReadable:
    """
    Regression, end to end: the o2 store, whose 2023+ files label the same 46
    CMEMS levels one float32 ULP away from the older ones. Reading it whole
    raised "cannot align objects with join='exact' ... 'depth'" — depth is not
    the concat axis, so the mismatch reached alignment rather than the
    monotonicity check lat/lon drift trips.
    """

    _DATES = ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]
    _OLD = TestDepthIsSnappedToo.OLD
    _NEW = TestDepthIsSnappedToo.NEW

    def _write_pair(self, tmp_path, second_depths):
        a = _ds_depth(self._DATES[:2], self._OLD)
        b = _ds_depth(self._DATES[2:], second_depths)
        pa, pb = tmp_path / "a.zarr", tmp_path / "b.zarr"
        a.to_zarr(pa)
        b.to_zarr(pb)
        return [str(pa), str(pb)]

    def _open(self, paths):
        index = MagicMock()
        index.var_key = "o2"
        index.map_dates_to_paths.return_value = {
            p: [pd.Timestamp(d) for d in self._DATES] for p in paths
        }
        return ZarrReader(index).open_dataset(dates=self._DATES)

    def test_ulp_drifted_depth_files_combine(self, tmp_path):
        paths = self._write_pair(tmp_path, self._NEW)

        with self._open(paths) as ds:
            assert ds.sizes["time"] == 4
            assert ds.sizes["depth"] == 3  # one axis, not two concatenated
            np.testing.assert_array_equal(ds.depth.values, self._OLD)

    def test_the_same_files_are_refused_without_the_snap(self, tmp_path):
        """Pins what the snap buys: this is the failure the o2 read hit."""
        paths = self._write_pair(tmp_path, self._NEW)

        with patch.object(ZarrReader, "_reference_axes", staticmethod(lambda _: {})):
            with pytest.raises(RuntimeError, match="(?i)align|monotonic"):
                self._open(paths).close()

    def test_a_genuinely_different_level_is_refused_even_with_the_snap(self, tmp_path):
        paths = self._write_pair(tmp_path, self._OLD + 1.0)

        with pytest.raises(RuntimeError, match="(?i)align|monotonic"):
            self._open(paths).close()


class TestRaggedVariableSetsAreReadable:
    """
    Regression, end to end: h2ds is ragged by design — ``run -v X`` compiles
    only X's columns, and a source that does not reach the current year yet
    leaves that year's file short. combine_by_coords groups datasets *by their
    set of data variables* before combining anything, so the store arrived at
    the final merge as two cubes covering different periods and join="exact"
    refused to align them: "cannot align objects ... (dimensions): 'time'",
    naming the axis the cubes differ on rather than the variables that split
    them.
    """

    _DATES = ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]

    def _write_pair(self, tmp_path, second_vars=("sst",)):
        a = _ds(self._DATES[:2], variables=("sst", "npp"))
        b = _ds(self._DATES[2:], variables=second_vars)
        pa, pb = tmp_path / "a.zarr", tmp_path / "b.zarr"
        a.to_zarr(pa)
        b.to_zarr(pb)
        return [str(pa), str(pb)]

    def _open(self, paths, **kwargs):
        index = MagicMock()
        index.var_key = "h2ds"
        index.map_dates_to_paths.return_value = {
            p: [pd.Timestamp(d) for d in self._DATES] for p in paths
        }
        return ZarrReader(index).open_dataset(dates=self._DATES, **kwargs)

    def test_a_short_file_does_not_break_the_read(self, tmp_path):
        paths = self._write_pair(tmp_path)

        with self._open(paths) as ds:
            assert ds.sizes["time"] == 4
            assert set(ds.data_vars) == {"sst", "npp"}

    def test_the_padded_variable_reads_back_as_nan_only_where_it_is_absent(
        self, tmp_path
    ):
        """
        The truthful answer, and the one join='outer' used to give: real values
        where the variable was written, NaN for the period it never covered.
        """
        paths = self._write_pair(tmp_path)

        with self._open(paths) as ds:
            npp = ds["npp"]
            assert not np.isnan(npp.sel(time=self._DATES[:2]).values).any()
            assert np.isnan(npp.sel(time=self._DATES[2:]).values).all()
            # The variable both files carry is untouched by the padding.
            assert not np.isnan(ds["sst"].values).any()

    def test_the_same_files_are_refused_without_the_padding(self, tmp_path):
        """Pins what the padding buys — this is the h2ds failure verbatim."""
        paths = self._write_pair(tmp_path)

        with patch.object(
            ZarrReader, "_reference_data_vars", staticmethod(lambda _: {})
        ):
            with pytest.raises(RuntimeError, match="(?i)align"):
                self._open(paths).close()

    def test_a_requested_variable_is_not_padded_back_in(self, tmp_path):
        """
        Padding to the full union would undo the caller's selection, and
        re-split the very group it exists to keep whole.
        """
        paths = self._write_pair(tmp_path)

        with self._open(paths, variables=["sst"]) as ds:
            assert set(ds.data_vars) == {"sst"}
            assert ds.sizes["time"] == 4

    def test_reference_data_vars_needs_more_than_one_file(self, tmp_path):
        """A lone file is never grouped against anything."""
        paths = self._write_pair(tmp_path)

        assert ZarrReader._reference_data_vars(paths[:1]) == {}
        assert set(ZarrReader._reference_data_vars(paths)) == {"sst", "npp"}

    def test_reference_data_vars_survives_an_unreadable_path(self):
        """Advisory only — a failure here must fall back, not break the read."""
        assert (
            ZarrReader._reference_data_vars(["/no/such/a.zarr", "/no/such/b.zarr"])
            == {}
        )
