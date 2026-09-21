"""Tests for processing/core/aviso.py — pure functions and EDDIESProcessor helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.models import AppConfig
from h2mare.processing.core.aviso import (
    EDDIESProcessor,
    _group_dates,
    _is_degenerate_axis,
    find_nearest_vectorized,
    process_fsle,
)
from h2mare.types import DateRange, FilePeriod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EDDIES_ENTRY = {
    "local_folder": "eddies",
    "source_vars": [
        "track",
        "effective_radius",
        "speed_radius",
        "amplitude",
        "speed_average",
        "observation_number",
    ],
    "dataset_id_rep": "META3.2_ALLSAT_PHY_L4_REP",
    "source": "aviso",
    "archive_raw": True,
    "pattern": r"(\d{8})_(\d{8})",
    "subset": False,
    "bbox": (-80, 0, 10, 70),
}

_FSLE_ENTRY = {
    "local_folder": "fsle",
    "source_vars": ["fsle_max"],
    "dataset_id_rep": "META_ALT_FSLE_OBS_010_006",
    "source": "aviso",
    "archive_raw": True,
    "pattern": r"(\d{4})(\d{2})(\d{2})",
    "subset": False,
    "bbox": (-10, 30, 20, 50),
}


def _make_config(var_key: str = "eddies", entry: dict = _EDDIES_ENTRY) -> AppConfig:
    return msgspec.convert({"variables": {var_key: entry}, "secrets": {}}, AppConfig)


@pytest.fixture
def eddies_proc(tmp_path):
    """EDDIESProcessor with ZarrCatalog mocked out."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    with patch("h2mare.processing.core.aviso.ZarrCatalog") as MockCat:
        MockCat.return_value.exists.return_value = False
        return EDDIESProcessor(
            var_key="eddies",
            app_config=_make_config(),
            store_root=store_dir,
            download_root=download_dir,
        )


# ---------------------------------------------------------------------------
# find_nearest_vectorized
# ---------------------------------------------------------------------------


class TestFindNearestVectorized:
    def test_single_query_finds_nearest_target(self):
        query_lats = np.array([40.0])
        query_lons = np.array([-10.0])
        target_lats = np.array([40.1, 45.0, 30.0])
        target_lons = np.array([-10.1, 0.0, -20.0])
        indices = find_nearest_vectorized(
            query_lats, query_lons, target_lats, target_lons
        )
        assert indices[0] == 0  # (40.1, -10.1) is nearest to (40.0, -10.0)

    def test_multiple_queries(self):
        # Two query points, each closest to a different target
        query_lats = np.array([0.0, 50.0])
        query_lons = np.array([0.0, 10.0])
        target_lats = np.array([0.1, 50.1])
        target_lons = np.array([0.1, 10.1])
        indices = find_nearest_vectorized(
            query_lats, query_lons, target_lats, target_lons
        )
        assert indices[0] == 0
        assert indices[1] == 1

    def test_output_shape_matches_query_count(self):
        lats = np.random.uniform(-90, 90, 20)
        lons = np.random.uniform(-180, 180, 20)
        target_lats = np.random.uniform(-90, 90, 5)
        target_lons = np.random.uniform(-180, 180, 5)
        result = find_nearest_vectorized(lats, lons, target_lats, target_lons)
        assert result.shape == (20,)
        assert result.max() < 5


# ---------------------------------------------------------------------------
# _group_dates
# ---------------------------------------------------------------------------


class TestGroupDates:
    def test_year_grouping_covers_all_dates(self):
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        groups = dict(_group_dates(dates, FilePeriod.YEAR))
        assert 2020 in groups and 2021 in groups
        total = sum(len(v) for v in groups.values())
        assert total == len(dates)

    def test_month_grouping_separates_months(self):
        dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        groups = dict(_group_dates(dates, FilePeriod.MONTH))
        assert (2020, 1) in groups
        assert (2020, 2) in groups
        assert (2020, 3) in groups
        assert len(groups[(2020, 1)]) == 31
        assert len(groups[(2020, 2)]) == 29  # 2020 is a leap year

    def test_year_group_key_is_integer(self):
        dates = pd.date_range("2021-06-01", "2021-06-30", freq="D")
        groups = dict(_group_dates(dates, FilePeriod.YEAR))
        assert isinstance(list(groups.keys())[0], int)

    def test_month_group_key_is_tuple(self):
        dates = pd.date_range("2021-06-01", "2021-06-30", freq="D")
        groups = dict(_group_dates(dates, FilePeriod.MONTH))
        assert isinstance(list(groups.keys())[0], tuple)


# ---------------------------------------------------------------------------
# process_fsle
# ---------------------------------------------------------------------------


class TestProcessFsle:
    def _make_fsle_ds(self) -> xr.Dataset:
        """Dataset with global coverage, lon in 0-360."""
        lons = np.arange(0, 360, 1.0)  # 360 points, 0–359
        lats = np.arange(-90, 91, 1.0)  # 181 points, -90–90
        data = np.random.default_rng(0).uniform(0.1, 10.0, (181, 360))
        return xr.Dataset(
            {"fsle_max": (["lat", "lon"], data)},
            coords={"lat": lats, "lon": lons},
        )

    def test_output_clipped_to_bbox(self):
        ds = self._make_fsle_ds()
        var_config = MagicMock()
        var_config.source_vars = ["fsle_max"]
        var_config.bbox = (-10, 30, 20, 50)

        result = process_fsle(ds, var_config)

        assert float(result.lon.min()) >= -10
        assert float(result.lon.max()) <= 20
        assert float(result.lat.min()) >= 30
        assert float(result.lat.max()) <= 50

    def test_lon_converted_from_360_to_180(self):
        ds = self._make_fsle_ds()
        var_config = MagicMock()
        var_config.source_vars = ["fsle_max"]
        var_config.bbox = (-180, -90, 180, 90)

        result = process_fsle(ds, var_config)
        assert float(result.lon.min()) >= -180
        assert float(result.lon.max()) <= 180

    def test_only_selected_variable_in_output(self):
        ds = self._make_fsle_ds()
        ds["extra_var"] = ds["fsle_max"] * 2
        var_config = MagicMock()
        var_config.source_vars = ["fsle_max"]
        var_config.bbox = (-10, 30, 20, 50)

        result = process_fsle(ds, var_config)
        assert "fsle_max" in result
        assert "extra_var" not in result


# ---------------------------------------------------------------------------
# EDDIESProcessor._get_downloaded_metadata
# ---------------------------------------------------------------------------


class TestGetDownloadedMetadata:
    def _create_eddy_files(self, root: Path, dates: str = "20210101_20211231") -> None:
        for eddy_type in ("anticyclonic", "cyclonic"):
            (root / f"META_{eddy_type}_{dates}.nc").touch()

    def test_returns_one_record_per_file(self, eddies_proc, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        self._create_eddy_files(root)
        records = eddies_proc._get_downloaded_metadata(root_dir=root)
        assert len(records) == 2

    def test_parses_eddy_type_and_date_range(self, eddies_proc, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (root / "META_anticyclonic_20210101_20211231.nc").touch()
        records = eddies_proc._get_downloaded_metadata(root_dir=root)
        eddy_type, date_range, path = records[0]
        assert eddy_type == "anticyclonic"
        assert pd.Timestamp(date_range.start).year == 2021
        assert pd.Timestamp(date_range.end).year == 2021

    def test_raises_when_no_files_found(self, eddies_proc, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            eddies_proc._get_downloaded_metadata(root_dir=empty)


# ---------------------------------------------------------------------------
# EDDIESProcessor._resolve_date_range
# ---------------------------------------------------------------------------


class TestEddiesResolveRange:
    def test_returns_intersection_of_requested_and_available(self, eddies_proc):
        download_range = DateRange("2020-01-01", "2021-12-31")
        requested = DateRange("2020-06-01", "2022-06-30")

        with patch(
            "h2mare.processing.core.aviso.resolve_date_range",
            return_value=requested,
        ):
            result = eddies_proc._resolve_date_range(download_range)

        # Intersection: 2020-06-01 to 2021-12-31
        assert pd.Timestamp(result.start) == pd.Timestamp("2020-06-01")
        assert pd.Timestamp(result.end) == pd.Timestamp("2021-12-31")

    def test_returns_none_when_no_overlap(self, eddies_proc):
        """A file outside the requested window is skipped, not fatal.

        It used to raise, which meant one irrelevant file in the raw directory
        (a rep file when asking for an nrt window, say) aborted the whole run.
        """
        download_range = DateRange("2000-01-01", "2005-12-31")
        requested = DateRange("2020-01-01", "2020-12-31")

        with patch(
            "h2mare.processing.core.aviso.resolve_date_range",
            return_value=requested,
        ):
            assert eddies_proc._resolve_date_range(download_range) is None


class TestEddiesResolveAllRanges:
    """Windows are resolved per file, not per eddy type."""

    def _records(self, tmp_path):
        # Same shape as a real store: several files per type, spanning
        # different periods — rep variants, nrt, and an unused product version.
        return [
            (
                "anticyclonic",
                DateRange("1993-01-01", "2022-02-09"),
                tmp_path / "META3.2_DT_allsat_Anticyclonic_long.nc",
            ),
            (
                "anticyclonic",
                DateRange("2018-01-01", "2026-07-13"),
                tmp_path / "Eddy_trajectory_nrt_anticyclonic.nc",
            ),
            (
                "cyclonic",
                DateRange("2018-01-01", "2026-07-13"),
                tmp_path / "Eddy_trajectory_nrt_cyclonic.nc",
            ),
        ]

    def test_keyed_by_path_not_eddy_type(self, eddies_proc, tmp_path):
        """Regression: two anticyclonic files collapsed into one entry."""
        records = self._records(tmp_path)

        with patch(
            "h2mare.processing.core.aviso.resolve_date_range",
            return_value=DateRange("1993-01-01", "2026-07-13"),
        ):
            resolved = eddies_proc._resolve_all_ranges(records, None, None)

        assert set(resolved) == {path for _, _, path in records}

    def test_non_overlapping_files_are_dropped(self, eddies_proc, tmp_path):
        """Asking for an nrt window must not abort on the rep files present."""
        records = self._records(tmp_path)

        with patch(
            "h2mare.processing.core.aviso.resolve_date_range",
            return_value=DateRange("2026-04-21", "2026-07-13"),
        ):
            resolved = eddies_proc._resolve_all_ranges(records, None, None)

        assert set(resolved) == {
            tmp_path / "Eddy_trajectory_nrt_anticyclonic.nc",
            tmp_path / "Eddy_trajectory_nrt_cyclonic.nc",
        }

    def test_windows_are_clipped_per_file(self, eddies_proc, tmp_path):
        records = self._records(tmp_path)

        with patch(
            "h2mare.processing.core.aviso.resolve_date_range",
            return_value=DateRange("2020-01-01", "2026-07-13"),
        ):
            resolved = eddies_proc._resolve_all_ranges(records, None, None)

        rep = resolved[tmp_path / "META3.2_DT_allsat_Anticyclonic_long.nc"]
        nrt = resolved[tmp_path / "Eddy_trajectory_nrt_anticyclonic.nc"]
        assert pd.Timestamp(rep.end) == pd.Timestamp("2022-02-09")
        assert pd.Timestamp(nrt.end) == pd.Timestamp("2026-07-13")

    def test_no_overlap_at_all_yields_empty(self, eddies_proc, tmp_path):
        records = self._records(tmp_path)

        with patch(
            "h2mare.processing.core.aviso.resolve_date_range",
            return_value=DateRange("1900-01-01", "1900-12-31"),
        ):
            assert eddies_proc._resolve_all_ranges(records, None, None) == {}


# ---------------------------------------------------------------------------
# Store grid resolution
#
# _get_gridded_data used to read the grid from catalog.open_dataset(), which
# with no arguments opens every file and unions their coordinate axes. Axes
# differing only in the last floating-point bits merged into a doubled axis of
# near-duplicate points, and because the result is written back to the store the
# corruption compounded on every run.
# ---------------------------------------------------------------------------


def _canonical_axes():
    """The store's established 0.1-degree grid."""
    return np.arange(0.0, 5.0001, 0.1), np.arange(-10.0, 0.0001, 0.1)


def _unioned(values):
    """Axis as open_mfdataset would union it: each point plus a 1-ULP twin."""
    twins = np.nextafter(values, values + 1)
    return np.unique(np.concatenate([values, twins]))


@pytest.fixture
def proc_with_store(tmp_path):
    """EDDIESProcessor whose catalog reports an existing store."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    with patch("h2mare.processing.core.aviso.ZarrCatalog") as MockCat:
        cat = MockCat.return_value
        cat.exists.return_value = True
        cat.df = pd.DataFrame(
            [
                {
                    "path": str(store_dir / "a.zarr"),
                    "start_date": pd.Timestamp("2020-01-01"),
                },
                {
                    "path": str(store_dir / "b.zarr"),
                    "start_date": pd.Timestamp("2021-01-01"),
                },
            ]
        )
        proc = EDDIESProcessor(
            var_key="eddies",
            # Grid lines, like the store on disk: the store grid is reused only
            # when it is the configured one.
            app_config=_make_config(entry={**_EDDIES_ENTRY, "values_at": "grid_line"}),
            store_root=store_dir,
            download_root=download_dir,
        )
    return proc, cat


class TestIsDegenerateAxis:
    def test_uniform_grid_is_fine(self):
        assert not _is_degenerate_axis(np.arange(0.0, 10.0, 0.1))

    def test_native_irregular_grid_is_fine(self):
        """A 1/12-degree grid is not exactly representable but is not degenerate."""
        assert not _is_degenerate_axis(np.arange(0.0, 20.0, 1 / 12))

    def test_unioned_axis_is_degenerate(self):
        lat, _ = _canonical_axes()
        assert _is_degenerate_axis(_unioned(lat))

    def test_exact_duplicates_are_degenerate(self):
        assert _is_degenerate_axis(np.array([0.0, 0.1, 0.1, 0.2]))

    def test_tiny_axis_is_not_flagged(self):
        assert not _is_degenerate_axis(np.array([0.0, 1.0]))


class TestGridFromStore:
    def test_does_not_open_the_whole_store(self, proc_with_store):
        """Regression: open_dataset() unions every file's axes."""
        proc, cat = proc_with_store
        lat, lon = _canonical_axes()
        with patch(
            "h2mare.processing.core.aviso.xr.open_zarr",
            return_value=xr.Dataset(coords={"lat": lat, "lon": lon}),
        ):
            proc._grid_from_store()

        cat.open_dataset.assert_not_called()

    def test_reads_the_earliest_file(self, proc_with_store):
        proc, _ = proc_with_store
        lat, lon = _canonical_axes()
        with patch(
            "h2mare.processing.core.aviso.xr.open_zarr",
            return_value=xr.Dataset(coords={"lat": lat, "lon": lon}),
        ) as mock_open:
            proc._grid_from_store()

        assert mock_open.call_args[0][0].endswith("a.zarr")

    def test_returns_the_store_grid(self, proc_with_store):
        proc, _ = proc_with_store
        lat, lon = _canonical_axes()
        with patch(
            "h2mare.processing.core.aviso.xr.open_zarr",
            return_value=xr.Dataset(coords={"lat": lat, "lon": lon}),
        ):
            got_lat, got_lon = proc._grid_from_store()

        assert np.array_equal(got_lat, lat)
        assert np.array_equal(got_lon, lon)

    def test_degenerate_store_grid_is_rejected(self, proc_with_store):
        """A doubled axis must not be adopted as the grid."""
        proc, _ = proc_with_store
        lat, lon = _canonical_axes()
        with patch(
            "h2mare.processing.core.aviso.xr.open_zarr",
            return_value=xr.Dataset(coords={"lat": _unioned(lat), "lon": lon}),
        ):
            assert proc._grid_from_store() is None

    def test_empty_catalog_returns_none(self, proc_with_store):
        proc, cat = proc_with_store
        cat.df = pd.DataFrame()

        assert proc._grid_from_store() is None

    def test_unreadable_file_returns_none(self, proc_with_store):
        proc, _ = proc_with_store
        with patch(
            "h2mare.processing.core.aviso.xr.open_zarr",
            side_effect=OSError("corrupt store"),
        ):
            assert proc._grid_from_store() is None


class TestGetGriddedDataFallback:
    def test_does_not_open_the_whole_store(self, proc_with_store):
        """Regression, asserted through the entry point that predates the fix.

        _get_gridded_data used to call catalog.open_dataset() with no arguments,
        which opens every file and unions their coordinate axes.
        """
        proc, cat = proc_with_store
        lat, lon = _canonical_axes()
        with patch(
            "h2mare.processing.core.aviso.xr.open_zarr",
            return_value=xr.Dataset(coords={"lat": lat, "lon": lon}),
        ):
            proc._get_gridded_data(dx=0.1, dy=0.1)

        cat.open_dataset.assert_not_called()

    def test_falls_back_to_bbox_when_store_grid_is_degenerate(self, proc_with_store):
        """The configured bbox wins over a corrupt store grid."""
        proc, _ = proc_with_store
        lat, lon = _canonical_axes()
        with patch(
            "h2mare.processing.core.aviso.xr.open_zarr",
            return_value=xr.Dataset(coords={"lat": _unioned(lat), "lon": lon}),
        ):
            grid = proc._get_gridded_data(dx=0.5, dy=0.5)

        assert not _is_degenerate_axis(grid.lat)
        assert not _is_degenerate_axis(grid.lon)


class TestGetGriddedDataConfiguredGrid:
    """
    The store grid used to win over config unconditionally, so regenerating the
    store after changing ``cells_per_degree`` rebuilt it on its old grid, and
    the full-period rewrite skipped the write path's grid check.
    """

    def test_changed_resolution_raises(self, proc_with_store):
        proc, _ = proc_with_store
        lat, lon = _canonical_axes()
        with patch(
            "h2mare.processing.core.aviso.xr.open_zarr",
            return_value=xr.Dataset(coords={"lat": lat, "lon": lon}),
        ):
            with pytest.raises(ValueError, match="different grid than config"):
                proc._get_gridded_data(dx=1 / 12, dy=1 / 12)

    def test_matching_store_grid_is_reused_as_is(self, proc_with_store):
        """Same lattice, drifted in the last bits: the store's axes are kept."""
        proc, _ = proc_with_store
        lat, lon = _canonical_axes()
        lat, lon = np.nextafter(lat, lat + 1), np.nextafter(lon, lon + 1)
        with patch(
            "h2mare.processing.core.aviso.xr.open_zarr",
            return_value=xr.Dataset(coords={"lat": lat, "lon": lon}),
        ):
            grid = proc._get_gridded_data(dx=0.1, dy=0.1)

        assert np.array_equal(grid.lat, lat)
        assert np.array_equal(grid.lon, lon)


# ---------------------------------------------------------------------------
# rep beats nrt in the overlap
#
# AVISO's nrt trajectory file spans 2018 to today while META3.2 delayed-time
# runs to 2022, so the two overlap by four years. The downloader already starts
# nrt the day after rep ends; a conversion reads whatever is on disk, so it
# needs the same rule or both sources land in the overlap.
# ---------------------------------------------------------------------------


class TestPreferRep:
    def _records(self, tmp_path):
        rep_dir = tmp_path / "rep"
        nrt_dir = tmp_path / "nrt"
        rep_dir.mkdir(exist_ok=True)
        nrt_dir.mkdir(exist_ok=True)
        return [
            ("cyclonic", DateRange("1993-01-01", "2022-02-09"), rep_dir / "rep_c.nc"),
            ("cyclonic", DateRange("2018-01-01", "2026-07-13"), nrt_dir / "nrt_c.nc"),
        ]

    def _resolve(self, proc, records, requested):
        with patch(
            "h2mare.processing.core.aviso.resolve_date_range", return_value=requested
        ):
            return proc._resolve_all_ranges(records, None, None)

    def test_nrt_starts_after_rep_ends(self, eddies_proc, tmp_path):
        records = self._records(tmp_path)

        out = self._resolve(eddies_proc, records, DateRange("2020-01-01", "2026-07-13"))

        nrt = out[tmp_path / "nrt" / "nrt_c.nc"]
        assert pd.Timestamp(nrt.start) == pd.Timestamp("2022-02-10")
        assert pd.Timestamp(nrt.end) == pd.Timestamp("2026-07-13")

    def test_rep_window_is_untouched(self, eddies_proc, tmp_path):
        records = self._records(tmp_path)

        out = self._resolve(eddies_proc, records, DateRange("2020-01-01", "2026-07-13"))

        rep = out[tmp_path / "rep" / "rep_c.nc"]
        assert pd.Timestamp(rep.start) == pd.Timestamp("2020-01-01")
        assert pd.Timestamp(rep.end) == pd.Timestamp("2022-02-09")

    def test_request_fully_inside_rep_drops_nrt(self, eddies_proc, tmp_path):
        """Regression: nrt used to contribute to a window rep already covers."""
        records = self._records(tmp_path)

        out = self._resolve(eddies_proc, records, DateRange("2019-01-01", "2019-12-31"))

        assert tmp_path / "nrt" / "nrt_c.nc" not in out
        assert tmp_path / "rep" / "rep_c.nc" in out

    def test_request_after_rep_leaves_nrt_alone(self, eddies_proc, tmp_path):
        records = self._records(tmp_path)

        out = self._resolve(eddies_proc, records, DateRange("2026-04-21", "2026-07-13"))

        nrt = out[tmp_path / "nrt" / "nrt_c.nc"]
        assert pd.Timestamp(nrt.start) == pd.Timestamp("2026-04-21")
        assert tmp_path / "rep" / "rep_c.nc" not in out

    def test_clipping_is_per_eddy_type(self, eddies_proc, tmp_path):
        """An anticyclonic rep file must not clip the cyclonic nrt window."""
        rep_dir, nrt_dir = tmp_path / "rep", tmp_path / "nrt"
        rep_dir.mkdir(exist_ok=True)
        nrt_dir.mkdir(exist_ok=True)
        records = [
            (
                "anticyclonic",
                DateRange("1993-01-01", "2022-02-09"),
                rep_dir / "rep_ac.nc",
            ),
            ("cyclonic", DateRange("2018-01-01", "2026-07-13"), nrt_dir / "nrt_c.nc"),
        ]

        out = self._resolve(eddies_proc, records, DateRange("2020-01-01", "2026-07-13"))

        nrt = out[nrt_dir / "nrt_c.nc"]
        assert pd.Timestamp(nrt.start) == pd.Timestamp("2020-01-01")

    def test_flat_layout_is_left_alone(self, eddies_proc, tmp_path):
        """No rep/nrt directory means no way to tell — do not guess."""
        records = [
            ("cyclonic", DateRange("1993-01-01", "2022-02-09"), tmp_path / "a_c.nc"),
            ("cyclonic", DateRange("2018-01-01", "2026-07-13"), tmp_path / "b_c.nc"),
        ]

        out = self._resolve(eddies_proc, records, DateRange("2020-01-01", "2026-07-13"))

        assert len(out) == 2
        assert pd.Timestamp(out[tmp_path / "b_c.nc"].start) == pd.Timestamp(
            "2020-01-01"
        )
