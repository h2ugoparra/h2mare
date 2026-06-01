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
    find_nearest_vectorized,
    process_fsle,
)
from h2mare.types import DateRange, TimeResolution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EDDIES_ENTRY = {
    "local_folder": "eddies",
    "variables": [
        "track",
        "effective_radius",
        "speed_radius",
        "amplitude",
        "speed_average",
        "observation_number",
    ],
    "dataset_id_rep": "META3.2_ALLSAT_PHY_L4_REP",
    "source": "aviso",
    "pattern": r"(\d{8})_(\d{8})",
    "subset": False,
    "bbox": (-80, 0, 10, 70),
}

_FSLE_ENTRY = {
    "local_folder": "fsle",
    "variables": ["fsle_max"],
    "dataset_id_rep": "META_ALT_FSLE_OBS_010_006",
    "source": "aviso",
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
        groups = dict(_group_dates(dates, TimeResolution.YEAR))
        assert 2020 in groups and 2021 in groups
        total = sum(len(v) for v in groups.values())
        assert total == len(dates)

    def test_month_grouping_separates_months(self):
        dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        groups = dict(_group_dates(dates, TimeResolution.MONTH))
        assert (2020, 1) in groups
        assert (2020, 2) in groups
        assert (2020, 3) in groups
        assert len(groups[(2020, 1)]) == 31
        assert len(groups[(2020, 2)]) == 29  # 2020 is a leap year

    def test_year_group_key_is_integer(self):
        dates = pd.date_range("2021-06-01", "2021-06-30", freq="D")
        groups = dict(_group_dates(dates, TimeResolution.YEAR))
        assert isinstance(list(groups.keys())[0], int)

    def test_month_group_key_is_tuple(self):
        dates = pd.date_range("2021-06-01", "2021-06-30", freq="D")
        groups = dict(_group_dates(dates, TimeResolution.MONTH))
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
        var_config.variables = ["fsle_max"]
        var_config.bbox = (-10, 30, 20, 50)

        result = process_fsle(ds, var_config)

        assert float(result.lon.min()) >= -10
        assert float(result.lon.max()) <= 20
        assert float(result.lat.min()) >= 30
        assert float(result.lat.max()) <= 50

    def test_lon_converted_from_360_to_180(self):
        ds = self._make_fsle_ds()
        var_config = MagicMock()
        var_config.variables = ["fsle_max"]
        var_config.bbox = (-180, -90, 180, 90)

        result = process_fsle(ds, var_config)
        assert float(result.lon.min()) >= -180
        assert float(result.lon.max()) <= 180

    def test_only_selected_variable_in_output(self):
        ds = self._make_fsle_ds()
        ds["extra_var"] = ds["fsle_max"] * 2
        var_config = MagicMock()
        var_config.variables = ["fsle_max"]
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

    def test_raises_when_no_overlap(self, eddies_proc):
        download_range = DateRange("2000-01-01", "2005-12-31")
        requested = DateRange("2020-01-01", "2020-12-31")

        with patch(
            "h2mare.processing.core.aviso.resolve_date_range",
            return_value=requested,
        ):
            with pytest.raises(ValueError):
                eddies_proc._resolve_date_range(download_range)
