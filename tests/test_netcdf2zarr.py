"""Tests for format_converters/netcdf2zarr.py — Netcdf2Zarr class."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.format_converters.netcdf2zarr import Netcdf2Zarr
from h2mare.models import AppConfig
from h2mare.types import TimeResolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SST_ENTRY_SUBSET = {
    "local_folder": "sst",
    "variables": ["analysed_sst"],
    "dataset_id_rep": "cmems-rep-sst",
    "source": "cmems",
    "pattern": r"(\d{8})_(\d{8})",
    "subset": True,
    "bbox": (-80, 0, 10, 70),
}

_SST_ENTRY_SINGLE = {
    **_SST_ENTRY_SUBSET,
    "pattern": r"(\d{4})(\d{2})(\d{2})",
    "subset": False,
}

_MLD_ENTRY = {
    "local_folder": "mld",
    "variables": ["mlotst"],
    "dataset_id_rep": "cmems-mld",
    "source": "cmems",
    "pattern": r"(\d{8})_(\d{8})",
    "subset": True,
}


def _make_config(var_key: str = "sst", entry: dict = _SST_ENTRY_SUBSET) -> AppConfig:
    return msgspec.convert({"variables": {var_key: entry}, "secrets": {}}, AppConfig)


def _make_converter(
    tmp_path: Path,
    var_key: str = "sst",
    entry: dict = _SST_ENTRY_SUBSET,
) -> Netcdf2Zarr:
    """Create a Netcdf2Zarr with ZarrCatalog mocked."""
    download_dir = tmp_path / f"dl_{var_key}"
    store_dir = tmp_path / f"store_{var_key}"
    download_dir.mkdir(exist_ok=True)
    store_dir.mkdir(exist_ok=True)

    with patch("h2mare.format_converters.netcdf2zarr.ZarrCatalog") as MockCat:
        MockCat.return_value.store_root = store_dir
        return Netcdf2Zarr(
            var_key,
            app_config=_make_config(var_key, entry),
            store_root=store_dir,
            download_root=download_dir,
        )


@pytest.fixture
def converter(tmp_path):
    return _make_converter(tmp_path)


@pytest.fixture
def single_converter(tmp_path):
    """Converter with subset=False (single-date filename pattern)."""
    return _make_converter(tmp_path, entry=_SST_ENTRY_SINGLE)


# ---------------------------------------------------------------------------
# _resolve_string
# ---------------------------------------------------------------------------


class TestResolveString:
    def test_integer_year_returns_string(self, converter):
        assert converter._resolve_string(2021) == "2021"

    def test_tuple_returns_year_backslash_month(self, converter):
        assert converter._resolve_string((2021, 3)) == r"2021\3"

    def test_invalid_input_raises(self, converter):
        with pytest.raises((ValueError, TypeError)):
            converter._resolve_string("not_valid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _parse_file_dates
# ---------------------------------------------------------------------------


class TestParseFileDates:
    def test_subset_true_expands_date_range(self, converter):
        f = Path("sst_20210101_20210131.nc")
        dates = converter._parse_file_dates(f)
        assert len(dates) == 31
        assert dates[0] == pd.Timestamp("2021-01-01")
        assert dates[-1] == pd.Timestamp("2021-01-31")

    def test_subset_false_returns_single_date(self, single_converter):
        f = Path("sst_20210115.nc")
        dates = single_converter._parse_file_dates(f)
        assert len(dates) == 1
        assert dates[0] == pd.Timestamp("2021-01-15")

    def test_no_match_returns_empty_list(self, converter):
        f = Path("README.txt")
        assert converter._parse_file_dates(f) == []


# ---------------------------------------------------------------------------
# _get_file_date_bounds
# ---------------------------------------------------------------------------


class TestGetFileDateBounds:
    def test_subset_true_returns_start_and_end(self, converter):
        f = Path("sst_20210601_20210630.nc")
        bounds = converter._get_file_date_bounds(f)
        assert bounds is not None
        start, end = bounds
        assert start == pd.Timestamp("2021-06-01")
        assert end == pd.Timestamp("2021-06-30")

    def test_no_match_returns_none(self, converter):
        assert converter._get_file_date_bounds(Path("unmatched.nc")) is None


# ---------------------------------------------------------------------------
# _get_downloaded_files
# ---------------------------------------------------------------------------


class TestGetDownloadedFiles:
    def test_finds_nc_files(self, tmp_path):
        n2z = _make_converter(tmp_path)
        (n2z.download_root / "file_20210101_20210131.nc").touch()
        files = n2z._get_downloaded_files()
        assert len(files) == 1

    def test_finds_grib_files(self, tmp_path):
        n2z = _make_converter(tmp_path)
        (n2z.download_root / "era5_202101.grib").touch()
        files = n2z._get_downloaded_files()
        assert len(files) == 1

    def test_raises_when_no_files(self, tmp_path):
        n2z = _make_converter(tmp_path)
        with pytest.raises(FileNotFoundError):
            n2z._get_downloaded_files()


# ---------------------------------------------------------------------------
# _read_manifest
# ---------------------------------------------------------------------------


class TestReadManifest:
    def test_returns_records_when_manifest_exists(self, tmp_path):
        n2z = _make_converter(tmp_path)
        records = [
            {
                "dataset_id": "cmems-rep",
                "dataset_type": "rep",
                "start": "2021-01-01",
                "end": "2021-12-31",
            }
        ]
        (n2z.download_root / "h2mare_manifest.json").write_text(json.dumps(records))
        result = n2z._read_manifest()
        assert len(result) == 1
        assert result[0]["dataset_type"] == "rep"

    def test_returns_empty_list_when_missing(self, tmp_path):
        n2z = _make_converter(tmp_path)
        assert n2z._read_manifest() == []


# ---------------------------------------------------------------------------
# _group_map
# ---------------------------------------------------------------------------


class TestGroupMap:
    def test_year_grouping(self, tmp_path):
        n2z = _make_converter(tmp_path)
        (n2z.download_root / "sst_20210101_20210131.nc").touch()
        (n2z.download_root / "sst_20210201_20210228.nc").touch()
        result = n2z._group_map(TimeResolution.YEAR)
        assert 2021 in result
        assert len(result[2021]) == 2

    def test_month_grouping(self, tmp_path):
        n2z = _make_converter(tmp_path)
        (n2z.download_root / "sst_20210101_20210131.nc").touch()
        (n2z.download_root / "sst_20210201_20210228.nc").touch()
        result = n2z._group_map(TimeResolution.MONTH)
        assert (2021, 1) in result
        assert (2021, 2) in result
        # Each month key has only its own file
        jan_names = [p.name for p in result[(2021, 1)]]
        assert all("20210101" in n for n in jan_names)

    def test_empty_downloads_returns_empty_dict(self, tmp_path):
        n2z = _make_converter(tmp_path)
        # No files; _get_file_date_series returns empty Series
        with patch.object(
            n2z, "_get_file_date_series", return_value=pd.Series(dtype="object")
        ):
            assert n2z._group_map(TimeResolution.YEAR) == {}


# ---------------------------------------------------------------------------
# _stage_eddies_to_store
# ---------------------------------------------------------------------------


class TestStageEddiesToStore:
    def test_rep_files_moved_to_store_rep_subdir(self, tmp_path):
        n2z = _make_converter(tmp_path)
        n2z.store_root = tmp_path / "store"
        n2z.store_root.mkdir(exist_ok=True)

        download_root = tmp_path / "dl_eddies"
        rep_src = download_root / "rep"
        rep_src.mkdir(parents=True)
        (rep_src / "anticyclonic_20210101_20211231.nc").touch()

        n2z._stage_eddies_to_store(download_root)

        assert (n2z.store_root / "rep" / "anticyclonic_20210101_20211231.nc").exists()

    def test_nrt_files_replace_existing_nrt_in_store(self, tmp_path):
        n2z = _make_converter(tmp_path)
        n2z.store_root = tmp_path / "store"
        n2z.store_root.mkdir(exist_ok=True)

        # Pre-populate old NRT file in store
        nrt_dst = n2z.store_root / "nrt"
        nrt_dst.mkdir()
        old_file = nrt_dst / "old_nrt.nc"
        old_file.touch()

        download_root = tmp_path / "dl_nrt"
        nrt_src = download_root / "nrt"
        nrt_src.mkdir(parents=True)
        (nrt_src / "new_nrt.nc").touch()

        n2z._stage_eddies_to_store(download_root)

        assert not old_file.exists()
        assert (nrt_dst / "new_nrt.nc").exists()

    def test_flat_download_layout_falls_back_to_store_root(self, tmp_path):
        n2z = _make_converter(tmp_path)
        n2z.store_root = tmp_path / "store"
        n2z.store_root.mkdir(exist_ok=True)

        download_root = tmp_path / "dl_flat"
        download_root.mkdir()
        (download_root / "anticyclonic.nc").touch()

        n2z._stage_eddies_to_store(download_root)

        assert (n2z.store_root / "anticyclonic.nc").exists()


# ---------------------------------------------------------------------------
# process_dataset
# ---------------------------------------------------------------------------


class TestProcessDataset:
    def test_calls_registered_processor_for_var_key(self, tmp_path):
        n2z = _make_converter(tmp_path)
        ds = xr.Dataset(
            {"analysed_sst": (["time", "lat", "lon"], np.ones((2, 2, 2)))},
            coords={
                "time": pd.date_range("2020-01-01", periods=2, freq="D"),
                "lat": [30.0, 35.0],
                "lon": [-10.0, -5.0],
            },
        )
        mock_proc = MagicMock(return_value=ds)
        with patch.dict(
            "h2mare.format_converters.netcdf2zarr.PROCESSORS", {"sst": mock_proc}
        ):
            n2z.process_dataset(ds)
        mock_proc.assert_called_once()

    def test_returns_chunked_dataset_when_no_processor(self, tmp_path):
        n2z = _make_converter(tmp_path)
        ds = xr.Dataset(
            {"sst": (["time", "lat", "lon"], np.ones((2, 2, 2)))},
            coords={
                "time": pd.date_range("2020-01-01", periods=2, freq="D"),
                "lat": [30.0, 35.0],
                "lon": [-10.0, -5.0],
            },
        )
        with patch.dict(
            "h2mare.format_converters.netcdf2zarr.PROCESSORS", {}, clear=True
        ):
            result = n2z.process_dataset(ds)
        assert "sst" in result
