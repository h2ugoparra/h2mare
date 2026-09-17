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
from h2mare.types import DateRange, FilePeriod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SST_ENTRY_SUBSET = {
    "local_folder": "sst",
    "source_vars": ["analysed_sst"],
    "dataset_id_rep": "cmems-rep-sst",
    "source": "cmems",
    "archive_raw": False,
    "pattern": r"(\d{8})_(\d{8})",
    "subset": True,
    "filename_date_range": True,
    "bbox": (-80, 0, 10, 70),
}

_SST_ENTRY_SINGLE = {
    **_SST_ENTRY_SUBSET,
    "pattern": r"(\d{4})(\d{2})(\d{2})",
    "subset": False,
    "filename_date_range": False,
}

_MLD_ENTRY = {
    "local_folder": "mld",
    "source_vars": ["mlotst"],
    "dataset_id_rep": "cmems-mld",
    "source": "cmems",
    "archive_raw": False,
    "pattern": r"(\d{8})_(\d{8})",
    "subset": True,
    "filename_date_range": True,
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

    def test_tuple_returns_year_slash_month(self, converter):
        assert converter._resolve_string((2021, 3)) == "2021/3"

    def test_tuple_resolves_to_nested_dirs_on_every_platform(self, converter, tmp_path):
        # Regression: a literal backslash separator made this one directory
        # named "2021\3" on POSIX instead of 2021/3, so archived raw files for a
        # month-partitioned store landed in the wrong place off Windows.
        resolved = tmp_path / converter._resolve_string((2021, 3))
        assert resolved.parent.name == "2021"
        assert resolved.name == "3"
        assert resolved.relative_to(tmp_path).parts == ("2021", "3")

    def test_invalid_input_raises(self, converter):
        with pytest.raises((ValueError, TypeError)):
            converter._resolve_string("not_valid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _parse_file_dates
# ---------------------------------------------------------------------------


class TestParseFileDates:
    def test_filename_date_range_true_expands_date_range(self, converter):
        f = Path("sst_20210101_20210131.nc")
        dates = converter._parse_file_dates(f)
        assert len(dates) == 31
        assert dates[0] == pd.Timestamp("2021-01-01")
        assert dates[-1] == pd.Timestamp("2021-01-31")

    def test_filename_date_range_false_returns_single_date(self, single_converter):
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
        result = n2z._group_map(FilePeriod.YEAR)
        assert 2021 in result
        assert len(result[2021]) == 2

    def test_month_grouping(self, tmp_path):
        n2z = _make_converter(tmp_path)
        (n2z.download_root / "sst_20210101_20210131.nc").touch()
        (n2z.download_root / "sst_20210201_20210228.nc").touch()
        result = n2z._group_map(FilePeriod.MONTH)
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
            assert n2z._group_map(FilePeriod.YEAR) == {}


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


class TestCleanupPeriodFiles:
    def test_cmems_period_files_deleted_after_convert(self, tmp_path):
        """A non-archiving source's raw files are removed once the period is done,
        so a later period's failure doesn't force reprocessing this one."""
        n2z = _make_converter(tmp_path)  # source=cmems, download != store
        f = n2z.download_root / "sst_20210101_20210131.nc"
        f.touch()
        n2z._cleanup_period_files([f])
        assert not f.exists()


class TestArchiveRawOption:
    """archive_raw drives the archive/delete decision."""

    def test_archive_raw_true_keeps_files(self, tmp_path):
        """archive_raw=True keeps raw files from per-period deletion."""
        entry = {**_SST_ENTRY_SUBSET, "archive_raw": True}
        n2z = _make_converter(tmp_path, entry=entry)
        f = n2z.download_root / "sst_20210101_20210131.nc"
        f.touch()
        n2z._cleanup_period_files([f])
        assert f.exists()

    def test_archive_raw_true_moves_cmems_to_store(self, tmp_path):
        """archive_raw=True archives a cmems source's files into the store."""
        entry = {**_SST_ENTRY_SUBSET, "archive_raw": True}
        n2z = _make_converter(tmp_path, entry=entry)
        f = n2z.download_root / "sst_20210101_20210131.nc"
        f.touch()
        n2z._archive_raw_files(2021, [f])
        assert (n2z.store_root / "2021" / f.name).exists()

    def test_archive_raw_false_deletes_cds_files(self, tmp_path):
        """archive_raw=False deletes a cds source's files instead of archiving."""
        entry = {**_SST_ENTRY_SUBSET, "source": "cds", "archive_raw": False}
        n2z = _make_converter(tmp_path, entry=entry)
        f = n2z.download_root / "sst_20210101_20210131.nc"
        f.touch()
        n2z._cleanup_period_files([f])
        assert not f.exists()

    def test_archive_raw_false_skips_cds_archive(self, tmp_path):
        """archive_raw=False makes _archive_raw_files a no-op for a cds source."""
        entry = {**_SST_ENTRY_SUBSET, "source": "cds", "archive_raw": False}
        n2z = _make_converter(tmp_path, entry=entry)
        f = n2z.download_root / "sst_20210101_20210131.nc"
        f.touch()
        n2z._archive_raw_files(2021, [f])
        assert not (n2z.store_root / "2021" / f.name).exists()
        assert f.exists()


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

    def test_derived_vars_read_the_processor_output(self, tmp_path):
        """derived_vars name variables as the processor leaves them (sst, not
        analysed_sst), and a declared entry reaches the store."""
        entry = {
            **_SST_ENTRY_SUBSET,
            "derived_vars": {"sst_std": {"op": "rolling_std", "source": "sst"}},
        }
        n2z = _make_converter(tmp_path, entry=entry)
        ds = xr.Dataset(
            {"analysed_sst": (["time", "lat", "lon"], np.ones((2, 2, 2)))},
            coords={
                "time": pd.date_range("2020-01-01", periods=2, freq="D"),
                "lat": [30.0, 35.0],
                "lon": [-10.0, -5.0],
            },
        )
        with patch.dict(
            "h2mare.format_converters.netcdf2zarr.PROCESSORS",
            {"sst": lambda d, *_: d.rename_vars({"analysed_sst": "sst"})},
        ):
            result = n2z.process_dataset(ds)
        np.testing.assert_array_equal(result["sst_std"].values, 0.0)

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


# ---------------------------------------------------------------------------
# run() date window
#
# Re-converting one period from raw files already on disk needs a window:
# without it the only way to redo a period was `h2mare run`, which downloads.
# ---------------------------------------------------------------------------


class TestRunWindow:
    def test_window_restricts_the_periods_converted(self, tmp_path):
        """Only files inside the window are grouped for conversion."""
        converter = _make_converter(tmp_path)
        series = pd.Series(
            ["a.nc", "b.nc", "c.nc"],
            index=pd.DatetimeIndex(["2020-06-01", "2021-06-01", "2022-06-01"]),
        )

        with patch.object(converter, "_get_file_date_series", return_value=series):
            groups = converter._group_map(
                FilePeriod.YEAR,
                window=DateRange(
                    pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")
                ),
            )

        assert list(groups) == [2021]

    def test_no_window_converts_everything(self, tmp_path):
        converter = _make_converter(tmp_path)
        series = pd.Series(
            ["a.nc", "b.nc"],
            index=pd.DatetimeIndex(["2020-06-01", "2021-06-01"]),
        )

        with patch.object(converter, "_get_file_date_series", return_value=series):
            groups = converter._group_map(FilePeriod.YEAR)

        assert sorted(groups) == [2020, 2021]

    def test_window_matching_nothing_returns_no_groups(self, tmp_path):
        converter = _make_converter(tmp_path)
        series = pd.Series(["a.nc"], index=pd.DatetimeIndex(["2020-06-01"]))

        with patch.object(converter, "_get_file_date_series", return_value=series):
            groups = converter._group_map(
                FilePeriod.YEAR,
                window=DateRange(
                    pd.Timestamp("2030-01-01"), pd.Timestamp("2030-12-31")
                ),
            )

        assert groups == {}

    # No groups now means one of two things, and run() tells them apart by
    # asking whether any file parsed a date at all. These two only care about
    # the window argument, so they stub a parseable series to land on the
    # benign "window matched nothing" branch rather than the pattern fault.
    _PARSED = pd.Series(["a.nc"], index=pd.DatetimeIndex(["2020-06-01"]))

    def test_run_passes_the_window_through(self, tmp_path):
        converter = _make_converter(tmp_path)

        with (
            patch.object(converter, "_group_map", return_value={}) as mock_group,
            patch.object(converter, "_get_file_date_series", return_value=self._PARSED),
            patch("h2mare.format_converters.netcdf2zarr.recover_zarr_store"),
        ):
            converter.run("2021-01-01", "2021-12-31")

        window = mock_group.call_args.kwargs["window"]
        assert window.start == pd.Timestamp("2021-01-01")
        assert window.end == pd.Timestamp("2021-12-31")

    def test_run_without_dates_passes_no_window(self, tmp_path):
        converter = _make_converter(tmp_path)

        with (
            patch.object(converter, "_group_map", return_value={}) as mock_group,
            patch.object(converter, "_get_file_date_series", return_value=self._PARSED),
            patch("h2mare.format_converters.netcdf2zarr.recover_zarr_store"),
        ):
            converter.run()

        assert mock_group.call_args.kwargs["window"] is None

    def test_trajectory_vars_forward_dates_to_the_processor(self, tmp_path):
        """Regression: eddies ignored the window because run() took no dates."""
        converter = _make_converter(tmp_path)
        converter.var_config.trajectory_format = True

        with (
            patch.object(converter, "_process_eddies") as mock_eddies,
            patch("h2mare.format_converters.netcdf2zarr.recover_zarr_store"),
        ):
            converter.run("2026-04-21", "2026-07-13")

        assert mock_eddies.call_args[0] == ("2026-04-21", "2026-07-13")


# ---------------------------------------------------------------------------
# Raw staging safety
#
# _stage_eddies_to_store moves freshly downloaded raw files into the store.
# Pointing `convert --in-dir` at an archive_raw store makes download_root and
# store_root the same directory, at which point every move is a move onto
# itself — and safe_move_files unlinks the destination before moving, so the
# source is deleted outright.
# ---------------------------------------------------------------------------


class TestStagingSafety:
    def _store_with_raw(self, tmp_path):
        """A store laid out the way archive_raw leaves it."""
        store = tmp_path / "store"
        (store / "rep").mkdir(parents=True)
        (store / "nrt").mkdir(parents=True)
        (store / "rep" / "rep_a_19930101_20220209.nc").write_text("rep")
        (store / "nrt" / "nrt_a_20180101_20260713.nc").write_text("nrt")
        return store

    def test_no_staging_when_raw_already_lives_in_the_store(self, tmp_path):
        """Regression: this deleted the raw files instead of moving them."""
        store = self._store_with_raw(tmp_path)
        converter = _make_converter(tmp_path)
        converter.store_root = store

        converter._stage_eddies_to_store(store)

        assert (store / "rep" / "rep_a_19930101_20220209.nc").exists()
        assert (store / "nrt" / "nrt_a_20180101_20260713.nc").exists()

    def test_staging_still_moves_from_a_separate_download_root(self, tmp_path):
        store = tmp_path / "store"
        (store).mkdir()
        downloads = tmp_path / "downloads"
        (downloads / "nrt").mkdir(parents=True)
        (downloads / "nrt" / "new_20180101_20260713.nc").write_text("new")

        converter = _make_converter(tmp_path)
        converter.store_root = store

        converter._stage_eddies_to_store(downloads)

        assert (store / "nrt" / "new_20180101_20260713.nc").exists()
        assert not (downloads / "nrt" / "new_20180101_20260713.nc").exists()

    def test_stale_nrt_is_replaced_but_incoming_survives(self, tmp_path):
        """The new snapshot lands before anything is deleted."""
        store = tmp_path / "store"
        (store / "nrt").mkdir(parents=True)
        (store / "nrt" / "old_20180101_20250101.nc").write_text("old")
        downloads = tmp_path / "downloads"
        (downloads / "nrt").mkdir(parents=True)
        (downloads / "nrt" / "new_20180101_20260713.nc").write_text("new")

        converter = _make_converter(tmp_path)
        converter.store_root = store

        converter._stage_eddies_to_store(downloads)

        assert (store / "nrt" / "new_20180101_20260713.nc").exists()
        assert not (store / "nrt" / "old_20180101_20250101.nc").exists()

    def test_failed_move_leaves_the_existing_snapshot_intact(self, tmp_path):
        """A move that dies part way must not have already cleared the store.

        Clearing the destination first meant a failure left neither the old
        snapshot nor the new one.
        """
        store = tmp_path / "store"
        (store / "nrt").mkdir(parents=True)
        (store / "nrt" / "old_20180101_20250101.nc").write_text("old")
        downloads = tmp_path / "downloads"
        (downloads / "nrt").mkdir(parents=True)
        (downloads / "nrt" / "new_20180101_20260713.nc").write_text("new")

        converter = _make_converter(tmp_path)
        converter.store_root = store

        with patch(
            "h2mare.format_converters.netcdf2zarr.safe_move_files",
            side_effect=OSError("disk full"),
        ):
            with pytest.raises(OSError):
                converter._stage_eddies_to_store(downloads)

        assert (store / "nrt" / "old_20180101_20250101.nc").exists()

    def test_staging_removes_the_spent_manifest(self, tmp_path):
        """Regression: the manifest outlived the raw files and kept the folder alive."""
        store = tmp_path / "store"
        store.mkdir()
        downloads = tmp_path / "downloads"
        (downloads / "nrt").mkdir(parents=True)
        (downloads / "nrt" / "new_20180101_20260713.nc").write_text("new")
        (downloads / "h2mare_manifest.json").write_text("[]")

        converter = _make_converter(tmp_path)
        converter.store_root = store

        converter._stage_eddies_to_store(downloads)

        assert not (downloads / "h2mare_manifest.json").exists()
        assert not any(downloads.rglob("*.nc"))

    def test_manifest_kept_when_raw_already_lives_in_the_store(self, tmp_path):
        """In-place layout: the raw files *are* the archive, manifest included."""
        store = self._store_with_raw(tmp_path)
        (store / "h2mare_manifest.json").write_text("[]")

        converter = _make_converter(tmp_path)
        converter.store_root = store

        converter._stage_eddies_to_store(store)

        assert (store / "h2mare_manifest.json").exists()

    def test_windowed_conversion_does_not_stage(self, tmp_path):
        """Staging is download cleanup; a re-conversion must not move raw files."""
        converter = _make_converter(tmp_path)
        converter.var_config.trajectory_format = True

        with (
            patch("h2mare.processing.core.aviso.EDDIESProcessor"),
            patch.object(converter, "_stage_eddies_to_store") as mock_stage,
        ):
            converter._process_eddies("2026-04-21", "2026-07-13")

        mock_stage.assert_not_called()

    def test_full_conversion_still_stages(self, tmp_path):
        converter = _make_converter(tmp_path)
        converter.var_config.trajectory_format = True

        with (
            patch("h2mare.processing.core.aviso.EDDIESProcessor"),
            patch.object(converter, "_stage_eddies_to_store") as mock_stage,
        ):
            converter._process_eddies()

        mock_stage.assert_called_once()


# ---------------------------------------------------------------------------
# Write verification
#
# A day that never downloaded is absent from the raw files, so anything that
# derives its expectations from disk moves to match the loss and reports
# success. Store coverage is a min/max watermark an interior hole cannot move,
# so the short year then goes on reporting itself full — AVISO_FSLE 1999 (128
# of 365 days) and 2025-06-02 both survived every existing check that way.
# ---------------------------------------------------------------------------

# archive_raw=False keeps these tests off the archive branch, which moves files
# and is not what they are about. TestArchiveRawReleasesFileHandles covers that
# path directly.
_DAILY_ENTRY = {
    "local_folder": "testvar",
    "source_vars": ["testvar"],
    "dataset_id_rep": "test-rep",
    "source": "cmems",
    "archive_raw": False,
    "pattern": r"(\d{8})",
    "subset": False,
    "filename_date_range": False,
}


def _daily_ds(dates, null_days=()) -> xr.Dataset:
    times = pd.DatetimeIndex(dates)
    nulls = pd.DatetimeIndex(null_days)
    rng = np.random.default_rng(0)
    data = rng.uniform(10.0, 30.0, size=(len(times), 3, 3))
    for i, t in enumerate(times):
        if t in nulls:
            data[i, :, :] = np.nan
    return xr.Dataset(
        {"testvar": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


def _period_converter(tmp_path, **overrides) -> Netcdf2Zarr:
    conv = _make_converter(
        tmp_path, var_key="testvar", entry={**_DAILY_ENTRY, **overrides}
    )
    conv.catalog.build_file_path.return_value = conv.store_root / "testvar_2020.zarr"
    return conv


def _write_raw_days(conv: Netcdf2Zarr, dates, null_days=()) -> list[Path]:
    """One raw file per day, named so the configured pattern matches."""
    paths = []
    for d in pd.DatetimeIndex(dates):
        p = conv.download_root / f"testvar_{d.strftime('%Y%m%d')}.nc"
        _daily_ds([d], null_days=null_days).to_netcdf(p)
        paths.append(p)
    return paths


_JAN = pd.date_range("2020-01-01", "2020-01-10", freq="D")


class TestVerifyWrittenDates:
    def test_missing_interior_day_raises(self, tmp_path):
        """The file for Jan 5 never arrived — exactly the FSLE 2025-06-02 shape."""
        conv = _period_converter(tmp_path)
        paths = _write_raw_days(conv, _JAN.drop(pd.Timestamp("2020-01-05")))

        with pytest.raises(RuntimeError) as excinfo:
            conv._process_period(2020, paths)

        assert "2020-01-05" in str(excinfo.value.__cause__)

    def test_raw_files_survive_the_failure(self, tmp_path):
        """Archive/cleanup run after the check, so the period stays re-convertible."""
        conv = _period_converter(tmp_path)
        paths = _write_raw_days(conv, _JAN.drop(pd.Timestamp("2020-01-05")))

        with pytest.raises(RuntimeError):
            conv._process_period(2020, paths)

        assert all(p.exists() for p in paths)

    def test_complete_period_passes(self, tmp_path):
        conv = _period_converter(tmp_path)
        paths = _write_raw_days(conv, _JAN)

        conv._process_period(2020, paths)  # must not raise

        stored = xr.open_zarr(conv.catalog.build_file_path.return_value)
        assert len(stored.time) == 10
        stored.close()

    def test_all_null_day_passes(self, tmp_path):
        """A day present but entirely null is a source gap, not a defect.

        chl has three in 1999 alone. A check that fires on those every year is
        one people learn to ignore, at which point it protects nothing.
        """
        conv = _period_converter(tmp_path)
        paths = _write_raw_days(conv, _JAN, null_days=[pd.Timestamp("2020-01-05")])

        conv._process_period(2020, paths)  # must not raise

    def test_short_tail_warns_but_does_not_raise(self, tmp_path):
        """Provider lag at the tail is ordinary and must not fail the run."""
        conv = _period_converter(tmp_path)
        paths = _write_raw_days(conv, _JAN)
        (conv.download_root / "h2mare_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "dataset_id": "test-rep",
                        "dataset_type": "rep",
                        "start": "2020-01-01",
                        "end": "2020-01-20",
                    }
                ]
            )
        )

        with patch("h2mare.format_converters.netcdf2zarr.logger") as mock_logger:
            conv._process_period(2020, paths)

        warned = " ".join(str(c) for c in mock_logger.warning.call_args_list)
        assert "2020-01-11" in warned

    def test_single_day_period_is_not_flagged(self, tmp_path):
        """One day has no interior, so there is nothing to be missing from it."""
        conv = _period_converter(tmp_path)
        paths = _write_raw_days(conv, [pd.Timestamp("2020-01-01")])

        conv._process_period(2020, paths)  # must not raise


class TestExpectedDates:
    def test_manifest_window_wins_over_files_on_disk(self, tmp_path):
        """The manifest states what was asked for; disk only shows what arrived."""
        conv = _period_converter(tmp_path)
        paths = _write_raw_days(conv, _JAN.drop(pd.Timestamp("2020-01-05")))
        (conv.download_root / "h2mare_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "dataset_id": "test-rep",
                        "dataset_type": "rep",
                        "start": "2020-01-01",
                        "end": "2020-01-10",
                    }
                ]
            )
        )

        expected = conv._expected_dates(2020, paths)

        assert pd.Timestamp("2020-01-05") in expected

    def test_falls_back_to_filenames_without_a_manifest(self, tmp_path):
        conv = _period_converter(tmp_path)
        paths = _write_raw_days(conv, _JAN)

        expected = conv._expected_dates(2020, paths)

        assert len(expected) == 10

    def test_manifest_is_clipped_to_the_period(self, tmp_path):
        conv = _period_converter(tmp_path)
        (conv.download_root / "h2mare_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "dataset_id": "test-rep",
                        "dataset_type": "rep",
                        "start": "2019-11-01",
                        "end": "2020-03-31",
                    }
                ]
            )
        )

        expected = conv._expected_dates(2020, [])

        assert expected[0] == pd.Timestamp("2020-01-01")
        assert expected[-1] == pd.Timestamp("2020-03-31")


class TestPeriodBounds:
    def test_year_period_spans_the_calendar_year(self, converter):
        bounds = converter._period_bounds(2021)
        assert bounds.start == pd.Timestamp("2021-01-01")
        assert bounds.end == pd.Timestamp("2021-12-31")

    def test_month_period_spans_the_month(self, converter):
        bounds = converter._period_bounds((2021, 2))
        assert bounds.start == pd.Timestamp("2021-02-01")
        assert bounds.end == pd.Timestamp("2021-02-28")

    def test_month_period_handles_leap_february(self, converter):
        assert converter._period_bounds((2020, 2)).end == pd.Timestamp("2020-02-29")


# ---------------------------------------------------------------------------
# Provenance on the generic converter path
#
# It recorded the raw *filename* spans and overwrote the attribute on every
# append, so a period built over several runs kept only the last run's claim.
# It also said nothing about what arrived — chl's 1999 record reads
# 1999-01-01 → 1999-12-31 because that is what was requested.
# ---------------------------------------------------------------------------


class TestProvenanceCoveredDates:
    def _manifest(self, conv, start="2020-01-01", end="2020-01-10"):
        (conv.download_root / "h2mare_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "dataset_id": "test-rep",
                        "dataset_type": "rep",
                        "start": start,
                        "end": end,
                    }
                ]
            )
        )

    def _read_provenance(self, conv):
        import zarr

        root = zarr.open_group(str(conv.catalog.build_file_path.return_value), mode="r")
        return json.loads(root.attrs["source_datasets"])

    def test_covered_days_are_recorded(self, tmp_path):
        conv = _period_converter(tmp_path)
        self._manifest(conv)
        conv._process_period(2020, _write_raw_days(conv, _JAN))

        [record] = self._read_provenance(conv)

        assert record["days"] == 10

    def test_covered_bounds_are_recorded(self, tmp_path):
        conv = _period_converter(tmp_path)
        self._manifest(conv)
        conv._process_period(2020, _write_raw_days(conv, _JAN))

        [record] = self._read_provenance(conv)

        assert record["start_date"] == "2020-01-01"
        assert record["end_date"] == "2020-01-10"

    def test_the_write_is_stamped(self, tmp_path):
        conv = _period_converter(tmp_path)
        self._manifest(conv)
        conv._process_period(2020, _write_raw_days(conv, _JAN))

        [record] = self._read_provenance(conv)

        assert record["updated"] == pd.Timestamp.today().strftime("%Y-%m-%d")

    def test_append_widens_rather_than_overwrites(self, tmp_path):
        """The old path replaced the attribute, dropping the earlier run."""
        conv = _period_converter(tmp_path)
        self._manifest(conv, "2020-01-01", "2020-01-05")
        conv._process_period(2020, _write_raw_days(conv, _JAN[:5]))

        self._manifest(conv, "2020-01-06", "2020-01-10")
        conv._process_period(2020, _write_raw_days(conv, _JAN[5:]))

        [record] = self._read_provenance(conv)

        assert record["start_date"] == "2020-01-01"
        assert record["end_date"] == "2020-01-10"

    def test_covered_count_reflects_the_whole_file_after_an_append(self, tmp_path):
        """Recomputed from the store, so appends do not double-count."""
        conv = _period_converter(tmp_path)
        self._manifest(conv, "2020-01-01", "2020-01-05")
        conv._process_period(2020, _write_raw_days(conv, _JAN[:5]))

        self._manifest(conv, "2020-01-06", "2020-01-10")
        conv._process_period(2020, _write_raw_days(conv, _JAN[5:]))

        [record] = self._read_provenance(conv)

        assert record["days"] == 10


# ---------------------------------------------------------------------------
# Single-day filenames and the silent no-op
#
# copernicusmarine names a one-day subset with a single date
# (`..._2026-07-31.nc`), not a range. A two-date pattern matched nothing, so
# _group_map produced no periods, run() reported "0 period(s)" success, and
# _cleanup_downloads then deleted the file that had downloaded fine. The store
# never changed and the run exited 0 — which is how the repair path for a
# one-day gap was itself broken.
# ---------------------------------------------------------------------------

_OPTIONAL_RANGE = r"(\d{4}-\d{2}-\d{2})(?:-(\d{4}-\d{2}-\d{2}))?"

_RANGE_ENTRY = {
    "local_folder": "testvar",
    "source_vars": ["testvar"],
    "dataset_id_rep": "test-rep",
    "source": "cmems",
    "archive_raw": False,
    "pattern": _OPTIONAL_RANGE,
    "subset": True,
    "filename_date_range": True,
}


class TestSingleDayFilename:
    def _conv(self, tmp_path, **overrides):
        return _make_converter(
            tmp_path, var_key="testvar", entry={**_RANGE_ENTRY, **overrides}
        )

    def test_one_day_subset_filename_parses_as_that_day(self, tmp_path):
        conv = self._conv(tmp_path)
        f = Path("METOFFICE-GLO-SST_multi-vars_79.97W-9.98E_2026-07-31.nc")

        assert conv._parse_file_dates(f) == [pd.Timestamp("2026-07-31")]

    def test_range_filename_still_expands(self, tmp_path):
        conv = self._conv(tmp_path)
        f = Path("METOFFICE_multi-vars_2026-07-30-2026-07-31.nc")

        assert conv._parse_file_dates(f) == [
            pd.Timestamp("2026-07-30"),
            pd.Timestamp("2026-07-31"),
        ]

    def test_one_day_bounds_are_the_same_day(self, tmp_path):
        conv = self._conv(tmp_path)
        f = Path("METOFFICE_multi-vars_2026-07-31.nc")

        assert conv._get_file_date_bounds(f) == (
            pd.Timestamp("2026-07-31"),
            pd.Timestamp("2026-07-31"),
        )

    def test_range_bounds_use_both_dates(self, tmp_path):
        conv = self._conv(tmp_path)
        f = Path("METOFFICE_multi-vars_2026-07-30-2026-07-31.nc")

        assert conv._get_file_date_bounds(f) == (
            pd.Timestamp("2026-07-30"),
            pd.Timestamp("2026-07-31"),
        )

    def test_a_one_day_file_produces_a_period(self, tmp_path):
        """The whole point: a one-day repair download must be convertible."""
        conv = self._conv(tmp_path)
        (conv.download_root / "METOFFICE_2026-07-31.nc").write_bytes(b"")

        assert list(conv._group_map(groupby=FilePeriod.YEAR)) == [2026]


class TestUnparseableFilenamesAreNotSilent:
    def _conv(self, tmp_path, pattern):
        return _make_converter(
            tmp_path,
            var_key="testvar",
            entry={**_RANGE_ENTRY, "pattern": pattern},
        )

    def test_run_raises_when_no_file_yields_a_date(self, tmp_path):
        conv = self._conv(tmp_path, r"(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})")
        (conv.download_root / "METOFFICE_2026-07-31.nc").write_bytes(b"")

        with pytest.raises(RuntimeError, match="none .*yielded a date"):
            conv.run()

    def test_the_error_names_the_pattern(self, tmp_path):
        conv = self._conv(tmp_path, r"(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})")
        (conv.download_root / "METOFFICE_2026-07-31.nc").write_bytes(b"")

        with pytest.raises(RuntimeError) as excinfo:
            conv.run()

        assert "pattern=" in str(excinfo.value)

    def test_raw_files_survive_the_failure(self, tmp_path):
        """They used to be deleted by _cleanup_downloads on the success path."""
        conv = self._conv(tmp_path, r"(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})")
        raw = conv.download_root / "METOFFICE_2026-07-31.nc"
        raw.write_bytes(b"")

        with pytest.raises(RuntimeError):
            conv.run()

        assert raw.exists()

    def test_a_window_matching_nothing_warns_instead_of_raising(self, tmp_path):
        """Files parse fine; the requested window just has none of them."""
        conv = self._conv(tmp_path, _OPTIONAL_RANGE)
        (conv.download_root / "METOFFICE_2026-07-31.nc").write_bytes(b"")

        assert conv.run("2020-01-01", "2020-01-31") is False


class TestKnownGapsAtConversion:
    """A day the provider never published must not fail the whole period."""

    def test_a_known_gap_does_not_fail_conversion(self, tmp_path):
        conv = _period_converter(tmp_path, known_gaps=["2020-01-05"])
        paths = _write_raw_days(conv, _JAN.drop(pd.Timestamp("2020-01-05")))

        conv._process_period(2020, paths)  # must not raise

    def test_an_unlisted_gap_still_fails(self, tmp_path):
        conv = _period_converter(tmp_path, known_gaps=["2020-01-08"])
        paths = _write_raw_days(conv, _JAN.drop(pd.Timestamp("2020-01-05")))

        with pytest.raises(RuntimeError) as excinfo:
            conv._process_period(2020, paths)

        assert "2020-01-05" in str(excinfo.value.__cause__)

    def test_an_interval_entry_covers_a_block(self, tmp_path):
        conv = _period_converter(tmp_path, known_gaps=["2020-01-04/2020-01-06"])
        paths = _write_raw_days(
            conv, _JAN.drop(pd.date_range("2020-01-04", "2020-01-06"))
        )

        conv._process_period(2020, paths)  # must not raise


# ---------------------------------------------------------------------------
# archive_raw and the Windows file lock
#
# process_dataset rebinds its argument, so the object open_mfdataset returned
# had no surviving reference and was never closed — and closing the *derived*
# dataset does not release the underlying *.nc handles. On Windows those
# handles blocked safe_move_files, so an archive_raw: true variable could not
# complete a conversion at all: PermissionError [WinError 32].
# ---------------------------------------------------------------------------


class TestArchiveRawReleasesFileHandles:
    def _conv(self, tmp_path):
        conv = _period_converter(tmp_path, archive_raw=True)
        # _archive_raw_files only moves when the two roots differ.
        assert conv.download_root != conv.store_root
        return conv

    def test_archiving_moves_the_raw_files(self, tmp_path):
        conv = self._conv(tmp_path)
        paths = _write_raw_days(conv, _JAN)

        conv._process_period(2020, paths)

        archived = sorted((conv.store_root / "2020").glob("*.nc"))
        assert len(archived) == len(paths)

    def test_the_originals_are_gone_from_the_download_dir(self, tmp_path):
        conv = self._conv(tmp_path)
        paths = _write_raw_days(conv, _JAN)

        conv._process_period(2020, paths)

        assert not any(p.exists() for p in paths)

    def test_the_zarr_is_still_written(self, tmp_path):
        conv = self._conv(tmp_path)
        paths = _write_raw_days(conv, _JAN)

        conv._process_period(2020, paths)

        stored = xr.open_zarr(conv.catalog.build_file_path.return_value)
        assert len(stored.time) == len(_JAN)
        stored.close()

    def test_raw_files_are_movable_after_a_failed_period(self, tmp_path):
        """The failure path leaked the handles too, locking the whole tree."""
        import shutil

        conv = self._conv(tmp_path)
        paths = _write_raw_days(conv, _JAN.drop(pd.Timestamp("2020-01-05")))

        with pytest.raises(RuntimeError):
            conv._process_period(2020, paths)

        dest = tmp_path / "moved"
        dest.mkdir()
        shutil.move(str(paths[0]), str(dest / paths[0].name))  # must not raise
        assert (dest / paths[0].name).exists()


# ---------------------------------------------------------------------------
# _cleanup_downloads — scoped to what the run actually consumed
# ---------------------------------------------------------------------------


class TestCleanupDownloadsScope:
    """
    The folder is only removed once a run has consumed it. Anything left was not
    converted — a period outside the window, or a file the pattern skipped — and
    used to be deleted anyway, so re-converting one year took the rest with it.
    """

    def test_windowed_run_leaves_other_periods_alone(self, tmp_path):
        conv = _period_converter(tmp_path)
        paths = _write_raw_days(conv, _JAN)  # Jan 1–10

        conv.run(start_date="2020-01-01", end_date="2020-01-05")

        survivors = [p for p in paths if p.exists()]
        assert len(survivors) == 5, (
            f"expected the 5 out-of-window files to survive, got {len(survivors)}"
        )
        assert conv.download_root.exists()

    def test_full_run_removes_the_spent_folder(self, tmp_path):
        """The tidy-up still happens once nothing is left to convert."""
        conv = _period_converter(tmp_path)
        _write_raw_days(conv, _JAN)

        conv.run()

        assert not conv.download_root.exists()

    def test_sidecars_do_not_keep_a_spent_folder_alive(self, tmp_path):
        """Only raw inputs count; a manifest left behind must not block cleanup."""
        conv = _period_converter(tmp_path)
        _write_raw_days(conv, _JAN)
        (conv.download_root / "h2mare_manifest.json").write_text("[]")

        conv.run()

        assert not conv.download_root.exists()
