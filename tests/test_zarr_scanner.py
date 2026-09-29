"""
ZarrDirectoryScanner: the filesystem layer under ZarrCatalog.

Its change detection is what tells the catalog its index is stale, so a file
added, removed or rewritten must register, and nothing else must.
"""

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from h2mare.storage.zarr_scanner import ZarrDirectoryScanner
from h2mare.types import FilePeriod

_CFG = SimpleNamespace(dataset_id_rep="demo-rep")


def _scanner(root) -> ZarrDirectoryScanner:
    return ZarrDirectoryScanner(root, FilePeriod.YEAR, _CFG)


def _write_store(path, start="2021-01-01", days=3) -> None:
    times = pd.date_range(start, periods=days, freq="D")
    xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.ones((days, 2, 2), dtype="float32"))},
        coords={"time": times, "lat": [30.0, 30.25], "lon": [-10.0, -9.75]},
    ).to_zarr(path, consolidated=False)


def _touch(path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


class TestHasChanges:
    def test_first_call_takes_a_baseline(self, tmp_path):
        (tmp_path / "a.zarr").mkdir()
        assert _scanner(tmp_path).has_changes() is False

    def test_nothing_changed_is_no_change(self, tmp_path):
        (tmp_path / "a.zarr").mkdir()
        s = _scanner(tmp_path)
        s.has_changes()
        assert s.has_changes() is False

    def test_an_added_store_is_a_change_once(self, tmp_path):
        (tmp_path / "a.zarr").mkdir()
        s = _scanner(tmp_path)
        s.has_changes()
        (tmp_path / "b.zarr").mkdir()

        assert s.has_changes() is True
        assert s.has_changes() is False  # the new state is the baseline now

    def test_a_removed_store_is_a_change(self, tmp_path):
        (tmp_path / "a.zarr").mkdir()
        (tmp_path / "b.zarr").mkdir()
        s = _scanner(tmp_path)
        s.has_changes()
        (tmp_path / "b.zarr").rmdir()
        assert s.has_changes() is True

    def test_a_rewritten_store_is_a_change(self, tmp_path):
        store = tmp_path / "a.zarr"
        store.mkdir()
        _touch(store, 1_000_000)
        s = _scanner(tmp_path)
        s.has_changes()
        _touch(store, 2_000_000)
        assert s.has_changes() is True

    def test_non_zarr_entries_are_ignored(self, tmp_path):
        (tmp_path / "a.zarr").mkdir()
        s = _scanner(tmp_path)
        s.has_changes()
        (tmp_path / "catalog.parquet").write_text("x")
        (tmp_path / "notes").mkdir()
        assert s.has_changes() is False

    def test_a_missing_directory_is_no_change(self, tmp_path):
        assert _scanner(tmp_path / "absent").has_changes() is False

    def test_reset_forgets_the_baseline(self, tmp_path):
        (tmp_path / "a.zarr").mkdir()
        s = _scanner(tmp_path)
        s.has_changes()
        (tmp_path / "b.zarr").mkdir()
        s.reset()
        assert s.has_changes() is False  # re-baselined, not compared


class TestChangeSummary:
    def test_without_a_baseline_everything_is_added(self, tmp_path):
        (tmp_path / "a.zarr").mkdir()
        (tmp_path / "b.zarr").mkdir()
        summary = _scanner(tmp_path).get_change_summary()
        assert sorted(summary["added"]) == ["a.zarr", "b.zarr"]
        assert summary["removed"] == [] and summary["modified"] == []
        assert summary["total"] == 2

    def test_names_added_removed_and_modified(self, tmp_path):
        for name in ("keep.zarr", "gone.zarr", "edit.zarr"):
            (tmp_path / name).mkdir()
            _touch(tmp_path / name, 1_000_000)
        s = _scanner(tmp_path)
        s.has_changes()
        (tmp_path / "gone.zarr").rmdir()
        (tmp_path / "new.zarr").mkdir()
        _touch(tmp_path / "edit.zarr", 2_000_000)

        summary = s.get_change_summary()

        assert summary == {
            "added": ["new.zarr"],
            "removed": ["gone.zarr"],
            "modified": ["edit.zarr"],
            "total": 3,
        }

    def test_a_missing_directory_says_so(self, tmp_path):
        assert "error" in _scanner(tmp_path / "absent").get_change_summary()


class TestScan:
    def test_reads_each_store_into_a_record(self, tmp_path):
        _write_store(tmp_path / "s_2021.zarr")

        (record,) = _scanner(tmp_path).scan()

        assert record["filename"] == "s_2021.zarr"
        assert record["variables"] == ["sst"]
        assert record["dataset"] == "demo-rep"
        assert record["num_timesteps"] == 3
        assert pd.Timestamp(record["start_date"]) == pd.Timestamp("2021-01-01")
        assert pd.Timestamp(record["end_date"]) == pd.Timestamp("2021-01-03")

    def test_an_unreadable_store_is_skipped_and_the_rest_read(self, tmp_path):
        _write_store(tmp_path / "good_2021.zarr")
        (tmp_path / "broken_2022.zarr").mkdir()  # no metadata at all

        records = _scanner(tmp_path).scan()

        assert [r["filename"] for r in records] == ["good_2021.zarr"]

    def test_a_missing_directory_scans_to_nothing(self, tmp_path):
        assert _scanner(tmp_path / "absent").scan() == []


class TestScanVariables:
    def test_unions_variables_across_stores(self, tmp_path):
        _write_store(tmp_path / "a.zarr")
        ds = xr.open_zarr(tmp_path / "a.zarr", consolidated=False)
        ds.rename_vars({"sst": "chl"}).to_zarr(tmp_path / "b.zarr", consolidated=False)
        ds.close()

        assert _scanner(tmp_path).scan_variables() == {"sst", "chl"}

    def test_an_unreadable_store_does_not_stop_the_rest(self, tmp_path):
        _write_store(tmp_path / "a.zarr")
        (tmp_path / "broken.zarr").mkdir()
        assert _scanner(tmp_path).scan_variables() == {"sst"}

    def test_empty_or_missing_directories_have_no_variables(self, tmp_path):
        assert _scanner(tmp_path).scan_variables() == set()
        assert _scanner(tmp_path / "absent").scan_variables() == set()
