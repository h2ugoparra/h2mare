"""Tests for utils/files_io.py — file I/O utilities."""
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.utils.files_io import (
    clean_era_dataset,
    move_files,
    safe_move_files,
    safe_rmtree,
    unizp_files,
)


# ---------------------------------------------------------------------------
# safe_rmtree
# ---------------------------------------------------------------------------

class TestSafeRmtree:

    def test_removes_directory_with_contents(self, tmp_path):
        d = tmp_path / "to_delete"
        d.mkdir()
        (d / "file.txt").write_text("hello")
        safe_rmtree(d)
        assert not d.exists()

    def test_nonexistent_path_is_no_op(self, tmp_path):
        safe_rmtree(tmp_path / "nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# safe_move_files
# ---------------------------------------------------------------------------

class TestSafeMoveFiles:

    def test_moves_single_file(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        f = src / "data.nc"
        f.write_text("content")
        safe_move_files(f, dst)
        assert (dst / "data.nc").exists()
        assert not f.exists()

    def test_overwrites_existing_destination_file(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "data.nc").write_text("old")
        (src / "data.nc").write_text("new")
        safe_move_files(src / "data.nc", dst)
        assert (dst / "data.nc").read_text() == "new"

    def test_moves_list_of_files(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        files = [src / f"f{i}.nc" for i in range(3)]
        for f in files:
            f.write_text("x")
        safe_move_files(files, dst)
        for f in files:
            assert (dst / f.name).exists()


# ---------------------------------------------------------------------------
# move_files
# ---------------------------------------------------------------------------

class TestMoveFiles:

    def test_moves_files_with_matching_extension(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        (src / "a.nc").write_text("x")
        (src / "b.nc").write_text("x")
        move_files(src, dst, "nc")
        assert (dst / "a.nc").exists()
        assert (dst / "b.nc").exists()

    def test_ignores_files_with_different_extension(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        (src / "keep.grib").write_text("x")
        move_files(src, dst, "nc")
        assert not (dst / "keep.grib").exists()

    def test_creates_destination_if_missing(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "new_dst"
        (src / "a.nc").write_text("x")
        move_files(src, dst, "nc")
        assert dst.exists()


# ---------------------------------------------------------------------------
# unizp_files
# ---------------------------------------------------------------------------

class TestUnizpFiles:

    def test_extracts_zip_contents(self, tmp_path):
        zip_path = tmp_path / "archive.zip"
        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file.txt", "hello")
            zf.writestr("subdir/other.txt", "world")
        unizp_files(zip_path, extract_dir)
        assert (extract_dir / "file.txt").exists()
        assert (extract_dir / "subdir" / "other.txt").exists()

    def test_creates_extract_dir_if_missing(self, tmp_path):
        zip_path = tmp_path / "archive.zip"
        extract_dir = tmp_path / "new_dir"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("f.txt", "x")
        unizp_files(zip_path, extract_dir)
        assert extract_dir.exists()


# ---------------------------------------------------------------------------
# clean_era_dataset
# ---------------------------------------------------------------------------

class TestCleanEraDataset:

    def _make_ds(self, times, values):
        return xr.Dataset(
            {"u10": ("time", values)},
            coords={"time": times},
        )

    def test_removes_duplicate_times(self):
        t = pd.date_range("2020-01-01", periods=3, freq="D")
        times = np.concatenate([t.values, [t.values[0]]])  # first day duplicated
        ds = self._make_ds(times, np.ones(4))
        result = clean_era_dataset(ds, "u10")
        assert len(result.time) == 3

    def test_drops_nat_times(self):
        t = pd.date_range("2020-01-01", periods=3, freq="D")
        times = np.concatenate([t.values, [np.datetime64("NaT")]])
        ds = self._make_ds(times, np.ones(4))
        result = clean_era_dataset(ds, "u10")
        assert len(result.time) == 3

    def test_drops_all_nan_timesteps(self):
        t = pd.date_range("2020-01-01", periods=3, freq="D")
        values = np.array([1.0, np.nan, 3.0])
        ds = self._make_ds(t, values)
        result = clean_era_dataset(ds, "u10")
        assert len(result.time) == 2

    def test_preserves_valid_times(self):
        t = pd.date_range("2020-01-01", periods=4, freq="D")
        ds = self._make_ds(t, np.arange(4.0))
        result = clean_era_dataset(ds, "u10")
        assert len(result.time) == 4
