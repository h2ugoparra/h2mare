"""Tests for utils/files_io.py — file I/O utilities."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from h2mare.utils.files_io import (
    filter_raw_files,
    mirror_tree,
    prune_empty_dirs,
    safe_move_files,
    safe_rmtree,
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

    def test_file_already_at_destination_is_left_alone(self, tmp_path):
        """Regression: a same-path move deleted the file.

        The retry loop unlinks the destination before moving, so when source
        and destination resolve to the same file the unlink destroys it and the
        move then has nothing to move. Callers hit this whenever a store is
        both the download root and the destination.
        """
        d = tmp_path / "data"
        d.mkdir()
        f = d / "raw.nc"
        f.write_text("payload")

        safe_move_files(f, d)

        assert f.exists()
        assert f.read_text() == "payload"

    def test_same_path_via_a_relative_route_is_also_left_alone(self, tmp_path):
        """The check resolves paths, so `dst/../dst` counts as the same place."""
        d = tmp_path / "data"
        d.mkdir()
        f = d / "raw.nc"
        f.write_text("payload")

        safe_move_files(f, d / ".." / "data")

        assert f.exists()

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
# prune_empty_dirs
# ---------------------------------------------------------------------------


class TestPruneEmptyDirs:
    def test_removes_nested_empty_chain(self, tmp_path):
        (tmp_path / "a" / "b" / "c").mkdir(parents=True)
        removed = prune_empty_dirs(tmp_path)
        assert removed == 3
        assert not (tmp_path / "a").exists()
        assert tmp_path.exists()  # root itself is kept

    def test_keeps_dirs_containing_files(self, tmp_path):
        rep = tmp_path / "var" / "rep"
        rep.mkdir(parents=True)
        (rep / "data.nc").touch()
        (tmp_path / "var" / "nrt").mkdir()

        removed = prune_empty_dirs(tmp_path)

        assert removed == 1  # only nrt
        assert (rep / "data.nc").exists()
        assert not (tmp_path / "var" / "nrt").exists()

    def test_missing_root_returns_zero(self, tmp_path):
        assert prune_empty_dirs(tmp_path / "nope") == 0


# ---------------------------------------------------------------------------
# raw_include
#
# A download directory can hold files the pipeline must not read. AVISO ships
# META3.2 eddy trajectories as long/short/untracked variants side by side: only
# the long ones belong in the store, and the untracked files carry no `track`
# variable at all.
# ---------------------------------------------------------------------------

_EDDY_FILES = [
    "META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc",
    "META3.2_DT_allsat_Anticyclonic_short_19930101_20220209.nc",
    "META3.2_DT_allsat_Anticyclonic_untracked_19930101_20220209.nc",
    "META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc",
    "META4_DT_allsat_cyclonic_19930101_20230908.nc",
    "Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20260713.nc",
]


class _Cfg:
    def __init__(self, raw_include=None):
        self.raw_include = raw_include


class TestFilterRawFiles:
    def _paths(self):
        return [Path(n) for n in _EDDY_FILES]

    def test_no_pattern_keeps_everything(self):
        assert len(filter_raw_files(self._paths(), _Cfg())) == len(_EDDY_FILES)

    def test_config_without_the_field_keeps_everything(self):
        """Variables predating raw_include must be unaffected."""

        class Old:
            pass

        assert len(filter_raw_files(self._paths(), Old())) == len(_EDDY_FILES)

    def test_keeps_only_long_and_nrt(self):
        kept = {p.name for p in filter_raw_files(self._paths(), _Cfg("_long_|_nrt_"))}
        assert kept == {
            "META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc",
            "META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc",
            "Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20260713.nc",
        }

    def test_untracked_is_excluded(self):
        """The file with no `track` variable must never reach the processor."""
        kept = {p.name for p in filter_raw_files(self._paths(), _Cfg("_long_|_nrt_"))}
        assert not any("untracked" in n for n in kept)

    def test_other_product_versions_are_excluded(self):
        kept = {p.name for p in filter_raw_files(self._paths(), _Cfg("_long_|_nrt_"))}
        assert not any(n.startswith("META4") for n in kept)

    def test_pattern_matching_nothing_returns_empty(self):
        assert filter_raw_files(self._paths(), _Cfg("_nosuchvariant_")) == []


# ---------------------------------------------------------------------------
# mirror_tree
# ---------------------------------------------------------------------------


class TestMirrorTree:
    def test_unchanged_files_are_not_recopied(self, tmp_path):
        src, dest = tmp_path / "src", tmp_path / "dest"
        (src / "a").mkdir(parents=True)
        (src / "a" / "same.bin").write_text("same")
        mirror_tree(src, dest)
        (src / "a" / "new.bin").write_text("new")

        with patch(
            "h2mare.utils.files_io.shutil.copy2", wraps=__import__("shutil").copy2
        ) as copy2:
            mirror_tree(src, dest)

        copied = {Path(c.args[0]).name for c in copy2.call_args_list}
        assert copied == {"new.bin"}

    def test_changed_file_is_recopied(self, tmp_path):
        src, dest = tmp_path / "src", tmp_path / "dest"
        src.mkdir()
        f = src / "x.bin"
        f.write_text("old")
        mirror_tree(src, dest)
        f.write_text("newer")  # different size
        mirror_tree(src, dest)
        assert (dest / "x.bin").read_text() == "newer"

    def test_same_size_rewrite_is_recopied(self, tmp_path):
        src, dest = tmp_path / "src", tmp_path / "dest"
        src.mkdir()
        f = src / "x.bin"
        f.write_text("aaa")
        mirror_tree(src, dest)
        f.write_text("bbb")
        st = f.stat()
        os.utime(f, (st.st_atime, st.st_mtime + 10))
        mirror_tree(src, dest)
        assert (dest / "x.bin").read_text() == "bbb"

    def test_returns_count_of_stale_paths_removed(self, tmp_path):
        src, dest = tmp_path / "src", tmp_path / "dest"
        src.mkdir()
        (src / "keep").write_text("k")
        (dest / "gone_dir").mkdir(parents=True)
        (dest / "gone_dir" / "f").write_text("x")
        (dest / "gone_file").write_text("x")
        assert mirror_tree(src, dest) == 3
        assert sorted(p.name for p in dest.iterdir()) == ["keep"]

    @pytest.mark.parametrize("dest_rel", [".", "inner", ".."])
    def test_refuses_overlapping_paths(self, tmp_path, dest_rel):
        src = tmp_path / "store"
        src.mkdir()
        (src / "f").write_text("x")
        with pytest.raises(ValueError, match="overlaps"):
            mirror_tree(src, src / dest_rel)
        assert (src / "f").exists()

    def test_refuses_an_empty_source(self, tmp_path):
        src, dest = tmp_path / "src", tmp_path / "dest"
        (src / "empty_dir").mkdir(parents=True)
        dest.mkdir()
        (dest / "kept").write_text("x")
        with pytest.raises(ValueError, match="no files"):
            mirror_tree(src, dest)
        assert (dest / "kept").exists()
