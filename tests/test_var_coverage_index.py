"""Tests for storage/var_coverage_index.py."""

import json

import pandas as pd
import pytest

from h2mare.storage.var_coverage_index import VarCoverageIndex


@pytest.fixture
def idx(tmp_path):
    return VarCoverageIndex(tmp_path / "coverage.json")


class TestVarCoverageIndex:
    def test_get_end_returns_none_when_no_entry(self, idx):
        assert idx.get_end("sst") is None

    def test_update_and_get_end(self, idx):
        idx.update("sst", pd.Timestamp("2026-05-29"))
        assert idx.get_end("sst") == pd.Timestamp("2026-05-29")

    def test_update_never_goes_backwards(self, idx):
        idx.update("sst", pd.Timestamp("2026-05-29"))
        idx.update("sst", pd.Timestamp("2026-01-01"))
        assert idx.get_end("sst") == pd.Timestamp("2026-05-29")

    def test_update_advances_when_newer(self, idx):
        idx.update("sst", pd.Timestamp("2026-05-01"))
        idx.update("sst", pd.Timestamp("2026-05-29"))
        assert idx.get_end("sst") == pd.Timestamp("2026-05-29")

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "coverage.json"
        idx = VarCoverageIndex(path)
        idx.update("sst", pd.Timestamp("2026-05-29"))
        idx.update("thetao", pd.Timestamp("2024-12-31"))
        idx.save()

        reloaded = VarCoverageIndex(path)
        assert reloaded.get_end("sst") == pd.Timestamp("2026-05-29")
        assert reloaded.get_end("thetao") == pd.Timestamp("2024-12-31")

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "coverage.json"
        idx = VarCoverageIndex(path)
        idx.update("sst", pd.Timestamp("2026-05-01"))
        idx.save()
        assert path.exists()

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "coverage.json"
        path.write_text("not json", encoding="utf-8")
        idx = VarCoverageIndex(path)
        assert idx.get_end("sst") is None

    def test_saved_format_is_iso_date(self, tmp_path):
        path = tmp_path / "coverage.json"
        idx = VarCoverageIndex(path)
        idx.update("sst", pd.Timestamp("2026-05-29"))
        idx.save()
        data = json.loads(path.read_text())
        assert data["sst"] == "2026-05-29"

    def test_multiple_vars_independent(self, idx):
        idx.update("sst", pd.Timestamp("2026-05-29"))
        idx.update("thetao", pd.Timestamp("2024-12-31"))
        assert idx.get_end("sst") == pd.Timestamp("2026-05-29")
        assert idx.get_end("thetao") == pd.Timestamp("2024-12-31")
        assert idx.get_end("ssh") is None
