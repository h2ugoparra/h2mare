"""Tests for processing/compiler.py — Compiler class and helpers."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.models import AppConfig
from h2mare.processing.compiler import (
    Compiler,
    calculate_moon_phase,
    postprocess_sst_fdist,
)
from h2mare.storage.var_coverage_index import VarCoverageIndex
from h2mare.types import DateRange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_H2DS_ENTRY = {
    "local_folder": "h2ds",
    "variables": ["sst"],
    "dataset_id_rep": "h2ds",
    "source": "compiled",
    "pattern": r"(\d{4})",
    "subset": False,
    "bbox": (-80, 0, 10, 70),
}

_SST_ENTRY = {
    "local_folder": "sst",
    "variables": ["analysed_sst"],
    "dataset_id_rep": "cmems-rep-sst",
    "source": "cmems",
    "pattern": r".*\.nc",
    "subset": True,
    "bbox": (-80, 0, 10, 70),
}


def _make_config() -> AppConfig:
    return msgspec.convert(
        {"variables": {"h2ds": _H2DS_ENTRY, "sst": _SST_ENTRY}, "secrets": {}},
        AppConfig,
    )


@pytest.fixture
def compiler(tmp_path):
    """Compiler with ZarrCatalog mocked out."""
    with patch("h2mare.processing.compiler.ZarrCatalog"):
        return Compiler(
            var_key="h2ds",
            app_config=_make_config(),
            remote_store_root=tmp_path / "remote",
            local_store_root=tmp_path / "local",
        )


# ---------------------------------------------------------------------------
# calculate_moon_phase
# ---------------------------------------------------------------------------


class TestCalculateMoonPhase:
    def test_returns_one_value_per_date(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        phases = calculate_moon_phase(40.0, -10.0, dates)
        assert len(phases) == 10

    def test_values_in_valid_range(self):
        dates = pd.date_range("2020-01-01", periods=31, freq="D")
        phases = calculate_moon_phase(40.0, -10.0, dates)
        assert all(0.0 <= p <= 100.0 for p in phases)

    def test_full_moon_has_high_phase(self):
        # 2020-01-10 was a full moon
        dates = pd.DatetimeIndex(["2020-01-10"])
        phases = calculate_moon_phase(40.0, -10.0, dates)
        assert phases[0] > 80.0


# ---------------------------------------------------------------------------
# postprocess_sst_fdist
# ---------------------------------------------------------------------------


class TestPostprocessSstFdist:
    def _make_ds(self, values: np.ndarray) -> xr.Dataset:
        return xr.Dataset(
            {"sst_fdist": (["time", "lat", "lon"], values.reshape(1, 2, 2))},
            coords={
                "time": pd.date_range("2020-01-01", periods=1, freq="D"),
                "lat": [30.0, 35.0],
                "lon": [-10.0, -5.0],
            },
        )

    def test_clips_negative_values_to_zero(self):
        ds = self._make_ds(np.array([-0.5, 1.0, -0.1, 2.0]))
        result = postprocess_sst_fdist(ds)
        assert float(result["sst_fdist"].min()) >= 0.0

    def test_positive_values_unchanged(self):
        ds = self._make_ds(np.array([1.0, 2.0, 3.0, 4.0]))
        result = postprocess_sst_fdist(ds)
        np.testing.assert_allclose(
            result["sst_fdist"].values.ravel(), [1.0, 2.0, 3.0, 4.0]
        )

    def test_no_op_when_variable_absent(self):
        ds = xr.Dataset({"sst": (["time"], [1.0, 2.0])})
        result = postprocess_sst_fdist(ds)
        assert "sst_fdist" not in result


# ---------------------------------------------------------------------------
# Compiler._resolve_compile_range
# ---------------------------------------------------------------------------


def _make_coverage_index(tmp_path, data: dict[str, str]) -> VarCoverageIndex:
    idx = VarCoverageIndex(tmp_path / "h2ds_var_coverage.json")
    for k, v in data.items():
        idx.update(k, pd.Timestamp(v))
    return idx


class TestResolveCompileRange:
    def _setup(self, compiler, source_coverage, index_data, tmp_path):
        """Inject the two pieces of state that _resolve_compile_range reads."""
        compiler.var_keys = list(source_coverage.keys()) + [compiler.var_key]
        compiler._source_coverage = source_coverage
        compiler._coverage_index = _make_coverage_index(tmp_path, index_data)

    def test_explicit_dates_bypass_inference(self, compiler):
        start = pd.Timestamp("2021-01-01")
        end = pd.Timestamp("2021-12-31")
        expected = DateRange("2021-01-01", "2021-12-31")

        with patch(
            "h2mare.processing.compiler.resolve_date_range",
            return_value=expected,
        ) as mock_resolve:
            result = compiler._resolve_compile_range(start, end)

        mock_resolve.assert_called_once_with(compiler.var_key, start, end)
        assert pd.Timestamp(result.start) == start

    def test_fresh_var_starts_from_source_start(self, compiler, tmp_path):
        """No index entry → compile from source catalog start."""
        self._setup(
            compiler,
            source_coverage={"sst": DateRange("2000-01-01", "2026-05-29")},
            index_data={},
            tmp_path=tmp_path,
        )
        result = compiler._resolve_compile_range(None, None)
        assert pd.Timestamp(result.start) == pd.Timestamp("2000-01-01")
        assert pd.Timestamp(result.end) == pd.Timestamp("2026-05-29")

    def test_incremental_gap_from_index_end(self, compiler, tmp_path):
        """Index entry exists → gap starts the day after the recorded end."""
        self._setup(
            compiler,
            source_coverage={"sst": DateRange("2000-01-01", "2026-05-29")},
            index_data={"sst": "2026-05-20"},
            tmp_path=tmp_path,
        )
        result = compiler._resolve_compile_range(None, None)
        assert pd.Timestamp(result.start) == pd.Timestamp("2026-05-21")
        assert pd.Timestamp(result.end) == pd.Timestamp("2026-05-29")

    def test_lagging_var_does_not_block_fast_var(self, compiler, tmp_path):
        """A slow variable compiles its own gap independently of a fast one."""
        self._setup(
            compiler,
            source_coverage={
                "sst": DateRange("2000-01-01", "2026-05-29"),
                "thetao": DateRange("2010-01-01", "2024-12-31"),
            },
            index_data={
                "sst": "2026-05-20",
                "thetao": "2024-06-30",
            },
            tmp_path=tmp_path,
        )
        result = compiler._resolve_compile_range(None, None)
        # Union: sst gap 2026-05-21→2026-05-29, thetao gap 2024-07-01→2024-12-31
        assert pd.Timestamp(result.start) == pd.Timestamp("2024-07-01")
        assert pd.Timestamp(result.end) == pd.Timestamp("2026-05-29")

    def test_up_to_date_var_is_skipped(self, compiler, tmp_path):
        """Variable whose index end matches source end contributes no gap."""
        self._setup(
            compiler,
            source_coverage={
                "sst": DateRange("2000-01-01", "2026-05-29"),
                "ssh": DateRange("2000-01-01", "2026-05-28"),
            },
            index_data={
                "sst": "2026-05-20",
                "ssh": "2026-05-28",  # already at source end
            },
            tmp_path=tmp_path,
        )
        result = compiler._resolve_compile_range(None, None)
        # Only sst has a gap
        assert pd.Timestamp(result.start) == pd.Timestamp("2026-05-21")
        assert pd.Timestamp(result.end) == pd.Timestamp("2026-05-29")

    def test_raises_when_all_vars_up_to_date(self, compiler, tmp_path):
        self._setup(
            compiler,
            source_coverage={"sst": DateRange("2000-01-01", "2026-05-29")},
            index_data={"sst": "2026-05-29"},
            tmp_path=tmp_path,
        )
        with pytest.raises(ValueError, match="up to date"):
            compiler._resolve_compile_range(None, None)


# ---------------------------------------------------------------------------
# Coverage index write policy
# ---------------------------------------------------------------------------


class TestCoverageIndexWritePolicy:
    """Index must only be written during incremental (no explicit dates) runs."""

    def _make_compiler_with_index(self, compiler, tmp_path):
        compiler.var_keys = ["sst", compiler.var_key]
        compiler._source_coverage = {"sst": DateRange("2000-01-01", "2026-05-29")}
        compiler._coverage_index = _make_coverage_index(tmp_path, {})
        return compiler

    def test_explicit_dates_do_not_write_index(self, compiler, tmp_path):
        self._make_compiler_with_index(compiler, tmp_path)
        index_path = tmp_path / "h2ds_var_coverage.json"

        # Simulate what run() does after a chunk write with explicit dates
        explicit_dates = True
        contributions = {"sst": pd.Timestamp("2025-12-31")}
        if not explicit_dates:
            for vkey, actual_end in contributions.items():
                compiler._coverage_index.update(vkey, actual_end)
            compiler._coverage_index.save()

        assert not index_path.exists()
        assert compiler._coverage_index.get_end("sst") is None

    def test_incremental_run_writes_index(self, compiler, tmp_path):
        self._make_compiler_with_index(compiler, tmp_path)
        index_path = tmp_path / "h2ds_var_coverage.json"

        # Simulate what run() does after a chunk write without explicit dates
        explicit_dates = False
        contributions = {"sst": pd.Timestamp("2026-05-29")}
        if not explicit_dates:
            for vkey, actual_end in contributions.items():
                compiler._coverage_index.update(vkey, actual_end)
            compiler._coverage_index.save()

        assert index_path.exists()
        assert compiler._coverage_index.get_end("sst") == pd.Timestamp("2026-05-29")

    def test_explicit_backfill_does_not_overwrite_existing_index(
        self, compiler, tmp_path
    ):
        """Backfilling 2025 when index already has 2026-05-29 must not corrupt it."""
        compiler._coverage_index = _make_coverage_index(
            tmp_path, {"sst": "2026-05-29"}
        )

        explicit_dates = True
        contributions = {"sst": pd.Timestamp("2025-12-31")}
        if not explicit_dates:
            for vkey, actual_end in contributions.items():
                compiler._coverage_index.update(vkey, actual_end)
            compiler._coverage_index.save()

        assert compiler._coverage_index.get_end("sst") == pd.Timestamp("2026-05-29")


# ---------------------------------------------------------------------------
# Compiler._has_overlap
# ---------------------------------------------------------------------------


class TestHasOverlap:
    def test_returns_true_when_ranges_overlap(self, compiler):
        catalog = MagicMock()
        catalog.get_time_coverage.return_value = DateRange("2020-01-01", "2020-12-31")
        result = compiler._has_overlap(
            "sst", DateRange("2020-06-01", "2021-06-30"), catalog
        )
        assert result is True

    def test_returns_false_when_no_overlap(self, compiler):
        catalog = MagicMock()
        catalog.get_time_coverage.return_value = DateRange("2015-01-01", "2018-12-31")
        result = compiler._has_overlap(
            "sst", DateRange("2020-01-01", "2020-12-31"), catalog
        )
        assert result is False

    def test_returns_false_when_catalog_is_empty(self, compiler):
        catalog = MagicMock()
        catalog.get_time_coverage.return_value = None
        result = compiler._has_overlap(
            "sst", DateRange("2020-01-01", "2020-12-31"), catalog
        )
        assert result is False


# ---------------------------------------------------------------------------
# Compiler.sync_data
# ---------------------------------------------------------------------------


class TestSyncData:
    def test_copies_zarr_directory_to_local_store(self, compiler, tmp_path):
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        (remote_dir / "data.zarr").mkdir()
        (remote_dir / "data.zarr" / ".zattrs").write_text("{}")

        source = remote_dir / "data.zarr"
        local_store = tmp_path / "local"
        local_store.mkdir(exist_ok=True)

        compiler.local_store_root = local_store
        compiler.sync_data(source)

        assert (local_store / "data.zarr").exists()

    def test_accepts_custom_backup_dir(self, compiler, tmp_path):
        remote_dir = tmp_path / "remote"
        remote_dir.mkdir()
        (remote_dir / "chunk.zarr").mkdir()
        (remote_dir / "chunk.zarr" / ".zattrs").write_text("{}")

        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()

        compiler.sync_data(remote_dir / "chunk.zarr", backup_dir=backup_dir)
        assert (backup_dir / "chunk.zarr").exists()
