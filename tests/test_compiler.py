"""Tests for processing/compiler.py — Compiler class and helpers."""

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


def _setup_compiler(compiler, source_coverage, h2ds_var_ends):
    """
    Inject source coverage and mock _get_h2ds_var_end to return
    the given per-variable h2ds end dates (None = never compiled).
    """
    compiler.var_keys = list(source_coverage.keys()) + [compiler.var_key]
    compiler._source_coverage = source_coverage
    compiler._get_h2ds_var_end = lambda vkey: h2ds_var_ends.get(vkey)


class TestResolveCompileRange:
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

    def test_fresh_h2ds_starts_from_source_start(self, compiler):
        """h2ds empty (None) → compile from source catalog start."""
        _setup_compiler(
            compiler,
            source_coverage={"sst": DateRange("2000-01-01", "2026-05-29")},
            h2ds_var_ends={"sst": None},
        )
        result = compiler._resolve_compile_range(None, None)
        assert pd.Timestamp(result.start) == pd.Timestamp("2000-01-01")
        assert pd.Timestamp(result.end) == pd.Timestamp("2026-05-29")

    def test_incremental_gap_from_h2ds_end(self, compiler):
        """h2ds has data → gap starts the day after h2ds var end."""
        _setup_compiler(
            compiler,
            source_coverage={"sst": DateRange("2000-01-01", "2026-05-29")},
            h2ds_var_ends={"sst": pd.Timestamp("2026-05-20")},
        )
        result = compiler._resolve_compile_range(None, None)
        assert pd.Timestamp(result.start) == pd.Timestamp("2026-05-21")
        assert pd.Timestamp(result.end) == pd.Timestamp("2026-05-29")

    def test_lagging_var_does_not_block_fast_var(self, compiler):
        """A slow variable compiles its own gap independently of a fast one."""
        _setup_compiler(
            compiler,
            source_coverage={
                "sst": DateRange("2000-01-01", "2026-05-29"),
                "thetao": DateRange("2010-01-01", "2024-12-31"),
            },
            h2ds_var_ends={
                "sst": pd.Timestamp("2026-05-20"),
                "thetao": pd.Timestamp("2024-06-30"),
            },
        )
        result = compiler._resolve_compile_range(None, None)
        # Union: sst gap 2026-05-21→2026-05-29, thetao gap 2024-07-01→2024-12-31
        assert pd.Timestamp(result.start) == pd.Timestamp("2024-07-01")
        assert pd.Timestamp(result.end) == pd.Timestamp("2026-05-29")

    def test_up_to_date_var_is_skipped(self, compiler):
        """Variable whose h2ds end matches source end contributes no gap."""
        _setup_compiler(
            compiler,
            source_coverage={
                "sst": DateRange("2000-01-01", "2026-05-29"),
                "ssh": DateRange("2000-01-01", "2026-05-28"),
            },
            h2ds_var_ends={
                "sst": pd.Timestamp("2026-05-20"),
                "ssh": pd.Timestamp("2026-05-28"),
            },
        )
        result = compiler._resolve_compile_range(None, None)
        assert pd.Timestamp(result.start) == pd.Timestamp("2026-05-21")
        assert pd.Timestamp(result.end) == pd.Timestamp("2026-05-29")

    def test_returns_none_when_all_vars_up_to_date(self, compiler):
        """Up-to-date incremental run is a benign no-op, signalled by None."""
        _setup_compiler(
            compiler,
            source_coverage={"sst": DateRange("2000-01-01", "2026-05-29")},
            h2ds_var_ends={"sst": pd.Timestamp("2026-05-29")},
        )
        assert compiler._resolve_compile_range(None, None) is None


# ---------------------------------------------------------------------------
# Compiler._get_h2ds_var_end
# ---------------------------------------------------------------------------


_EDDIES_ENTRY = {
    "local_folder": "eddies",
    "variables": ["amplitude", "speed_average"],  # raw source names
    "variables_to_compile": ["ac_amp", "c_amp"],  # compiled h2ds names
    "dataset_id_rep": "aviso-eddies",
    "source": "aviso",
    "pattern": r".*",
    "subset": True,
    "bbox": (-80, 0, 10, 70),
}


def _compiler_with_eddies(tmp_path):
    cfg = msgspec.convert(
        {"variables": {"h2ds": _H2DS_ENTRY, "eddies": _EDDIES_ENTRY}, "secrets": {}},
        AppConfig,
    )
    with patch("h2mare.processing.compiler.ZarrCatalog"):
        c = Compiler(
            var_key="h2ds",
            app_config=cfg,
            remote_store_root=tmp_path / "remote",
            local_store_root=tmp_path / "local",
        )
    c.var_keys = ["eddies", "h2ds"]
    c._source_coverage = {"eddies": DateRange("2026-01-01", "2026-05-16")}
    return c


class TestGetH2dsVarEnd:
    def test_uses_nonnull_end_of_representative_compiled_column(self, tmp_path):
        """Keyed off variables_to_compile[0], measured by non-null data."""
        c = _compiler_with_eddies(tmp_path)
        c.catalog.get_vars_nonnull_end.return_value = {
            "ac_amp": pd.Timestamp("2026-05-15")
        }
        assert c._get_h2ds_var_end("eddies") == pd.Timestamp("2026-05-15")
        # rep column is the first variables_to_compile entry
        c.catalog.get_vars_nonnull_end.assert_called_once_with(["ac_amp"])

    def test_falls_back_to_file_end_when_no_nonnull(self, tmp_path):
        c = _compiler_with_eddies(tmp_path)
        c.catalog.get_vars_nonnull_end.return_value = {}
        c.catalog.get_var_time_coverage.return_value = DateRange(
            "2026-01-01", "2026-05-30"
        )
        assert c._get_h2ds_var_end("eddies") == pd.Timestamp("2026-05-30")

    def test_returns_none_when_column_absent(self, tmp_path):
        c = _compiler_with_eddies(tmp_path)
        c.catalog.get_vars_nonnull_end.return_value = {}
        c.catalog.get_var_time_coverage.return_value = None
        assert c._get_h2ds_var_end("eddies") is None


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
