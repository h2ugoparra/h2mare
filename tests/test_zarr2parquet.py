"""
Unit tests for Zarr2Parquet.

Covers:
- __init__ raises ValueError when no zarr data exists
- _resolve_date_range: all branches (explicit, first-run, incremental, up-to-date)
- run(): always splits by month regardless of range length
- run(): depth filtering logic for depth-aware variables
- sync_data(): skips when STORE_ROOT is None; copies when remote_root is given
"""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from h2mare.format_converters.zarr2parquet import Zarr2Parquet
from h2mare.types import DateRange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZARR_RANGE = DateRange(
    start=pd.Timestamp("1998-01-01"),
    end=pd.Timestamp("1998-12-31"),
)


def _make_converter(
    tmp_path: Path,
    *,
    parquet_initialized: bool = False,
    parquet_end: pd.Timestamp | None = None,
) -> Zarr2Parquet:
    """
    Build a Zarr2Parquet with all I/O mocked out.

    The patches only need to be active during __init__ (the mock instances are
    stored on the returned object and keep their behaviour afterwards).
    """
    mock_indexer = MagicMock()
    mock_indexer._dataset_meta_initialized = parquet_initialized
    if parquet_initialized and parquet_end is not None:
        mock_indexer.get_time_coverage.return_value = DateRange(
            start=_ZARR_RANGE.start, end=parquet_end
        )

    with (
        patch("h2mare.format_converters.zarr2parquet.ZarrCatalog") as MockCatalog,
        patch(
            "h2mare.format_converters.zarr2parquet.ParquetIndexer",
            return_value=mock_indexer,
        ),
    ):
        MockCatalog.return_value.get_time_coverage.return_value = _ZARR_RANGE
        z = Zarr2Parquet("h2ds", tmp_path / "parquet")

    return z


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:

    def test_raises_when_no_zarr_data(self, tmp_path):
        """ValueError is raised immediately when the zarr catalog is empty."""
        with (
            patch("h2mare.format_converters.zarr2parquet.ZarrCatalog") as MockCatalog,
            patch("h2mare.format_converters.zarr2parquet.ParquetIndexer"),
        ):
            MockCatalog.return_value.get_time_coverage.return_value = None
            with pytest.raises(ValueError, match="No zarr data"):
                Zarr2Parquet("h2ds", tmp_path / "parquet")

    def test_stores_repo_dates(self, tmp_path):
        """repo_start and repo_end are populated from the zarr catalog coverage."""
        z = _make_converter(tmp_path)
        assert z.repo_start == pd.Timestamp("1998-01-01")
        assert z.repo_end == pd.Timestamp("1998-12-31")


# ---------------------------------------------------------------------------
# _resolve_date_range
# ---------------------------------------------------------------------------

class TestResolveDateRange:

    def test_explicit_dates_returned_unchanged(self, tmp_path):
        z = _make_converter(tmp_path)
        start, end = z._resolve_date_range("1998-03-01", "1998-06-30")
        assert start == pd.Timestamp("1998-03-01")
        assert end == pd.Timestamp("1998-06-30")

    def test_explicit_start_after_end_raises(self, tmp_path):
        z = _make_converter(tmp_path)
        with pytest.raises(ValueError, match="before end_date"):
            z._resolve_date_range("1998-12-31", "1998-01-01")

    def test_first_run_uses_full_zarr_range(self, tmp_path):
        """Empty parquet store → infer zarr_start → zarr_end."""
        z = _make_converter(tmp_path, parquet_initialized=False)
        start, end = z._resolve_date_range(None, None)
        assert start == pd.Timestamp("1998-01-01")
        assert end == pd.Timestamp("1998-12-31")

    def test_incremental_run_starts_after_parquet_end(self, tmp_path):
        """Existing parquet → infer parquet_end + 1 day → zarr_end."""
        z = _make_converter(
            tmp_path,
            parquet_initialized=True,
            parquet_end=pd.Timestamp("1998-06-30"),
        )
        start, end = z._resolve_date_range(None, None)
        assert start == pd.Timestamp("1998-07-01")
        assert end == pd.Timestamp("1998-12-31")

    def test_already_up_to_date_raises(self, tmp_path):
        """Inferred start beyond zarr end → nothing new to convert."""
        z = _make_converter(
            tmp_path,
            parquet_initialized=True,
            parquet_end=pd.Timestamp("1998-12-31"),
        )
        with pytest.raises(ValueError, match="up to date"):
            z._resolve_date_range(None, None)

    def test_explicit_end_only_uses_inferred_start(self, tmp_path):
        """Partial override: explicit end_date, inferred start from empty store."""
        z = _make_converter(tmp_path, parquet_initialized=False)
        start, end = z._resolve_date_range(None, "1998-06-30")
        assert start == pd.Timestamp("1998-01-01")  # zarr_start (empty parquet)
        assert end == pd.Timestamp("1998-06-30")


# ---------------------------------------------------------------------------
# run() — always splits monthly
# ---------------------------------------------------------------------------

class TestRun:

    def test_open_dataset_called_once_per_month(self, tmp_path):
        """run() splits the requested range into monthly chunks."""
        z = _make_converter(tmp_path)

        mock_ds = MagicMock()
        mock_ds.to_dataframe.return_value.reset_index.return_value = MagicMock()
        z.zarr_repo.open_dataset.return_value = mock_ds

        with patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl:
            mock_pl.from_pandas.return_value = MagicMock()
            z.run("1998-01-01", "1998-03-31")

        # Jan, Feb, Mar → 3 monthly chunks
        assert z.zarr_repo.open_dataset.call_count == 3

    def test_open_dataset_called_for_single_month(self, tmp_path):
        """A range within one month still calls open_dataset exactly once."""
        z = _make_converter(tmp_path)

        mock_ds = MagicMock()
        mock_ds.to_dataframe.return_value.reset_index.return_value = MagicMock()
        z.zarr_repo.open_dataset.return_value = mock_ds

        with patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl:
            mock_pl.from_pandas.return_value = MagicMock()
            z.run("1998-01-01", "1998-01-31")

        assert z.zarr_repo.open_dataset.call_count == 1

    def test_open_dataset_error_logged_not_raised(self, tmp_path):
        """An error in open_dataset is caught and logged, not propagated."""
        z = _make_converter(tmp_path)
        z.zarr_repo.open_dataset.side_effect = RuntimeError("zarr read error")

        z.run("1998-01-01", "1998-01-31")  # must not raise


# ---------------------------------------------------------------------------
# run() — depth filtering
# ---------------------------------------------------------------------------

class TestRunDepthFiltering:

    def _make_mock_ds(self, *, has_depth: bool) -> MagicMock:
        """Return a dataset mock with or without a depth dimension."""
        mock_ds = MagicMock()
        # Use a real dict so 'in' checks work correctly.
        mock_ds.dims = {"time": 3, "lat": 4, "lon": 4, **({"depth": 5} if has_depth else {})}
        # sel() returns the same mock so the rest of the pipeline still works.
        mock_ds.sel.return_value = mock_ds
        mock_ds.to_dataframe.return_value.reset_index.return_value = MagicMock()
        return mock_ds

    def test_sel_called_with_correct_depth(self, tmp_path):
        """When depth is given and the dim exists, sel(depth=..., method='nearest') is called."""
        z = _make_converter(tmp_path)
        mock_ds = self._make_mock_ds(has_depth=True)
        z.zarr_repo.open_dataset.return_value = mock_ds

        with patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl:
            mock_pl.from_pandas.return_value = MagicMock()
            z.run("1998-01-01", "1998-01-31", depth=200.0)

        mock_ds.sel.assert_called_once_with(depth=200.0, method="nearest")

    def test_add_data_called_after_depth_sel(self, tmp_path):
        """Indexer receives data when depth filtering succeeds."""
        z = _make_converter(tmp_path)
        mock_ds = self._make_mock_ds(has_depth=True)
        z.zarr_repo.open_dataset.return_value = mock_ds

        with patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl:
            mock_pl.from_pandas.return_value = MagicMock()
            z.run("1998-01-01", "1998-01-31", depth=0.0)

        z.indexer.add_data.assert_called_once()

    def test_depth_ignored_for_surface_variable(self, tmp_path):
        """When depth is given but the dataset has no depth dim, sel is not called."""
        z = _make_converter(tmp_path)
        mock_ds = self._make_mock_ds(has_depth=False)
        z.zarr_repo.open_dataset.return_value = mock_ds

        with patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl:
            mock_pl.from_pandas.return_value = MagicMock()
            z.run("1998-01-01", "1998-01-31", depth=200.0)

        mock_ds.sel.assert_not_called()
        z.indexer.add_data.assert_called_once()

    def test_missing_depth_arg_skips_indexing(self, tmp_path):
        """No depth given but dim exists: the chunk is skipped (add_data not called)."""
        z = _make_converter(tmp_path)
        mock_ds = self._make_mock_ds(has_depth=True)
        z.zarr_repo.open_dataset.return_value = mock_ds

        with patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl:
            mock_pl.from_pandas.return_value = MagicMock()
            z.run("1998-01-01", "1998-01-31")  # no depth → error logged, chunk skipped

        z.indexer.add_data.assert_not_called()

    def test_no_depth_no_depth_dim_normal_flow(self, tmp_path):
        """Surface variable without depth arg: no sel call and add_data proceeds normally."""
        z = _make_converter(tmp_path)
        mock_ds = self._make_mock_ds(has_depth=False)
        z.zarr_repo.open_dataset.return_value = mock_ds

        with patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl:
            mock_pl.from_pandas.return_value = MagicMock()
            z.run("1998-01-01", "1998-01-31")

        mock_ds.sel.assert_not_called()
        z.indexer.add_data.assert_called_once()


# ---------------------------------------------------------------------------
# sync_data()
# ---------------------------------------------------------------------------

class TestSyncData:

    def test_skips_when_store_root_is_none(self, tmp_path):
        """sync_data() returns without error when STORE_ROOT is not configured."""
        z = _make_converter(tmp_path)
        with patch("h2mare.format_converters.zarr2parquet.get_settings") as mock_get_settings:
            mock_get_settings.return_value.STORE_ROOT = None
            z.sync_data()  # must not raise

    def test_copies_to_explicit_remote_root(self, tmp_path):
        """When remote_root is given explicitly, parquet_root is copied there."""
        # parquet_root is the PARENT; Zarr2Parquet appends the derived folder name.
        # With a mocked catalog (empty df) the fallback is var_key → "h2ds".
        parquet_parent = tmp_path / "local"
        derived_dir = parquet_parent / "h2ds"
        derived_dir.mkdir(parents=True)
        (derived_dir / "data.parquet").write_bytes(b"\x00")

        remote_root = tmp_path / "remote"

        with (
            patch("h2mare.format_converters.zarr2parquet.ZarrCatalog") as MockCatalog,
            patch("h2mare.format_converters.zarr2parquet.ParquetIndexer"),
        ):
            mock_catalog = MockCatalog.return_value
            mock_catalog.get_time_coverage.return_value = _ZARR_RANGE
            mock_catalog.df = __import__("pandas").DataFrame()  # empty → fallback to var_key
            z = Zarr2Parquet("h2ds", parquet_parent)

        z.sync_data(remote_root=remote_root)

        assert (remote_root / "data.parquet").exists()
