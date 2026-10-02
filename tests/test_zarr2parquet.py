"""
Unit tests for Zarr2Parquet.

Covers:
- __init__ raises ValueError when no zarr data exists
- _resolve_date_range: returns a DateRange; all branches (explicit, first-run,
  incremental, up-to-date)
- run(): always splits by month regardless of range length
- run(): depth filtering logic for depth-aware variables
- sync_data(): skips when STORE_ROOT is None; copies when remote_root is given
"""

import re
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from loguru import logger

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
    def test_returns_date_range(self, tmp_path):
        """The window is a DateRange, not a bare (start, end) tuple."""
        z = _make_converter(tmp_path)
        assert isinstance(z._resolve_date_range("1998-03-01", "1998-06-30"), DateRange)

    def test_explicit_dates_returned_unchanged(self, tmp_path):
        z = _make_converter(tmp_path)
        window = z._resolve_date_range("1998-03-01", "1998-06-30")
        assert window.start == pd.Timestamp("1998-03-01")
        assert window.end == pd.Timestamp("1998-06-30")

    def test_explicit_start_after_end_raises(self, tmp_path):
        """The argument-level message wins over DateRange's generic one."""
        z = _make_converter(tmp_path)
        with pytest.raises(ValueError, match="before end_date"):
            z._resolve_date_range("1998-12-31", "1998-01-01")

    def test_first_run_uses_full_zarr_range(self, tmp_path):
        """Empty parquet store → infer zarr_start → zarr_end."""
        z = _make_converter(tmp_path, parquet_initialized=False)
        window = z._resolve_date_range(None, None)
        assert window.start == pd.Timestamp("1998-01-01")
        assert window.end == pd.Timestamp("1998-12-31")

    def test_incremental_run_starts_after_parquet_end(self, tmp_path):
        """Existing parquet → infer parquet_end + 1 day → zarr_end."""
        z = _make_converter(
            tmp_path,
            parquet_initialized=True,
            parquet_end=pd.Timestamp("1998-06-30"),
        )
        window = z._resolve_date_range(None, None)
        assert window.start == pd.Timestamp("1998-07-01")
        assert window.end == pd.Timestamp("1998-12-31")

    def test_already_up_to_date_returns_none(self, tmp_path):
        """Inferred start beyond zarr end → nothing to convert, a clean no-op.

        Mirrors storage.coverage.resolve_date_range: inferred ranges signal
        'nothing to do' with None; only explicit dates raise.
        """
        z = _make_converter(
            tmp_path,
            parquet_initialized=True,
            parquet_end=pd.Timestamp("1998-12-31"),
        )
        assert z._resolve_date_range(None, None) is None

    def test_explicit_end_only_uses_inferred_start(self, tmp_path):
        """Partial override: explicit end_date, inferred start from empty store."""
        z = _make_converter(tmp_path, parquet_initialized=False)
        window = z._resolve_date_range(None, "1998-06-30")
        assert window.start == pd.Timestamp("1998-01-01")  # zarr_start (empty parquet)
        assert window.end == pd.Timestamp("1998-06-30")


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

    def test_returns_false_when_all_chunks_fail(self, tmp_path):
        """run() returns False when every chunk raises."""
        z = _make_converter(tmp_path)
        z.zarr_repo.open_dataset.side_effect = RuntimeError("zarr read error")

        assert z.run("1998-01-01", "1998-03-31") is False

    def test_partial_override_with_nothing_left_is_a_noop(self, tmp_path):
        """Explicit end_date already covered by the store: skip, do not fail.

        Mode 2 used to let the 'already up to date' ValueError escape run(),
        which PipelineManager counted as a failed pipeline.
        """
        z = _make_converter(
            tmp_path,
            parquet_initialized=True,
            parquet_end=pd.Timestamp("1998-12-31"),  # == zarr end
        )

        with patch.object(z, "_convert_window", return_value=True) as mock_conv:
            result = z.run(end_date=pd.Timestamp("1998-12-31"))

        assert result is True
        mock_conv.assert_not_called()

    def test_returns_false_when_any_chunk_fails(self, tmp_path):
        """run() returns False even when only one of several chunks fails."""
        z = _make_converter(tmp_path)

        mock_ds = MagicMock()
        mock_ds.dims = {}
        mock_ds.to_dataframe.return_value.reset_index.return_value = MagicMock()

        # First chunk succeeds, second fails, third succeeds.
        z.zarr_repo.open_dataset.side_effect = [
            mock_ds,
            RuntimeError("zarr read error"),
            mock_ds,
        ]

        with patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl:
            mock_pl.from_pandas.return_value = MagicMock()
            result = z.run("1998-01-01", "1998-03-31")

        assert result is False

    def test_returns_true_when_all_chunks_succeed(self, tmp_path):
        """run() returns True when every chunk converts without error."""
        z = _make_converter(tmp_path)

        mock_ds = MagicMock()
        mock_ds.dims = {}
        mock_ds.to_dataframe.return_value.reset_index.return_value = MagicMock()
        z.zarr_repo.open_dataset.return_value = mock_ds

        with patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl:
            mock_pl.from_pandas.return_value = MagicMock()
            result = z.run("1998-01-01", "1998-03-31")

        assert result is True


# ---------------------------------------------------------------------------
# run() — depth filtering
# ---------------------------------------------------------------------------


class TestRunDepthFiltering:
    def _make_mock_ds(self, *, has_depth: bool) -> MagicMock:
        """Return a dataset mock with or without a depth dimension."""
        mock_ds = MagicMock()
        # Use a real dict so 'in' checks work correctly.
        mock_ds.dims = {
            "time": 3,
            "lat": 4,
            "lon": 4,
            **({"depth": 5} if has_depth else {}),
        }
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
        with patch(
            "h2mare.format_converters.zarr2parquet.get_settings"
        ) as mock_get_settings:
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
            mock_catalog.df = __import__(
                "pandas"
            ).DataFrame()  # empty → fallback to var_key
            z = Zarr2Parquet("h2ds", parquet_parent)

        z.sync_data(remote_root=remote_root)

        assert (remote_root / "h2ds" / "data.parquet").exists()

    def test_skips_when_destination_is_the_store_itself(self, tmp_path):
        """
        The default backup root, STORE_ROOT/parquet, is now where the store
        lives. Copying it onto itself failed file by file and logged an error.
        """
        z = _make_converter(tmp_path)
        z.parquet_root.mkdir(parents=True)
        (z.parquet_root / "data.parquet").write_bytes(b"\x00")

        with patch("h2mare.utils.files_io.shutil.copytree") as mock_copytree:
            z.sync_data(remote_root=z.parquet_root.parent)

        mock_copytree.assert_not_called()

    def test_mirror_removes_what_the_source_no_longer_has(self, tmp_path):
        """
        A partition rewritten into fewer part files, or dropped entirely, must not
        survive in the backup — stale part files read back as duplicate rows.
        """
        z = _make_converter(tmp_path)
        month = z.parquet_root / "year=2020" / "month=1"
        month.mkdir(parents=True)
        (month / "part-0.parquet").write_bytes(b"rewritten")

        backup = tmp_path / "backup"
        old_month = backup / z.parquet_root.name / "year=2020" / "month=1"
        old_month.mkdir(parents=True)
        (old_month / "part-0.parquet").write_bytes(b"old")
        (old_month / "part-1.parquet").write_bytes(b"old")
        dropped = backup / z.parquet_root.name / "year=2019"
        (dropped / "month=12").mkdir(parents=True)
        (dropped / "month=12" / "part-0.parquet").write_bytes(b"old")
        neighbour = backup / "other_store" / "keep.parquet"
        neighbour.parent.mkdir(parents=True)
        neighbour.write_bytes(b"keep")

        z.sync_data(remote_root=backup)

        assert (old_month / "part-0.parquet").read_bytes() == b"rewritten"
        assert not (old_month / "part-1.parquet").exists()
        assert not dropped.exists()
        assert neighbour.exists()  # only the store's own folder is mirrored

    def test_empty_source_does_not_wipe_the_backup(self, tmp_path):
        z = _make_converter(tmp_path)
        z.parquet_root.mkdir(parents=True)

        backup = tmp_path / "backup"
        kept = backup / z.parquet_root.name / "year=2020" / "part-0.parquet"
        kept.parent.mkdir(parents=True)
        kept.write_bytes(b"old")

        z.sync_data(remote_root=backup)

        assert kept.exists()

    def test_skips_a_destination_containing_the_source(self, tmp_path):
        """Pruning a destination that holds the store would delete the store."""
        z = _make_converter(tmp_path)
        # The destination store folder is tmp_path/"h2ds", which holds the source.
        z.parquet_root = tmp_path / "h2ds" / "inner" / "h2ds"
        z.parquet_root.mkdir(parents=True)
        (z.parquet_root / "part-0.parquet").write_bytes(b"live")

        z.sync_data(remote_root=tmp_path)

        assert (z.parquet_root / "part-0.parquet").exists()


# ---------------------------------------------------------------------------
# _resolve_backfill_groups
# ---------------------------------------------------------------------------


class TestResolveBackfillGroups:
    def _prime(
        self,
        z: Zarr2Parquet,
        *,
        parquet_end: pd.Timestamp,
        zarr_vars: set[str],
        var_end: dict,
        hole_start: dict | None = None,
    ) -> None:
        """Wire up the mocked indexer/zarr_repo state the resolver reads.

        ``var_end`` maps a representative column to its last non-null date in the
        parquet store, mirroring ``get_var_coverage_end``; ``hole_start`` mirrors
        ``get_var_backfill_start`` (earliest all-null date per column).
        """
        z.indexer._dataset_meta_initialized = True
        z.indexer.get_time_coverage.return_value = DateRange(
            pd.Timestamp("1998-01-01"), parquet_end
        )
        z.indexer.get_var_coverage_end.return_value = var_end
        z.indexer.get_var_backfill_start.return_value = hole_start or {}
        z.zarr_repo.get_variables.return_value = zarr_vars

    def test_empty_when_store_uninitialized(self, tmp_path):
        z = _make_converter(tmp_path, parquet_initialized=False)
        z.indexer._dataset_meta_initialized = False
        assert z._resolve_backfill_groups() == []

    def test_detects_lagging_var_key(self, tmp_path):
        """sst parquet ends earlier than its source coverage → backfill window."""
        z = _make_converter(tmp_path)
        self._prime(
            z,
            parquet_end=pd.Timestamp("2021-06-30"),
            zarr_vars={"sst"},
            var_end={"sst": pd.Timestamp("2021-03-31")},
        )
        with patch(
            "h2mare.format_converters.zarr2parquet.get_store_coverage"
        ) as mock_src:
            mock_src.return_value = DateRange(
                pd.Timestamp("1998-01-01"), pd.Timestamp("2021-06-30")
            )
            groups = z._resolve_backfill_groups()

        assert len(groups) == 1
        window, cols = groups[0]
        assert window.start.date() == date(2021, 4, 1)  # parquet end + 1 day
        assert window.end.date() == date(2021, 6, 30)  # capped at parquet end
        # All of sst's compiled_vars are read, not just the representative
        assert "sst" in cols
        assert "sst_std" in cols

    def test_up_to_date_var_key_excluded(self, tmp_path):
        """parquet column already at source end → no backfill."""
        z = _make_converter(tmp_path)
        self._prime(
            z,
            parquet_end=pd.Timestamp("2021-06-30"),
            zarr_vars={"sst"},
            var_end={"sst": pd.Timestamp("2021-06-30")},
        )
        with patch(
            "h2mare.format_converters.zarr2parquet.get_store_coverage"
        ) as mock_src:
            mock_src.return_value = DateRange(
                pd.Timestamp("1998-01-01"), pd.Timestamp("2021-06-30")
            )
            assert z._resolve_backfill_groups() == []

    def test_window_capped_at_parquet_end(self, tmp_path):
        """Even when the source extends past parquet, backfill stops at parquet end."""
        z = _make_converter(tmp_path)
        self._prime(
            z,
            parquet_end=pd.Timestamp("2021-06-30"),
            zarr_vars={"sst"},
            var_end={"sst": pd.Timestamp("2021-03-31")},
        )
        with patch(
            "h2mare.format_converters.zarr2parquet.get_store_coverage"
        ) as mock_src:
            mock_src.return_value = DateRange(
                pd.Timestamp("1998-01-01"), pd.Timestamp("2025-12-31")
            )
            groups = z._resolve_backfill_groups()

        _, _ = groups[0]
        assert groups[0][0].end.date() == date(2021, 6, 30)

    def test_new_column_backfilled_from_source_start(self, tmp_path):
        """A var_key absent from the parquet store backfills from its source start."""
        z = _make_converter(tmp_path)
        self._prime(
            z,
            parquet_end=pd.Timestamp("2021-06-30"),
            zarr_vars={"sst"},
            var_end={},  # sst never written to parquet
        )
        with patch(
            "h2mare.format_converters.zarr2parquet.get_store_coverage"
        ) as mock_src:
            mock_src.return_value = DateRange(
                pd.Timestamp("2000-01-01"), pd.Timestamp("2021-06-30")
            )
            groups = z._resolve_backfill_groups()

        assert groups[0][0].start.date() == date(2000, 1, 1)

    def test_var_key_not_in_zarr_skipped(self, tmp_path):
        """A var_key whose columns are not in this Zarr store is ignored."""
        z = _make_converter(tmp_path)
        self._prime(
            z,
            parquet_end=pd.Timestamp("2021-06-30"),
            zarr_vars=set(),  # nothing present
            var_end={},
        )
        assert z._resolve_backfill_groups() == []

    def test_fillable_hole_widens_window_past_nonnull_island(self, tmp_path):
        """The island scenario: sst non-null again on recent dates, but a NaN
        stretch sits behind it. The window must start at the hole, not at
        last-non-null + 1 day."""
        z = _make_converter(tmp_path)
        self._prime(
            z,
            parquet_end=pd.Timestamp("2021-06-30"),
            zarr_vars={"sst"},
            var_end={"sst": pd.Timestamp("2021-06-03")},  # island
            hole_start={"sst": pd.Timestamp("2021-05-01")},
        )
        with (
            patch(
                "h2mare.format_converters.zarr2parquet.get_store_coverage"
            ) as mock_src,
            patch.object(
                Zarr2Parquet,
                "_fillable_hole_dates",
                return_value=[pd.Timestamp("2021-05-01")],
            ),
        ):
            mock_src.return_value = DateRange(
                pd.Timestamp("1998-01-01"), pd.Timestamp("2021-06-30")
            )
            groups = z._resolve_backfill_groups()

        assert len(groups) == 1
        assert groups[0][0].start.date() == date(2021, 5, 1)
        assert groups[0][0].end.date() == date(2021, 6, 30)

    def test_unfillable_hole_falls_back_to_end_based_window(self, tmp_path):
        """A hole the Zarr cannot fill (legitimate source gap, e.g. chl days the
        raw product never published) must not widen the window — otherwise every
        incremental run re-merges a window that can never converge."""
        z = _make_converter(tmp_path)
        self._prime(
            z,
            parquet_end=pd.Timestamp("2021-06-30"),
            zarr_vars={"sst"},
            var_end={"sst": pd.Timestamp("2021-06-03")},
            hole_start={"sst": pd.Timestamp("2021-05-01")},
        )
        with (
            patch(
                "h2mare.format_converters.zarr2parquet.get_store_coverage"
            ) as mock_src,
            patch.object(Zarr2Parquet, "_fillable_hole_dates", return_value=[]),
        ):
            mock_src.return_value = DateRange(
                pd.Timestamp("1998-01-01"), pd.Timestamp("2021-06-30")
            )
            groups = z._resolve_backfill_groups()

        # End-based gap (Jun 4 → Jun 30) still backfills; the hole does not.
        assert len(groups) == 1
        assert groups[0][0].start.date() == date(2021, 6, 4)

    def test_hole_floor_passes_source_start_and_lookback(self, tmp_path):
        """get_var_backfill_start receives a per-column floor bounded by source
        start and the lookback window."""
        z = _make_converter(tmp_path)
        parquet_end = pd.Timestamp("2021-06-30")
        self._prime(
            z,
            parquet_end=parquet_end,
            zarr_vars={"sst"},
            var_end={"sst": parquet_end},
        )
        with patch(
            "h2mare.format_converters.zarr2parquet.get_store_coverage"
        ) as mock_src:
            mock_src.return_value = DateRange(pd.Timestamp("1998-01-01"), parquet_end)
            z._resolve_backfill_groups()

        _, kwargs = z.indexer.get_var_backfill_start.call_args
        floor = kwargs["not_before"]["sst"]
        # 1998 source start is older than the lookback → lookback wins.
        assert pd.Timestamp(floor) == parquet_end - pd.Timedelta(days=400)


# ---------------------------------------------------------------------------
# _fillable_hole_dates
# ---------------------------------------------------------------------------


class TestFillableHoleDates:
    def test_separates_lag_holes_from_source_gaps(self, tmp_path):
        import numpy as np
        import polars as pl
        import xarray as xr

        z = _make_converter(tmp_path)
        times = pd.date_range("2021-05-01", periods=4, freq="D")
        data = np.ones((4, 2, 2))
        data[2] = np.nan  # May 3: the zarr has no data either (source gap)
        z.zarr_repo.open_dataset.return_value = xr.Dataset(
            {"sst": (["time", "lat", "lon"], data)},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        # Parquet: sst populated on May 1 only; May 2 row exists but is null.
        z.indexer.time_col = "time"
        z.indexer.scan.return_value = pl.LazyFrame(
            {"time": [date(2021, 5, 1), date(2021, 5, 2)], "sst": [1.0, None]}
        )

        out = z._fillable_hole_dates("sst", DateRange("2021-05-01", "2021-05-04"))

        # Zarr has data on May 1, 2, 4; parquet covers May 1 → fillable: 2 and 4.
        # May 3 is a source gap and must not appear.
        assert out == [pd.Timestamp("2021-05-02"), pd.Timestamp("2021-05-04")]

    def test_no_zarr_files_returns_empty(self, tmp_path):
        z = _make_converter(tmp_path)
        z.zarr_repo.open_dataset.side_effect = FileNotFoundError("no files")
        assert (
            z._fillable_hole_dates("sst", DateRange("2021-05-01", "2021-05-04")) == []
        )


# ---------------------------------------------------------------------------
# run() — incremental mode orchestration
# ---------------------------------------------------------------------------


class TestRunIncremental:
    def test_appends_then_backfills(self, tmp_path):
        """Mode 3 runs the append window, then each backfill group."""
        z = _make_converter(tmp_path)

        backfill_window = DateRange(
            pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30")
        )
        with (
            patch.object(z, "_convert_window", return_value=True) as mock_conv,
            patch.object(
                z,
                "_resolve_date_range",
                return_value=DateRange(
                    pd.Timestamp("2021-07-01"), pd.Timestamp("2021-12-31")
                ),
            ),
            patch.object(
                z,
                "_resolve_backfill_groups",
                return_value=[(backfill_window, {"sst", "sst_fdist"})],
            ),
        ):
            result = z.run()

        assert result is True
        assert mock_conv.call_count == 2
        # First call = append (all variables → None)
        first = mock_conv.call_args_list[0]
        assert first.args[3] is None
        # Second call = backfill (explicit column subset)
        second = mock_conv.call_args_list[1]
        assert second.args[3] == ["sst", "sst_fdist"]

    def test_each_window_logs_exactly_one_labelled_line(self, tmp_path):
        """Regression: every backfill window was announced twice — once by run()
        and again by _convert_window's generic header, with the same dates on
        both lines and nothing marking which regime produced them.

        Also pins the pivot line: the pre-append store end is what separates
        'appending after' from 'backfilling within', and the windows below
        cannot be interpreted without it.
        """
        z = _make_converter(
            tmp_path,
            parquet_initialized=True,
            parquet_end=pd.Timestamp("1998-06-30"),
        )

        messages: list[str] = []
        sink = logger.add(messages.append, level="INFO", format="{message}")
        try:
            with (
                patch.object(z, "_convert_window", wraps=z._convert_window),
                patch.object(
                    z,
                    "_resolve_backfill_groups",
                    return_value=[
                        (
                            DateRange(
                                pd.Timestamp("1998-06-01"), pd.Timestamp("1998-06-30")
                            ),
                            {"sst"},
                        )
                    ],
                ),
                patch("h2mare.format_converters.zarr2parquet.pl") as mock_pl,
            ):
                mock_pl.from_pandas.return_value = MagicMock()
                z.zarr_repo.open_dataset.return_value = MagicMock(dims={})
                z.run()
        finally:
            logger.remove(sink)

        assert any(
            m.startswith("Parquet store covers 1998-01-01 → 1998-06-30")
            for m in messages
        )
        # One line per window, each naming its regime — not two saying the same.
        assert sum(m.startswith("Appending new dates:") for m in messages) == 1
        assert sum(m.startswith("Backfilling ['sst']") for m in messages) == 1
        # The step header is logged once for the whole run, never once per
        # window — announcing each window twice is the regression above. It has
        # to be logged at all, or the first line of a conversion is a bare
        # "Converting requested range: ..." naming neither the step nor where it
        # writes. Checked for *not* being the completion line, which used to be
        # the only place the phrase appeared and would otherwise satisfy a
        # count on its own.
        headers = [m for m in messages if m.startswith("Zarr → Parquet conversion for")]
        assert len(headers) == 1
        assert "complete" not in headers[0]

    def test_the_run_is_bracketed_by_a_header_and_an_elapsed_time(self, tmp_path):
        """
        The step has to be identifiable from the log without reading the module
        column, and it has to say how long it took — as the netcdf2zarr and
        compile steps both already do. Neither the destination nor the duration
        appears anywhere else in this path.
        """
        z = _make_converter(
            tmp_path,
            parquet_initialized=True,
            parquet_end=pd.Timestamp("1998-06-30"),
        )

        messages: list[str] = []
        sink = logger.add(messages.append, level="INFO", format="{message}")
        try:
            with (
                patch.object(z, "_convert_window", return_value=True),
                patch.object(z, "_resolve_backfill_groups", return_value=[]),
            ):
                assert z.run() is True
        finally:
            logger.remove(sink)

        header = [m for m in messages if m.startswith("Zarr → Parquet conversion for")]
        assert len(header) == 1
        assert "'H2DS'" in header[0]
        assert str(z.parquet_root) in header[0], (
            "the destination is stated nowhere else"
        )

        closing = [m for m in messages if m.startswith("Conversion complete for")]
        assert len(closing) == 1
        assert re.search(r"in \d+\.\d+s\.$", closing[0]), (
            f"no elapsed time in the closing line: {closing[0]!r}"
        )
        # The header phrase is not repeated by the closing line.
        assert not closing[0].startswith("Zarr → Parquet")

    def test_backfill_runs_even_when_nothing_to_append(self, tmp_path):
        """An 'up to date' append still lets backfill proceed.

        The store is primed so the real resolver returns None — the append is
        skipped through the production code path, not a patched exception.
        """
        z = _make_converter(
            tmp_path,
            parquet_initialized=True,
            parquet_end=pd.Timestamp("1998-12-31"),  # == zarr end
        )

        backfill_window = DateRange(
            pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30")
        )
        with (
            patch.object(z, "_convert_window", return_value=True) as mock_conv,
            patch.object(
                z,
                "_resolve_backfill_groups",
                return_value=[(backfill_window, {"sst"})],
            ),
        ):
            result = z.run()

        assert result is True
        assert mock_conv.call_count == 1  # only the backfill
        assert mock_conv.call_args_list[0].args[0] == backfill_window

    def test_nothing_to_append_and_no_backfill_is_a_clean_noop(self, tmp_path):
        """Fully up-to-date store: no conversion, and run() still succeeds."""
        z = _make_converter(
            tmp_path,
            parquet_initialized=True,
            parquet_end=pd.Timestamp("1998-12-31"),  # == zarr end
        )

        with (
            patch.object(z, "_convert_window", return_value=True) as mock_conv,
            patch.object(z, "_resolve_backfill_groups", return_value=[]),
        ):
            result = z.run()

        assert result is True
        mock_conv.assert_not_called()

    def test_convert_window_error_is_not_swallowed(self, tmp_path):
        """A real failure must not be reported as 'nothing to append'.

        The append call used to sit inside a ``except ValueError`` that logged
        every failure as a benign no-op and left the return value True.
        """
        z = _make_converter(tmp_path, parquet_initialized=False)

        with (
            patch.object(
                z, "_convert_window", side_effect=ValueError("invalid split value")
            ),
            patch.object(z, "_resolve_backfill_groups", return_value=[]),
        ):
            with pytest.raises(ValueError, match="invalid split value"):
                z.run()
