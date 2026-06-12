"""Unit tests for ParquetStore — write-path and overlap-resolution logic."""

from datetime import date, datetime

import pandas as pd
import polars as pl
import pytest
from conftest import make_grid_df

from h2mare.storage.parquet_store import ParquetStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, **kwargs) -> ParquetStore:
    return ParquetStore(tmp_path / "store", **kwargs)


def _write_one(store: ParquetStore, dates, variables=None) -> None:
    df = make_grid_df(dates, variables=variables or {"sst": 20.0})
    store.add_data(df)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestParquetStoreInit:
    def test_creates_directory_when_empty(self, tmp_path):
        store = _store(tmp_path)
        assert store.parquet_root.exists()

    def test_physical_schema_none_on_empty(self, tmp_path):
        store = _store(tmp_path)
        assert store.physical_schema is None
        assert store.physical_cols == set()

    def test_partition_cols_defaults(self, tmp_path):
        store = _store(tmp_path)
        assert store.partition_cols == {"year", "month"}
        assert store._partition_by == ["year", "month"]

    def test_raises_when_required_col_missing(self, tmp_path, jan_df):
        import polars.exceptions

        store = _store(tmp_path)
        store.add_data(jan_df)
        # polars raises ColumnNotFoundError when scanning with a non-existent
        # time column before the ValueError check is reached
        with pytest.raises((ValueError, polars.exceptions.ColumnNotFoundError)):
            ParquetStore(store.parquet_root, time_col="nonexistent")

    def test_physical_schema_populated_when_data_exists(self, tmp_path, jan_df):
        store = _store(tmp_path)
        store.add_data(jan_df)

        store2 = ParquetStore(store.parquet_root)
        assert store2.physical_schema is not None
        assert "sst" in store2.physical_schema


# ---------------------------------------------------------------------------
# Partition path helpers
# ---------------------------------------------------------------------------


class TestPartitionHelpers:
    def test_partition_path(self, tmp_path):
        store = _store(tmp_path)
        p = store._partition_path((2021, 6))
        assert p == store.parquet_root / "year=2021" / "month=6"

    def test_partition_filter_sql_single(self, tmp_path):
        store = _store(tmp_path)
        sql = store._partition_filter_sql([(2021, 6)])
        assert "year = 2021" in sql
        assert "month = 6" in sql

    def test_partition_filter_sql_multiple(self, tmp_path):
        store = _store(tmp_path)
        sql = store._partition_filter_sql([(2021, 6), (2021, 7)])
        assert "OR" in sql

    def test_partition_filter_expr_matches_row(self, tmp_path):
        store = _store(tmp_path)
        expr = store._partition_filter_expr((2021, 6))
        df = pl.DataFrame({"year": [2021, 2021], "month": [6, 7]})
        filtered = df.filter(expr)
        assert len(filtered) == 1
        assert filtered["month"][0] == 6


# ---------------------------------------------------------------------------
# atomic_partition_write
# ---------------------------------------------------------------------------


class TestAtomicPartitionWrite:
    def test_creates_partition_directory(self, tmp_path, jan_df):
        store = _store(tmp_path)
        store.add_data(jan_df)
        assert (store.parquet_root / "year=2020" / "month=1").exists()

    def test_parquet_file_is_readable(self, tmp_path, jan_df):
        store = _store(tmp_path)
        store.add_data(jan_df)
        files = list((store.parquet_root / "year=2020" / "month=1").rglob("*.parquet"))
        assert len(files) >= 1
        df = pl.read_parquet(files[0])
        assert len(df) > 0

    def test_removes_stale_tmp_dir_before_write(self, tmp_path):
        store = _store(tmp_path)
        df = make_grid_df([date(2021, 6, 1)])
        store.add_data(df)  # first write sets up schema

        # Simulate a stale tmp from a previously interrupted run
        tmp = store.parquet_root / ".tmp_write_2021_6"
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "stale.txt").touch()

        # Direct call to atomic_partition_write should remove the stale tmp
        df_physical = store._align_to_schema(
            store._prepare_df(
                store._resolve_time_col(make_grid_df([date(2021, 6, 1)]))
            ),
            include_partitions=False,
        )
        store.atomic_partition_write(df_physical, (2021, 6))
        assert not tmp.exists()

    def test_append_into_existing_partition_keeps_old_rows(self, tmp_path):
        """Regression: appending non-overlapping dates into a partition that
        already has files used to overwrite part-0.parquet (pyarrow restarts
        the default basename at 0 each write), silently destroying its rows —
        e.g. appending one new day into the current month's partition."""
        store = _store(tmp_path)
        store.add_data(make_grid_df([date(2021, 6, 1), date(2021, 6, 2)]))
        # Non-overlapping date in the same month → plain append path
        store.add_data(make_grid_df([date(2021, 6, 3)]))

        files = list((store.parquet_root / "year=2021" / "month=6").rglob("*.parquet"))
        loaded = pl.concat([pl.read_parquet(f) for f in files])
        days = set(loaded["time"].cast(pl.Utf8).to_list())
        assert days == {"2021-06-01", "2021-06-02", "2021-06-03"}

    def test_second_write_replaces_partition(self, tmp_path):
        store = _store(tmp_path)
        df1 = make_grid_df([date(2021, 1, 1)], variables={"sst": 5.0})
        store.add_data(df1)

        df2 = make_grid_df([date(2021, 1, 1)], variables={"sst": 99.0})
        store.add_data(df2)

        loaded = pl.read_parquet(
            list((store.parquet_root / "year=2021" / "month=1").rglob("*.parquet"))[0]
        )
        assert float(loaded["sst"].mean()) == pytest.approx(99.0, abs=1.0)


# ---------------------------------------------------------------------------
# resolve_dims_overlap
# ---------------------------------------------------------------------------


class TestResolveDimsOverlap:
    def _setup(self, tmp_path, dates=None, variables=None) -> ParquetStore:
        store = _store(tmp_path)
        _write_one(store, dates or [date(2021, 1, 1)], variables)
        return store

    def _prepare(self, store: ParquetStore, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the same pre-processing add_data does before calling resolve_dims_overlap."""
        df = store._resolve_time_col(df)
        return store._prepare_df(df)

    def test_returns_none_when_no_temporal_overlap(self, tmp_path):
        store = self._setup(tmp_path, [date(2021, 1, 1)])
        df_mar = self._prepare(store, make_grid_df([date(2021, 3, 1)]))
        # No temporal overlap → None (caller should append)
        assert store.resolve_dims_overlap(df_mar) is None

    def test_new_timestamps_within_overlap_are_retained(self, tmp_path):
        """FULL OUTER JOIN keeps rows from both sides — new timestamps must survive."""
        store = self._setup(tmp_path, [date(2021, 1, 1), date(2021, 1, 2)])
        # Overlap on Jan 1-2 (same partition) with a new variable AND a new date Jan 3
        df_new = make_grid_df(
            [date(2021, 1, 2), date(2021, 1, 3)], variables={"chl": 0.5}
        )
        store.add_data(df_new)

        # All four dates must be present (Jan 1 had no chl, Jan 3 had no sst)
        files = list(store.parquet_root.rglob("*.parquet"))
        loaded = pl.concat([pl.read_parquet(f) for f in files]).sort("time")
        dates_present = set(loaded["time"].cast(pl.Utf8).to_list())
        assert "2021-01-01" in dates_present
        assert "2021-01-03" in dates_present

    def test_returns_true_when_new_column_added(self, tmp_path):
        store = self._setup(tmp_path, [date(2021, 1, 1)])
        df_chl = make_grid_df([date(2021, 1, 1)], variables={"chl": 0.5})
        df_prep = self._prepare(store, df_chl)
        # _update_physical_schema is called internally by resolve_dims_overlap
        result = store.resolve_dims_overlap(df_prep)
        assert result is True

    def test_merged_data_has_both_variables(self, tmp_path):
        store = self._setup(tmp_path, [date(2021, 1, 1)])
        df_chl = make_grid_df([date(2021, 1, 1)], variables={"chl": 0.5})
        store.add_data(df_chl)

        files = list(store.parquet_root.rglob("*.parquet"))
        merged = pl.concat([pl.read_parquet(f) for f in files])
        assert "sst" in merged.columns
        assert "chl" in merged.columns

    def test_returns_true_when_duplicate_column_overwritten(self, tmp_path):
        store = self._setup(tmp_path, [date(2021, 1, 1)])
        df2 = make_grid_df([date(2021, 1, 1)], variables={"sst": 99.0})
        df_prep = self._prepare(store, df2)
        result = store.resolve_dims_overlap(df_prep)
        assert result is True

    def test_merge_spanning_multiple_partitions(self, tmp_path):
        """Each affected partition is merged independently — a new column
        spanning several month partitions must land in all of them, with
        existing values preserved."""
        store = self._setup(
            tmp_path,
            [date(2021, 1, 10), date(2021, 2, 10), date(2021, 3, 10)],
            variables={"sst": 20.0},
        )
        df_chl = make_grid_df(
            [date(2021, 1, 10), date(2021, 2, 10), date(2021, 3, 10)],
            variables={"chl": 0.5},
        )
        store.add_data(df_chl)

        for month in (1, 2, 3):
            files = list(
                (store.parquet_root / "year=2021" / f"month={month}").rglob("*.parquet")
            )
            assert files, f"partition month={month} missing"
            part = pl.concat([pl.read_parquet(f) for f in files])
            assert part["sst"].null_count() == 0, f"sst lost in month={month}"
            assert part["chl"].null_count() == 0, f"chl missing in month={month}"

    def test_raises_on_no_spatial_overlap(self, tmp_path):
        store = self._setup(tmp_path, [date(2021, 1, 1)], variables={"sst": 5.0})
        # New data at completely different coordinates
        df_far = make_grid_df(
            [date(2021, 1, 1)],
            lons=[100.0, 105.0],
            lats=[10.0, 15.0],
            variables={"sst": 5.0},
        )
        df_prep = self._prepare(store, df_far)
        with pytest.raises(ValueError, match="spatial overlap"):
            store.resolve_dims_overlap(df_prep)


# ---------------------------------------------------------------------------
# Coverage queries
# ---------------------------------------------------------------------------


class TestGetTimeCoverage:
    def test_raises_on_empty_store(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(RuntimeError, match="not initialized"):
            store.get_time_coverage()

    def test_returns_correct_range_after_write(self, tmp_path):
        store = _store(tmp_path)
        _write_one(store, [date(2021, 3, 1), date(2021, 3, 5)])
        cov = store.get_time_coverage()
        assert cov is not None
        assert cov.start.date() == date(2021, 3, 1)
        assert cov.end.date() == date(2021, 3, 5)

    def test_returns_correct_range_year_only_partition(self, tmp_path):
        store = ParquetStore(tmp_path / "store", partition_by=["year"])
        _write_one(store, [date(2021, 3, 1), date(2022, 9, 1)])
        cov = store.get_time_coverage()
        assert cov.start.date() == date(2021, 3, 1)
        assert cov.end.date() == date(2022, 9, 1)

    def test_min_max_found_across_split_partition_files(self, tmp_path):
        # Regression: when a boundary partition is split across several files
        # (row-count splits), only the alphabetically first/last file used to be
        # scanned — here the true min lives in b.parquet and the true max in
        # a.parquet, so the old shortcut reported 3/10 → 3/20.
        part = tmp_path / "store" / "year=2021" / "month=3"
        part.mkdir(parents=True)
        make_grid_df([date(2021, 3, 10), date(2021, 3, 28)]).write_parquet(
            part / "a.parquet"
        )
        make_grid_df([date(2021, 3, 1), date(2021, 3, 20)]).write_parquet(
            part / "b.parquet"
        )

        store = ParquetStore(tmp_path / "store")
        cov = store.get_time_coverage()
        assert cov.start.date() == date(2021, 3, 1)
        assert cov.end.date() == date(2021, 3, 28)


class TestGetVarBackfillStart:
    def _store_with_holes(self, tmp_path, dates, hole_dates, extra=None):
        """Store where 'sst' is null on hole_dates; 'other' is always populated."""
        variables = {"sst": 20.0, "other": 1.0}
        df = make_grid_df(dates, variables=variables)
        df = df.with_columns(
            pl.when(pl.col("time").is_in(hole_dates))
            .then(None)
            .otherwise(pl.col("sst"))
            .alias("sst")
        )
        store = _store(tmp_path)
        store.add_data(df)
        return store

    def test_detects_earliest_hole_in_partition(self, tmp_path):
        dates = [date(2021, 1, d) for d in range(1, 6)]
        store = self._store_with_holes(
            tmp_path, dates, [date(2021, 1, 3), date(2021, 1, 4)]
        )
        out = store.get_var_backfill_start(["sst", "other"])
        assert out == {"sst": pd.Timestamp("2021-01-03")}

    def test_hole_spanning_partitions_returns_oldest(self, tmp_path):
        # Jan fully covered; Feb and Mar have holes; the island in Mar must not
        # mask the Feb hole.
        dates = [
            date(2021, 1, 10),
            date(2021, 2, 10),
            date(2021, 3, 10),
            date(2021, 3, 20),
        ]
        store = self._store_with_holes(
            tmp_path, dates, [date(2021, 2, 10), date(2021, 3, 10)]
        )
        out = store.get_var_backfill_start(["sst"])
        assert out == {"sst": pd.Timestamp("2021-02-10")}

    def test_no_holes_returns_empty(self, tmp_path):
        store = self._store_with_holes(tmp_path, [date(2021, 1, 1)], [])
        assert store.get_var_backfill_start(["sst"]) == {}

    def test_hole_below_covered_partition_is_invisible(self, tmp_path):
        # Jan has a hole but Feb is fully covered → scanning stops at Feb and
        # the Jan gap is treated as sound (this is what bounds the scan and
        # keeps deep legitimate source gaps from being rescanned every run).
        dates = [date(2021, 1, 10), date(2021, 2, 10)]
        store = self._store_with_holes(tmp_path, dates, [date(2021, 1, 10)])
        assert store.get_var_backfill_start(["sst"]) == {}

    def test_not_before_floor_skips_old_gaps(self, tmp_path):
        # Holes in both Jan and Feb (no fully-covered partition above them);
        # a floor at Feb 1 must hide the Jan gap.
        dates = [date(2021, 1, 10), date(2021, 2, 10)]
        store = self._store_with_holes(
            tmp_path, dates, [date(2021, 1, 10), date(2021, 2, 10)]
        )
        assert store.get_var_backfill_start(
            ["sst"], not_before={"sst": datetime(2021, 2, 1)}
        ) == {"sst": pd.Timestamp("2021-02-10")}
        # Without the floor the scan walks back to the Jan gap.
        assert store.get_var_backfill_start(["sst"]) == {
            "sst": pd.Timestamp("2021-01-10")
        }

    def test_empty_store_returns_empty(self, tmp_path):
        store = _store(tmp_path)
        assert store.get_var_backfill_start(["sst"]) == {}


class TestGetVarCoverage:
    def test_empty_store_returns_empty(self, tmp_path):
        store = _store(tmp_path)
        assert store.get_var_coverage() == {}

    def test_returns_per_column_range(self, tmp_path):
        store = _store(tmp_path)
        _write_one(store, [date(2021, 1, 1), date(2021, 1, 5)], variables={"sst": 20.0})
        cov = store.get_var_coverage()
        assert "sst" in cov
        assert cov["sst"].start.date() == date(2021, 1, 1)
        assert cov["sst"].end.date() == date(2021, 1, 5)

    def test_excludes_coordinate_columns(self, tmp_path):
        store = _store(tmp_path)
        _write_one(store, [date(2021, 1, 1)])
        cov = store.get_var_coverage()
        assert "time" not in cov
        assert "lon" not in cov
        assert "lat" not in cov

    def test_lagging_column_reports_its_own_end(self, tmp_path):
        """A column present only on earlier dates ends earlier than a fuller one."""
        store = _store(tmp_path)
        # sst on Jan 1-3
        _write_one(store, [date(2021, 1, 1), date(2021, 1, 3)], variables={"sst": 20.0})
        # chl only on Jan 1 (lags behind sst)
        store.add_data(make_grid_df([date(2021, 1, 1)], variables={"chl": 0.5}))

        cov = store.get_var_coverage()
        assert cov["sst"].end.date() == date(2021, 1, 3)
        assert cov["chl"].end.date() == date(2021, 1, 1)

    def test_column_filter_restricts_result(self, tmp_path):
        store = _store(tmp_path)
        _write_one(store, [date(2021, 1, 1)], variables={"sst": 20.0, "chl": 0.5})
        cov = store.get_var_coverage(columns=["sst"])
        assert set(cov.keys()) == {"sst"}

    def test_all_null_column_omitted(self, tmp_path):
        store = _store(tmp_path)
        _write_one(store, [date(2021, 1, 1)], variables={"sst": 20.0})
        # Add a fully-null column by merging a frame whose chl is null
        df = make_grid_df([date(2021, 1, 1)], variables={"chl": 0.5}).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("chl")
        )
        store.add_data(df)
        cov = store.get_var_coverage()
        assert "sst" in cov
        assert "chl" not in cov


class TestGetGeoextent:
    def test_raises_on_empty_store(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(RuntimeError, match="not initialized"):
            store.get_geoextent()

    def test_returns_bbox_after_write(self, tmp_path):
        store = _store(tmp_path)
        _write_one(store, [date(2021, 1, 1)])
        bbox = store.get_geoextent()
        assert bbox is not None
        assert bbox.xmin == pytest.approx(-10.0)
        assert bbox.xmax == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestGetSchema:
    def test_returns_schema_after_write(self, tmp_path, jan_df):
        store = _store(tmp_path)
        store.add_data(jan_df)
        schema = store.get_schema()
        assert "sst" in schema
        assert "time" in schema

    def test_schema_reflects_new_variable(self, tmp_path):
        store = _store(tmp_path)
        _write_one(store, [date(2021, 1, 1)])
        store.add_data(make_grid_df([date(2021, 1, 1)], variables={"chl": 0.5}))
        schema = store.get_schema()
        assert "chl" in schema
        assert "sst" in schema
