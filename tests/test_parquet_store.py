"""Unit tests for ParquetStore — write-path and overlap-resolution logic."""

from datetime import date

import polars as pl
import pytest

from h2mare.storage.parquet_store import ParquetStore
from conftest import make_grid_df


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

    def test_partition_glob_format(self, tmp_path):
        store = _store(tmp_path)
        g = store._partition_glob()
        assert "year=*" in g
        assert "month=*" in g
        assert g.endswith("*.parquet")

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
            store._prepare_df(store._resolve_time_col(
                make_grid_df([date(2021, 6, 1)])
            )),
            include_partitions=False,
        )
        store.atomic_partition_write(df_physical, (2021, 6))
        assert not tmp.exists()

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

    def test_raises_on_no_spatial_overlap(self, tmp_path):
        store = self._setup(tmp_path, [date(2021, 1, 1)],
                            variables={"sst": 5.0})
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
