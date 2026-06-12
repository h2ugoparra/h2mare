"""Integration tests for ParquetIndexer."""

from datetime import date

import polars as pl
import pytest
from conftest import make_grid_df

from h2mare.storage.parquet_indexer import ParquetIndexer

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_empty_dir_creates_directory(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir)
        assert parquet_dir.exists()
        assert idx.physical_schema is None
        assert idx.physical_cols == set()

    def test_repr_empty(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir)
        assert repr(idx) == ""  # no crash, returns empty string

    def test_wrong_col_names_raises(self, parquet_dir, jan_df):
        import polars.exceptions

        idx = ParquetIndexer(parquet_dir)
        idx.add_data(jan_df)
        # polars raises ColumnNotFoundError when scanning with a non-existent
        # time column before the ValueError check is reached
        with pytest.raises((ValueError, polars.exceptions.ColumnNotFoundError)):
            ParquetIndexer(parquet_dir, time_col="datetime")


# ---------------------------------------------------------------------------
# First write
# ---------------------------------------------------------------------------


class TestFirstWrite:
    def test_schema_initialised(self, loaded_indexer):
        schema = loaded_indexer.get_schema()
        assert "time" in schema
        assert "lon" in schema
        assert "lat" in schema
        assert "sst" in schema

    def test_physical_cols_populated(self, loaded_indexer):
        assert "sst" in loaded_indexer.physical_cols

    def test_time_coverage(self, loaded_indexer):
        cov = loaded_indexer.get_time_coverage()
        assert cov.start.date() == date(2020, 1, 1)
        assert cov.end.date() == date(2020, 1, 3)

    def test_geoextent(self, loaded_indexer):
        bbox = loaded_indexer.get_geoextent()
        assert bbox.xmin == pytest.approx(-10.0)
        assert bbox.xmax == pytest.approx(0.0)
        assert bbox.ymin == pytest.approx(30.0)
        assert bbox.ymax == pytest.approx(40.0)

    def test_repr_has_path(self, loaded_indexer):
        r = repr(loaded_indexer)
        assert "ParquetIndexer" in r
        assert "2020-01-01" in r


# ---------------------------------------------------------------------------
# scan / load
# ---------------------------------------------------------------------------


class TestScan:
    def test_load_all(self, loaded_indexer, jan_df):
        df = loaded_indexer.load()
        assert len(df) == len(jan_df)

    def test_scan_date_range(self, loaded_indexer):
        df = loaded_indexer.load(dates=("2020-01-01", "2020-01-02"))
        assert df["time"].max() <= date(2020, 1, 2)

    def test_scan_date_list(self, loaded_indexer):
        df = loaded_indexer.load(dates=["2020-01-01"])
        assert set(df["time"].cast(pl.Utf8).to_list()) == {"2020-01-01"}

    def test_scan_bbox(self, loaded_indexer):
        df = loaded_indexer.load(bbox=(-10, 30, -5, 35))
        assert df["lon"].max() <= -5.0
        assert df["lat"].max() <= 35.0

    def test_scan_columns(self, loaded_indexer):
        df = loaded_indexer.load(columns=["sst"])
        assert "sst" in df.columns
        # time/lon/lat always included
        assert "time" in df.columns

    def test_scan_before_data_raises(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir)
        with pytest.raises(RuntimeError):
            idx.scan(columns=["sst"]).collect()


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------


class TestOverlapResolution:
    def test_new_variable_added(self, parquet_dir, jan_df):
        """Adding a second variable to overlapping dates merges correctly."""
        idx = ParquetIndexer(parquet_dir)
        idx.add_data(jan_df)

        # Second write: same dates + new column chl
        jan_chl = jan_df.with_columns(pl.lit(0.5).alias("chl"))
        idx.add_data(jan_chl)

        df = idx.load()
        assert "chl" in df.columns
        assert "sst" in df.columns

    def test_new_timestamps_not_dropped(self, parquet_dir):
        """
        Regression: joining with how='left' dropped new timestamps in the same
        partition. Verify Jan 4 is kept after overlap-resolve with Jan 1-4.
        """
        idx = ParquetIndexer(parquet_dir)

        batch1 = make_grid_df([date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)])
        idx.add_data(batch1)

        # batch2 overlaps (Jan 2-3) and extends (Jan 4) — same month partition
        batch2 = make_grid_df(
            [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)],
            variables={"chl": 0.5},
        )
        idx.add_data(batch2)

        df = idx.load()
        dates_stored = set(df["time"].cast(pl.Utf8).to_list())
        assert "2020-01-04" in dates_stored, "Jan 4 was dropped by left-join bug"

    def test_non_overlapping_append(self, parquet_dir, jan_df):
        """Data from a different month should be appended, not merged."""
        idx = ParquetIndexer(parquet_dir)
        idx.add_data(jan_df)

        feb_df = make_grid_df(
            [date(2020, 2, 1), date(2020, 2, 2)],
        )
        idx.add_data(feb_df)

        df = idx.load()
        months = set(df["time"].dt.month().to_list())
        assert months == {1, 2}


# ---------------------------------------------------------------------------
# Plot accessor caching
# ---------------------------------------------------------------------------


class TestPlotCache:
    def test_cached_property_same_instance(self, loaded_indexer):
        assert loaded_indexer.plot is loaded_indexer.plot

    def test_cache_populated_on_first_call(self, loaded_indexer):
        key = ("sst", "month", None, None)
        assert key not in loaded_indexer.plot._cache
        loaded_indexer.plot._get_agg_df("sst", "month", None, None)
        assert key in loaded_indexer.plot._cache

    def test_cache_reused_on_second_call(self, loaded_indexer):
        df1 = loaded_indexer.plot._get_agg_df("sst", "month", None, None)
        df2 = loaded_indexer.plot._get_agg_df("sst", "month", None, None)
        assert df1 is df2  # same object, not recomputed

    def test_cache_cleared_after_add_data(self, loaded_indexer):
        loaded_indexer.plot._get_agg_df("sst", "month", None, None)
        assert len(loaded_indexer.plot._cache) == 1

        extra = make_grid_df([date(2020, 2, 1)])
        loaded_indexer.add_data(extra)

        assert len(loaded_indexer.plot._cache) == 0

    def test_clear_cache_explicit(self, loaded_indexer):
        loaded_indexer.plot._get_agg_df("sst", "month", None, None)
        loaded_indexer.plot.clear_cache()
        assert len(loaded_indexer.plot._cache) == 0

    def test_different_args_cached_separately(self, loaded_indexer):
        loaded_indexer.plot._get_agg_df("sst", "month", None, None)
        loaded_indexer.plot._get_agg_df("sst", "season", None, None)
        assert len(loaded_indexer.plot._cache) == 2


# ---------------------------------------------------------------------------
# Configurable partition_by
# ---------------------------------------------------------------------------


class TestPartitionBy:
    def test_default_partition_attributes(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir)
        assert idx._partition_by == ["year", "month"]
        assert idx.partition_cols == {"year", "month"}

    def test_year_month_day_dirs(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir, partition_by=["year", "month", "day"])
        df = make_grid_df([date(2021, 6, 15), date(2021, 6, 16)])
        idx.add_data(df)
        assert (parquet_dir / "year=2021" / "month=6" / "day=15").exists()
        assert (parquet_dir / "year=2021" / "month=6" / "day=16").exists()

    def test_year_only_dirs(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir, partition_by=["year"])
        df = make_grid_df([date(2021, 6, 15), date(2021, 7, 1)])
        idx.add_data(df)
        assert (parquet_dir / "year=2021").exists()
        assert not any(
            (parquet_dir / "year=2021").iterdir().__next__().name.startswith("month=")
            for _ in [None]
        )

    def test_custom_nontemporal_col_dirs(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir, partition_by=["year", "region"])
        df = make_grid_df([date(2021, 6, 15)]).with_columns(
            pl.lit("atlantic").alias("region")
        )
        idx.add_data(df)
        assert (parquet_dir / "year=2021" / "region=atlantic").exists()

    def test_missing_custom_col_raises(self, parquet_dir, jan_df):
        idx = ParquetIndexer(parquet_dir, partition_by=["year", "region"])
        with pytest.raises(ValueError, match="region"):
            idx.add_data(jan_df)

    def test_roundtrip_year_month_day(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir, partition_by=["year", "month", "day"])
        df = make_grid_df([date(2021, 6, 15), date(2021, 6, 16)])
        idx.add_data(df)
        loaded = idx.load()
        assert len(loaded) == len(df)
        assert set(loaded["time"].cast(pl.Utf8).to_list()) == {
            "2021-06-15",
            "2021-06-16",
        }

    def test_roundtrip_custom_col(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir, partition_by=["year", "region"])
        df = make_grid_df([date(2021, 6, 15), date(2021, 7, 1)]).with_columns(
            pl.lit("atlantic").alias("region")
        )
        idx.add_data(df)
        assert len(idx.load()) == len(df)

    def test_time_coverage_year_only(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir, partition_by=["year"])
        df = make_grid_df([date(2021, 3, 1), date(2022, 9, 1)])
        idx.add_data(df)
        cov = idx.get_time_coverage()
        assert cov.start.date() == date(2021, 3, 1)
        assert cov.end.date() == date(2022, 9, 1)

    def test_time_coverage_no_temporal_partition(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir, partition_by=["region"])
        df = make_grid_df([date(2021, 3, 1), date(2022, 9, 1)]).with_columns(
            pl.lit("atlantic").alias("region")
        )
        idx.add_data(df)
        cov = idx.get_time_coverage()
        assert cov.start.date() == date(2021, 3, 1)
        assert cov.end.date() == date(2022, 9, 1)

    def test_resolve_files_year_only_fast_path(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir, partition_by=["year"])
        df = make_grid_df([date(2021, 6, 15), date(2022, 1, 1)])
        idx.add_data(df)
        files = idx._resolve_files(("2021-01-01", "2021-12-31"))
        assert files
        assert all("year=2021" in str(f) for f in files)
        assert not any("year=2022" in str(f) for f in files)

    def test_resolve_files_no_temporal_returns_all(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir, partition_by=["region"])
        df = make_grid_df([date(2021, 6, 15), date(2022, 1, 1)]).with_columns(
            pl.lit("atlantic").alias("region")
        )
        idx.add_data(df)
        all_files = idx._resolve_files(None)
        range_files = idx._resolve_files(("2021-01-01", "2021-12-31"))
        assert set(range_files) == set(all_files)

    def test_resolve_files_year_month_range_selects_partitions(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir)
        df = make_grid_df([date(2021, 5, 1), date(2021, 6, 15), date(2021, 8, 1)])
        idx.add_data(df)
        files = idx._resolve_files(("2021-06-01", "2021-06-30"))
        assert files
        assert all("month=6" in str(f) for f in files)

    def test_resolve_files_date_list_selects_partitions(self, parquet_dir):
        idx = ParquetIndexer(parquet_dir)
        df = make_grid_df([date(2021, 5, 1), date(2021, 6, 15)])
        idx.add_data(df)
        files = idx._resolve_files(["2021-06-15"])
        assert files
        assert all("month=6" in str(f) for f in files)

    def test_overlap_resolution_custom_partition(self, parquet_dir):
        """Overlap resolution with a custom partition col merges correctly."""
        idx = ParquetIndexer(parquet_dir, partition_by=["year", "region"])
        df1 = make_grid_df([date(2021, 6, 15)]).with_columns(
            pl.lit("atlantic").alias("region")
        )
        idx.add_data(df1)

        df2 = df1.with_columns(pl.lit(0.5).alias("chl"))
        idx.add_data(df2)

        loaded = idx.load()
        assert "chl" in loaded.columns
        assert "sst" in loaded.columns
