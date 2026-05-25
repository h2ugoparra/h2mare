"""Tests for parquet_helpers utility functions."""

import pytest
import polars as pl
import numpy as np
from datetime import date

from h2mare.storage.parquet_helpers import (
    polars_float64_to_float32,
    aggregate_by_time,
    aggregate_by_space_time,
    _required_columns,
)
from h2mare.utils.plot import split_by_group, df_to_grid


# ---------------------------------------------------------------------------
# polars_float64_to_float32
# ---------------------------------------------------------------------------


class TestFloat64ToFloat32:
    def test_converts_float64(self):
        df = pl.DataFrame({"a": pl.Series([1.0, 2.0], dtype=pl.Float64)})
        result = polars_float64_to_float32(df)
        assert result["a"].dtype == pl.Float32

    def test_leaves_float32_unchanged(self):
        df = pl.DataFrame({"a": pl.Series([1.0], dtype=pl.Float32)})
        result = polars_float64_to_float32(df)
        assert result["a"].dtype == pl.Float32

    def test_leaves_int_unchanged(self):
        df = pl.DataFrame({"a": pl.Series([1, 2], dtype=pl.Int32)})
        result = polars_float64_to_float32(df)
        assert result["a"].dtype == pl.Int32

    def test_mixed_dtypes(self):
        df = pl.DataFrame(
            {
                "f64": pl.Series([1.0], dtype=pl.Float64),
                "f32": pl.Series([1.0], dtype=pl.Float32),
                "i32": pl.Series([1], dtype=pl.Int32),
            }
        )
        result = polars_float64_to_float32(df)
        assert result["f64"].dtype == pl.Float32
        assert result["f32"].dtype == pl.Float32
        assert result["i32"].dtype == pl.Int32


# ---------------------------------------------------------------------------
# _required_columns
# ---------------------------------------------------------------------------


class TestRequiredColumns:
    def test_passes_when_present(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        _required_columns(df, ["a", "b"])  # should not raise

    def test_raises_when_missing(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="b"):
            _required_columns(df, ["a", "b"])

    def test_single_string(self):
        df = pl.DataFrame({"a": [1]})
        _required_columns(df, "a")

    def test_lazyframe(self):
        lf = pl.DataFrame({"a": [1], "b": [2]}).lazy()
        _required_columns(lf, ["a", "b"])


# ---------------------------------------------------------------------------
# aggregate_by_time
# ---------------------------------------------------------------------------


class TestAggregateByTime:
    @pytest.fixture
    def lf(self):
        dates = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), "1d", eager=True)
        rng = np.random.default_rng(0)
        return pl.DataFrame(
            {
                "time": dates,
                "sst": rng.uniform(15, 25, len(dates)).astype("float32"),
            }
        ).lazy()

    def test_agg_by_month(self, lf):
        result = aggregate_by_time(lf, "sst", agg_by="month").collect()
        assert "time_agg" in result.columns
        assert "sst" in result.columns
        assert len(result) == 12  # one row per month

    def test_agg_by_year(self, lf):
        result = aggregate_by_time(lf, "sst", agg_by="year").collect()
        assert len(result) == 1

    def test_agg_by_season(self, lf):
        # Dec is placed into the *next* year's winter (meteorological convention),
        # so a full Jan-Dec year produces 5 year-season groups. Use Feb-Nov to get
        # exactly 4 complete seasons without any year-split winter.
        dates = pl.date_range(date(2020, 2, 1), date(2020, 11, 30), "1d", eager=True)
        rng = np.random.default_rng(0)
        lf_4s = pl.DataFrame(
            {
                "time": dates,
                "sst": rng.uniform(15, 25, len(dates)).astype("float32"),
            }
        ).lazy()
        result = aggregate_by_time(lf_4s, "sst", agg_by="season").collect()
        assert len(result) == 4

    def test_missing_column_raises(self, lf):
        with pytest.raises(ValueError, match="chl"):
            aggregate_by_time(lf, "chl", agg_by="month").collect()


# ---------------------------------------------------------------------------
# aggregate_by_space_time
# ---------------------------------------------------------------------------


class TestAggregateBySpaceTime:
    @pytest.fixture
    def lf(self):
        lons = [-10.0, 0.0]
        lats = [30.0, 40.0]
        dates = pl.date_range(date(2020, 1, 1), date(2020, 3, 31), "1d", eager=True)
        rng = np.random.default_rng(1)
        rows = [
            {"time": d, "lon": lon, "lat": lat, "sst": float(rng.uniform(15, 25))}
            for d in dates
            for lon in lons
            for lat in lats
        ]
        return pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Date)).lazy()

    def test_monthly_output_columns(self, lf):
        result = aggregate_by_space_time(lf, "sst", agg_by="month").collect()
        assert "month" in result.columns
        assert "lon" in result.columns
        assert "lat" in result.columns
        assert "sst" in result.columns

    def test_monthly_groups(self, lf):
        result = aggregate_by_space_time(lf, "sst", agg_by="month").collect()
        # Jan, Feb, Mar
        assert sorted(result["month"].unique().to_list()) == [1, 2, 3]

    def test_season_labels(self, lf):
        result = aggregate_by_space_time(lf, "sst", agg_by="season").collect()
        seasons = set(result["season"].to_list())
        assert seasons <= {"spring", "summer", "autumn", "winter"}


# ---------------------------------------------------------------------------
# split_by_group / df_to_grid
# ---------------------------------------------------------------------------


class TestSplitByGroup:
    def test_month_order(self):
        df = pl.DataFrame({"month": [3, 1, 2], "val": [0.3, 0.1, 0.2]})
        groups = split_by_group(df, "month")
        assert list(groups.keys()) == [1, 2, 3]

    def test_season_order(self):
        df = pl.DataFrame(
            {
                "season": ["winter", "spring", "autumn", "summer"],
                "val": [1.0, 2.0, 3.0, 4.0],
            }
        )
        groups = split_by_group(df, "season")
        assert list(groups.keys()) == ["spring", "summer", "autumn", "winter"]

    def test_returns_subframes(self):
        df = pl.DataFrame({"month": [1, 1, 2], "val": [1.0, 2.0, 3.0]})
        groups = split_by_group(df, "month")
        assert len(groups[1]) == 2
        assert len(groups[2]) == 1


class TestDfToGrid:
    def test_grid_shape(self):
        lons = [-10.0, 0.0, 10.0]
        lats = [30.0, 40.0]
        rows = [
            {"lon": lon, "lat": lat, "sst": float(lon + lat)}
            for lon in lons
            for lat in lats
        ]
        df = pl.DataFrame(rows)
        unique_lon, unique_lat, grid = df_to_grid(df, "sst")
        assert grid.shape == (len(lats), len(lons))

    def test_grid_values(self):
        df = pl.DataFrame(
            {
                "lon": [0.0, 1.0],
                "lat": [0.0, 0.0],
                "sst": [10.0, 20.0],
            }
        )
        _, _, grid = df_to_grid(df, "sst")
        assert grid[0, 0] == pytest.approx(10.0)
        assert grid[0, 1] == pytest.approx(20.0)
