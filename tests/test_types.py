"""Tests for DateRange and BBox."""

import numpy as np
import pytest
import polars as pl
import pandas as pd
import xarray as xr
from datetime import date, datetime

from h2mare.types import DateRange, BBox, DownloadTask


# ---------------------------------------------------------------------------
# DateRange
# ---------------------------------------------------------------------------


class TestDateRange:
    def test_basic_construction(self):
        dr = DateRange("2020-01-01", "2020-12-31")
        assert dr.start == datetime(2020, 1, 1)
        assert dr.end == datetime(2020, 12, 31)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="start.*after end"):
            DateRange("2020-12-31", "2020-01-01")

    def test_same_day_allowed(self):
        dr = DateRange("2020-06-15", "2020-06-15")
        assert dr.start == dr.end

    def test_normalises_to_midnight(self):
        dr = DateRange(datetime(2020, 6, 15, 12, 30), datetime(2020, 6, 20, 8, 0))
        assert dr.start.hour == 0
        assert dr.end.hour == 0

    def test_accepts_date_objects(self):
        dr = DateRange(date(2020, 1, 1), date(2020, 3, 31))
        assert dr.start.year == 2020

    def test_overlaps_true(self):
        a = DateRange("2020-01-01", "2020-06-30")
        b = DateRange("2020-06-01", "2020-12-31")
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_overlaps_touching_boundary(self):
        a = DateRange("2020-01-01", "2020-06-30")
        b = DateRange("2020-06-30", "2020-12-31")
        assert a.overlaps(b)

    def test_no_overlap(self):
        a = DateRange("2020-01-01", "2020-03-31")
        b = DateRange("2020-07-01", "2020-12-31")
        assert not a.overlaps(b)

    def test_intersection(self):
        a = DateRange("2020-01-01", "2020-06-30")
        b = DateRange("2020-04-01", "2020-09-30")
        ix = a.intersection(b)
        assert ix is not None
        assert ix.start == datetime(2020, 4, 1)
        assert ix.end == datetime(2020, 6, 30)

    def test_intersection_none_when_no_overlap(self):
        a = DateRange("2020-01-01", "2020-03-31")
        b = DateRange("2020-07-01", "2020-09-30")
        assert a.intersection(b) is None

    def test_spans_multiple_years_true(self):
        assert DateRange("2019-12-01", "2020-01-31").spans_multiple_years()

    def test_spans_multiple_years_false(self):
        assert not DateRange("2020-03-01", "2020-11-30").spans_multiple_years()

    def test_to_label_date(self):
        dr = DateRange("2020-01-15", "2020-03-20")
        assert dr.to_label("date") == "2020-01-15-2020-03-20"

    def test_to_label_same_day(self):
        dr = DateRange("2020-06-01", "2020-06-01")
        assert dr.to_label("date") == "2020-06-01"

    def test_from_polars(self):
        df = pl.DataFrame({"time": [date(2020, 1, 1), date(2020, 6, 30)]}).with_columns(
            pl.col("time").cast(pl.Date)
        )
        dr = DateRange.from_polars(df, "time")
        assert dr.start.date() == date(2020, 1, 1)
        assert dr.end.date() == date(2020, 6, 30)

    def test_from_pandas(self):
        df = pd.DataFrame({"time": pd.to_datetime(["2020-01-01", "2020-12-31"])})
        dr = DateRange.from_pandas(df, "time")
        assert dr.start.year == 2020
        assert dr.end.month == 12

    def test_to_label_year(self):
        dr = DateRange("2020-01-01", "2020-12-31")
        assert dr.to_label("year") == "2020"

    def test_to_label_yearmonth_same_month(self):
        dr = DateRange("2020-03-01", "2020-03-31")
        assert dr.to_label("yearmonth") == "2020-03"

    def test_to_label_yearmonth_different_months(self):
        dr = DateRange("2020-03-01", "2020-05-31")
        assert dr.to_label("yearmonth") == "2020-03-2020-05"

    def test_from_dataset(self):
        times = pd.date_range("2020-01-01", periods=3, freq="D")
        ds = xr.Dataset(coords={"time": times})
        dr = DateRange.from_dataset(ds)
        assert dr.start.date() == date(2020, 1, 1)

    def test_from_dataset_missing_time_raises(self):
        ds = xr.Dataset(coords={"lat": [1.0, 2.0]})
        with pytest.raises(ValueError, match="time"):
            DateRange.from_dataset(ds)

    def test_from_polars_missing_col_raises(self):
        df = pl.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            DateRange.from_polars(df, "time")

    def test_from_polars_empty_col_raises(self):
        df = pl.DataFrame({"time": pl.Series([], dtype=pl.Date)})
        with pytest.raises(ValueError):
            DateRange.from_polars(df, "time")

    def test_from_polars_lazy(self):
        df = pl.LazyFrame({"time": [date(2020, 1, 1), date(2020, 6, 30)]}).with_columns(
            pl.col("time").cast(pl.Date)
        )
        dr = DateRange.from_polars_lazy(df, "time")
        assert dr.start.date() == date(2020, 1, 1)

    def test_from_pandas_missing_col_raises(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="not found"):
            DateRange.from_pandas(df, "time")

    def test_from_dataframe_lazy(self):
        df = pl.LazyFrame({"time": [date(2020, 1, 1), date(2020, 6, 30)]}).with_columns(
            pl.col("time").cast(pl.Date)
        )
        dr = DateRange.from_dataframe(df, "time")
        assert dr.start.year == 2020

    def test_from_dataframe_polars(self):
        df = pl.DataFrame({"time": [date(2020, 1, 1), date(2020, 6, 30)]}).with_columns(
            pl.col("time").cast(pl.Date)
        )
        dr = DateRange.from_dataframe(df, "time")
        assert dr.end.month == 6

    def test_from_dataframe_pandas(self):
        df = pd.DataFrame({"time": pd.to_datetime(["2020-01-01", "2020-06-30"])})
        dr = DateRange.from_dataframe(df, "time")
        assert dr.start.year == 2020

    def test_from_dataframe_unsupported_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            DateRange.from_dataframe({"time": [1, 2]}, "time")


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------


class TestBBox:
    def test_basic_construction(self):
        b = BBox(xmin=-10, ymin=30, xmax=20, ymax=50)
        assert b.area() == 30 * 20

    def test_invalid_x(self):
        with pytest.raises(ValueError, match="xmin"):
            BBox(xmin=10, ymin=30, xmax=5, ymax=50)

    def test_invalid_y(self):
        with pytest.raises(ValueError, match="ymin"):
            BBox(xmin=-10, ymin=50, xmax=20, ymax=30)

    def test_overlaps_true(self):
        a = BBox(-10, 30, 10, 50)
        b = BBox(0, 40, 20, 60)
        assert a.overlaps(b)

    def test_overlaps_touching_edge(self):
        a = BBox(-10, 30, 10, 50)
        b = BBox(10, 30, 20, 50)
        assert a.overlaps(b)

    def test_no_overlap(self):
        a = BBox(-10, 30, 0, 50)
        b = BBox(5, 30, 20, 50)
        assert not a.overlaps(b)

    def test_contains_true(self):
        b = BBox(-10, 30, 10, 50)
        assert b.contains(0, 40)

    def test_contains_false(self):
        b = BBox(-10, 30, 10, 50)
        assert not b.contains(15, 40)

    def test_to_label_mixed(self):
        b = BBox(xmin=-10, ymin=30, xmax=20, ymax=50)
        assert b.to_label() == "10W-20E-30N-50N"

    def test_from_tuple(self):
        b = BBox.from_tuple((-10, 30, 20, 50))
        assert b.xmin == -10

    def test_from_tuple_wrong_length(self):
        with pytest.raises(ValueError):
            BBox.from_tuple((-10, 30, 20))

    def test_from_polars(self):
        df = pl.DataFrame({"lon": [-10.0, 0.0, 20.0], "lat": [30.0, 40.0, 50.0]})
        b = BBox.from_polars(df, lon_col="lon", lat_col="lat")
        assert b.xmin == -10.0
        assert b.ymax == 50.0

    def test_from_pandas(self):
        df = pd.DataFrame({"lon": [-10.0, 20.0], "lat": [30.0, 50.0]})
        b = BBox.from_pandas(df, lon_col="lon", lat_col="lat")
        assert b.xmax == 20.0

    def test_repr(self):
        b = BBox(-10, 30, 20, 50)
        assert "BBox" in repr(b)
        assert "-10" in repr(b)

    def test_to_tuple(self):
        b = BBox(-10, 30, 20, 50)
        assert b.to_tuple() == (-10, 30, 20, 50)

    def test_area(self):
        b = BBox(0, 0, 10, 5)
        assert b.area() == 50.0

    def test_from_dataset(self):
        ds = xr.Dataset(coords={"lon": [-10.0, 0.0, 10.0], "lat": [30.0, 40.0, 50.0]})
        b = BBox.from_dataset(ds)
        assert b.xmin == -10.0
        assert b.ymax == 50.0

    def test_from_dataset_missing_coords_raises(self):
        ds = xr.Dataset(coords={"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="missing"):
            BBox.from_dataset(ds)

    def test_from_dataset_with_longitude_latitude_names(self):
        ds = xr.Dataset(coords={"longitude": [-10.0, 10.0], "latitude": [30.0, 50.0]})
        b = BBox.from_dataset(ds)
        assert b.xmin == -10.0

    def test_from_polars_lazy(self):
        df = pl.LazyFrame({"lon": [-10.0, 20.0], "lat": [30.0, 50.0]})
        b = BBox.from_polars_lazy(df, lon_col="lon", lat_col="lat")
        assert b.xmax == 20.0

    def test_from_polars_lazy_missing_col_raises(self):
        df = pl.LazyFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        with pytest.raises(ValueError):
            BBox.from_polars_lazy(df, lon_col="lon", lat_col="lat")

    def test_from_polars_missing_col_raises(self):
        df = pl.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(ValueError, match="not found"):
            BBox.from_polars(df, lon_col="lon", lat_col="lat")

    def test_from_polars_null_values_raises(self):
        df = pl.DataFrame(
            {
                "lon": pl.Series([None], dtype=pl.Float64),
                "lat": pl.Series([None], dtype=pl.Float64),
            }
        )
        with pytest.raises(ValueError):
            BBox.from_polars(df, lon_col="lon", lat_col="lat")

    def test_from_pandas_missing_col_raises(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(ValueError, match="not found"):
            BBox.from_pandas(df, lon_col="lon", lat_col="lat")

    def test_from_dataframe_pandas(self):
        df = pd.DataFrame({"lon": [-10.0, 20.0], "lat": [30.0, 50.0]})
        b = BBox.from_dataframe(df, lon_col="lon", lat_col="lat")
        assert b.xmin == -10.0

    def test_from_dataframe_polars_lazy(self):
        df = pl.LazyFrame({"lon": [-10.0, 20.0], "lat": [30.0, 50.0]})
        b = BBox.from_dataframe(df, lon_col="lon", lat_col="lat")
        assert b.ymax == 50.0

    def test_from_dataframe_unsupported_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            BBox.from_dataframe({"lon": [1.0]}, lon_col="lon", lat_col="lat")


class TestDownloadTask:
    def test_repr(self):
        dr = DateRange("2020-01-01", "2020-12-31")
        task = DownloadTask(dataset_id="cmems_sst", date_range=dr, dataset_type="rep")
        r = repr(task)
        assert "cmems_sst" in r
        assert "rep" in r
