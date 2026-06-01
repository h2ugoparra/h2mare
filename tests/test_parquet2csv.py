"""Tests for parquet2csv converter."""

from datetime import date, datetime

import polars as pl
import pytest
from conftest import make_grid_df

from h2mare.format_converters.parquet2csv import parquet2csv


@pytest.fixture
def parquet_root(tmp_path):
    df = make_grid_df(
        dates=[date(2021, 1, 1), date(2021, 1, 2), date(2021, 2, 1), date(2022, 3, 15)],
        variables={"sst": 20.0, "ssh": 0.5},
    ).with_columns(pl.col("time").cast(pl.Datetime))
    path = tmp_path / "parquet"
    path.mkdir()
    df.write_parquet(path / "data.parquet")
    return path


def test_invalid_freq_raises(parquet_root, tmp_path):
    with pytest.raises(ValueError, match="freq must be"):
        parquet2csv(
            parquet_root, tmp_path / "out", "2021-01-01", "2021-12-31", freq="hourly"
        )


def test_daily_creates_one_file_per_day(parquet_root, tmp_path):
    out = tmp_path / "out"
    parquet2csv(parquet_root, out, "2021-01-01", "2021-01-02", freq="daily")
    files = list(out.rglob("*.csv"))
    assert len(files) == 2
    names = {f.stem for f in files}
    assert names == {"2021-01-01", "2021-01-02"}


def test_monthly_creates_one_file_per_month(parquet_root, tmp_path):
    out = tmp_path / "out"
    parquet2csv(parquet_root, out, "2021-01-01", "2021-02-28", freq="monthly")
    files = list(out.rglob("*.csv"))
    assert len(files) == 2
    names = {f.stem for f in files}
    assert names == {"2021-01", "2021-02"}


def test_yearly_creates_one_file_per_year(parquet_root, tmp_path):
    out = tmp_path / "out"
    parquet2csv(parquet_root, out, "2021-01-01", "2022-12-31", freq="yearly")
    files = list(out.rglob("*.csv"))
    assert len(files) == 2
    names = {f.stem for f in files}
    assert names == {"2021", "2022"}


def test_files_organised_under_year_subdir(parquet_root, tmp_path):
    out = tmp_path / "out"
    parquet2csv(parquet_root, out, "2021-01-01", "2021-01-01", freq="daily")
    assert (out / "2021" / "2021-01-01.csv").exists()


def test_date_range_filters_data(parquet_root, tmp_path):
    out = tmp_path / "out"
    parquet2csv(parquet_root, out, "2021-01-01", "2021-01-01", freq="daily")
    files = list(out.rglob("*.csv"))
    assert len(files) == 1
    assert files[0].stem == "2021-01-01"


def test_year_month_partition_cols_absent(parquet_root, tmp_path):
    # Write a parquet with explicit year/month columns (as ParquetIndexer does)
    df = make_grid_df(
        dates=[date(2021, 1, 1)],
        variables={"sst": 20.0},
    ).with_columns(
        pl.col("time").cast(pl.Datetime),
        pl.lit(2021).alias("year"),
        pl.lit(1).alias("month"),
    )
    p = tmp_path / "with_partition"
    p.mkdir()
    df.write_parquet(p / "data.parquet")

    out = tmp_path / "out"
    parquet2csv(p, out, "2021-01-01", "2021-01-31", freq="daily")
    result = pl.read_csv(out / "2021" / "2021-01-01.csv")
    assert "year" not in result.columns
    assert "month" not in result.columns


def test_all_nan_rows_dropped(tmp_path):
    df = pl.DataFrame(
        {
            "time": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
            "lat": [30.0, 35.0],
            "lon": [-10.0, -5.0],
            "sst": [float("nan"), 20.0],
        }
    )
    p = tmp_path / "parquet"
    p.mkdir()
    df.write_parquet(p / "data.parquet")

    out = tmp_path / "out"
    parquet2csv(p, out, "2021-01-01", "2021-01-31", freq="daily")
    result = pl.read_csv(out / "2021" / "2021-01-02.csv")
    assert len(result) == 1
    assert (out / "2021" / "2021-01-01.csv").exists() is False


def test_csv_contains_expected_columns(parquet_root, tmp_path):
    out = tmp_path / "out"
    parquet2csv(parquet_root, out, "2021-01-01", "2021-01-01", freq="daily")
    result = pl.read_csv(out / "2021" / "2021-01-01.csv")
    assert set(result.columns) >= {"time", "lat", "lon", "sst", "ssh"}
    assert "date_key" not in result.columns


def test_float64_downcast_to_float32(tmp_path):
    df = pl.DataFrame(
        {
            "time": [datetime(2021, 1, 1)],
            "lat": [30.0],
            "lon": [-10.0],
            "sst": pl.Series([20.123456789], dtype=pl.Float64),
        }
    )
    p = tmp_path / "parquet"
    p.mkdir()
    df.write_parquet(p / "data.parquet")

    out = tmp_path / "out"
    parquet2csv(p, out, "2021-01-01", "2021-01-31", freq="daily")
    result = pl.read_csv(out / "2021" / "2021-01-01.csv")
    # float32 precision: ~7 significant digits
    assert abs(result["sst"][0] - 20.123456789) < 1e-4
