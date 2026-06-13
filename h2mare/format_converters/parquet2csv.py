"""Export a date-filtered slice of the Parquet store to per-period CSV files."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl
from loguru import logger


def parquet2csv(
    parquet_root: Path | str,
    csv_root: Path | str,
    start_date: str,
    end_date: str,
    freq: Literal["day", "month", "year"] = "day",
    n_workers: int = 8,
) -> Path:
    """
    Convert Parquet data into per-day, per-month, or per-year CSV files.

    Args:
        parquet_root: Directory or file containing Parquet data.
        csv_root: Output directory for CSV files; year subdirectories are
            created automatically.
        start_date: Start of the export period.
        end_date: End of the export period.
        freq: Aggregation level — "day", "month", or "year".
        n_workers: Number of threads for parallel CSV writes.

    Returns:
        The ``csv_root`` directory that was written.

    Raises:
        ValueError: If ``freq`` is not "day", "month", or "year".
    """
    if freq not in ("day", "month", "year"):
        raise ValueError("freq must be 'day', 'month', or 'year'")

    parquet_root = Path(parquet_root)
    csv_root = Path(csv_root)
    start_dt = pd.Timestamp(start_date).to_pydatetime()
    end_dt = pd.Timestamp(end_date).to_pydatetime()

    fmt = {"day": "%Y-%m-%d", "month": "%Y-%m", "year": "%Y"}[freq]

    logger.info(f"Converting parquet to {freq} CSV files: {start_date} -> {end_date}")

    lf = pl.scan_parquet(parquet_root, missing_columns="insert")
    cols_to_drop = [c for c in ("year", "month") if c in lf.collect_schema().names()]

    lf = (
        lf.filter(pl.col("time").is_between(pl.lit(start_dt), pl.lit(end_dt)))
        .with_columns(pl.col("time").dt.truncate("1d"))
        .drop(cols_to_drop)
        .with_columns(pl.col(pl.Float64).cast(pl.Float32))
    )

    df = lf.collect(engine="streaming")
    # Drop rows whose every variable column is empty. The store backfills absent
    # columns with nulls and source gaps surface as NaN, so both count as empty;
    # fill_nan(None) folds NaN into null before the all-empty check. time/lat/lon
    # are coordinate columns and always present.
    df = df.filter(
        ~pl.all_horizontal(pl.exclude(["time", "lat", "lon"]).fill_nan(None).is_null())
    )
    df = df.with_columns(pl.col("time").dt.strftime(fmt).alias("date_key"))

    date_keys = df["date_key"].unique().to_list()

    def write_group(date_key: str) -> None:
        year_dir = csv_root / date_key[:4]
        year_dir.mkdir(parents=True, exist_ok=True)
        (
            df.filter(pl.col("date_key") == date_key)
            .drop("date_key")
            .with_columns(pl.col("time").dt.strftime("%Y-%m-%d"))
            .write_csv(year_dir / f"{date_key}.csv")
        )

    # list() drains the lazy map so exceptions raised inside write_group surface
    # here instead of being silently swallowed when the iterator is discarded.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(write_group, date_keys))

    logger.success(
        f"Finished exporting {len(date_keys)} {freq} CSV files to {csv_root}"
    )
    return csv_root
