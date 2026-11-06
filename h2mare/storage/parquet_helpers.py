"""Utilitites for parquet_indexer"""

from __future__ import annotations

from typing import Literal

import polars as pl

from h2mare.validators import validate_columns

SEASON_INDEX = {
    "spring": 0,
    "summer": 1,
    "autumn": 2,
    "winter": 3,
}


def _season_key(time_col: str) -> pl.Expr:
    """
    Chronological ``YYYYS`` key of each row's meteorological season.

    ``S`` is 1 spring (MAM), 2 summer (JJA), 3 autumn (SON), 4 winter (DJF), and
    ``YYYY`` is the year the season *starts* in — so a winter is labelled by its
    December, and January/February belong to the previous year's winter. That
    keeps the key sorting in time order (``20213`` autumn 2021 < ``20214``
    winter 2021/22 < ``20221`` spring 2022) and lets a plot anchor each season
    at its first month in that year (winter → ``YYYY-12-01``).
    """
    month = pl.col(time_col).dt.month()
    season = (
        pl.when(month.is_in([12, 1, 2]))
        .then(4)
        .when(month.is_in([3, 4, 5]))
        .then(1)
        .when(month.is_in([6, 7, 8]))
        .then(2)
        .otherwise(3)
    )
    year = (
        pl.when(month.is_in([1, 2]))
        .then(pl.col(time_col).dt.year() - 1)
        .otherwise(pl.col(time_col).dt.year())
    )
    return year * 10 + season


def polars_float64_to_float32(df: pl.DataFrame) -> pl.DataFrame:
    """Converts dtype column from float64 to float32"""

    def _convert(col: pl.Series) -> pl.Series:
        if col.dtype == pl.Float64:
            return col.cast(pl.Float32)
        return col

    return df.with_columns([_convert(df[col]) for col in df.columns])


def aggregate_by_space_time(
    lf: pl.LazyFrame,
    vars_name: str | list[str],
    *,
    agg_by: Literal["month", "season", "month_year"],
    time_col: str = "time",
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> pl.LazyFrame:
    """
    Aggregate multi-year gridded data (lat, lon) into monthly or seasonal averages.

    Args:
        lf (pl.LazyFrame): Data for agg.
        vars_name (str | list[str]): Variables(s) name for agg.
        time_col, lon_col, lat_col (str): time, longitude and latitude column names. Defaults to 'time', 'lon', 'lat', respectively.
        agg_by (Literal['month', 'season']): Agg by month or season.

    Notes:
        Seasons Meteorological and for the Northen Hemisphere.

    Returns:
        pl.LazyFrame: Output grouped by lat, lon, time_key ('month' or 'season') and vars_name.
    """

    def order_seasons(
        lf: pl.LazyFrame, season_index: dict = SEASON_INDEX
    ) -> pl.LazyFrame:
        """Order seasons by season_index (spring, summer, autumn, winter) and not alphabetically."""
        lf = lf.with_columns(
            pl.col("season")
            .replace(season_index)
            .cast(pl.UInt8)
            .alias("_season_order"),
        )
        lf = lf.sort("_season_order").drop("_season_order")
        return lf

    vars_name = [vars_name] if isinstance(vars_name, str) else vars_name

    validate_columns(lf, [*vars_name, time_col, lon_col, lat_col])

    if agg_by == "month":
        lf = lf.with_columns(pl.col(time_col).dt.month().alias(agg_by))

    elif agg_by == "month_year":
        lf = lf.with_columns(pl.col(time_col).dt.truncate("1mo").alias(agg_by))

    elif agg_by == "season":
        lf = lf.with_columns(
            pl.when(pl.col(time_col).dt.month().is_in([12, 1, 2]))
            .then(pl.lit("winter"))
            .when(pl.col(time_col).dt.month().is_in([3, 4, 5]))
            .then(pl.lit("spring"))
            .when(pl.col(time_col).dt.month().is_in([6, 7, 8]))
            .then(pl.lit("summer"))
            .otherwise(pl.lit("autumn"))
            .alias(agg_by)
        )

    lf_out = (
        lf.group_by([lat_col, lon_col, agg_by])
        .agg(*[pl.col(v).mean() for v in vars_name], pl.len().alias("n"))
        .sort([agg_by, lat_col, lon_col])
    )

    if agg_by == "season":
        lf_out = order_seasons(lf_out)

    return lf_out


def aggregate_by_time(
    lf: pl.LazyFrame,
    vars_name: str | list[str],
    *,
    agg_by: Literal["day", "week", "month", "season", "year"],
    time_col: str = "time",
) -> pl.LazyFrame:
    """
    Aggregates variable(s) by daily, weekly, monthly, seasonal, or yearly resolution. This is for time series plots.
    With multi-year data, 'monthly'/'seasonal' agg does not return unique months/seasons. It keeps yearly data.

    Args:
        lf: Lazyframe for aggregation.
        var_name: Variable(s) name for aggregation.
        agg_by: Time range for aggregation. Options are:
            - day, week, month, season, year
        time_col: Time column name. Defaults to 'time'.

    Raises:
        ValueError: if agg_by is non of the options

    Returns:
        pl.Lazyframe: Dataframe with columns 'time_agg' and '{var_name}'.
    """
    vars_list = [vars_name] if isinstance(vars_name, str) else vars_name

    validate_columns(lf, [*vars_list, time_col])

    # Build aggregation expressions once
    agg_exprs = [pl.col(v).mean() for v in vars_list]

    if agg_by == "day":
        key = time_col
        lf2 = lf

    elif agg_by == "week":
        key = "week_start"
        lf2 = lf.with_columns(pl.col(time_col).dt.truncate("1w").alias(key))
    elif agg_by == "month":
        key = "month_start"
        lf2 = lf.with_columns(pl.col(time_col).dt.truncate("1mo").alias(key))
    elif agg_by == "year":
        key = "year_start"
        lf2 = lf.with_columns(pl.col(time_col).dt.truncate("1y").alias(key))
    elif agg_by == "season":
        key = "season"
        lf2 = lf.with_columns(season=_season_key(time_col))
    else:
        raise ValueError(f"Unsupported aggregation: {agg_by}")

    result = lf2.group_by(key).agg(agg_exprs).rename({key: "time_agg"}).sort("time_agg")

    return result


def aggregate_by_time_stats(
    lf: pl.LazyFrame,
    var_name: str,
    *,
    agg_by: Literal["day", "week", "month", "season", "year"],
    time_col: str = "time",
) -> pl.LazyFrame:
    """
    Aggregates a variable by time, computing mean, std, min, and max per bucket.

    Args:
        lf: LazyFrame for aggregation.
        var_name: Variable name for aggregation.
        agg_by: Time range for aggregation. Options are: day, week, month, season, year.
        time_col: Time column name. Defaults to 'time'.

    Returns:
        pl.LazyFrame: Columns 'time_agg', '{var}_mean', '{var}_std', '{var}_min', '{var}_max'.
    """
    validate_columns(lf, [var_name, time_col])

    agg_exprs = [
        pl.col(var_name).mean().alias(f"{var_name}_mean"),
        pl.col(var_name).std(ddof=1).alias(f"{var_name}_std"),
        pl.col(var_name).min().alias(f"{var_name}_min"),
        pl.col(var_name).max().alias(f"{var_name}_max"),
    ]

    if agg_by == "day":
        key = time_col
        lf2 = lf

    elif agg_by == "week":
        key = "week_start"
        lf2 = lf.with_columns(pl.col(time_col).dt.truncate("1w").alias(key))
    elif agg_by == "month":
        key = "month_start"
        lf2 = lf.with_columns(pl.col(time_col).dt.truncate("1mo").alias(key))
    elif agg_by == "year":
        key = "year_start"
        lf2 = lf.with_columns(pl.col(time_col).dt.truncate("1y").alias(key))
    elif agg_by == "season":
        key = "season"
        lf2 = lf.with_columns(season=_season_key(time_col))
    else:
        raise ValueError(f"Unsupported aggregation: {agg_by}")

    result = lf2.group_by(key).agg(agg_exprs).rename({key: "time_agg"}).sort("time_agg")

    return result
