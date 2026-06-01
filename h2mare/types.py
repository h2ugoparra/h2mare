"""
Core data structures used throughout h2mare.

These are runtime objects (not configuration models) that represent
fundamental concepts like bounding boxes and date ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional, Sequence, cast

import pandas as pd
import polars as pl
import xarray as xr

DateLike = str | pd.Timestamp | datetime | date


def to_datetime(value) -> datetime:
    """Coerce date, pd.Timestamp, str, or datetime to stdlib datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    if hasattr(value, "to_pydatetime"):  # pd.Timestamp
        return value.to_pydatetime()
    raise TypeError(f"Cannot convert {type(value)} to datetime")


class TimeResolution(str, Enum):
    """Supported period granularity for data storage."""

    YEAR = "year"
    MONTH = "month"


@dataclass
class DateRange:
    """Represents a date range."""

    start: datetime
    end: datetime

    def __post_init__(self):
        """Coerce inputs to datetime and normalize to midnight."""
        self.start = to_datetime(self.start).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.end = to_datetime(self.end).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if self.start > self.end:
            raise ValueError(
                f"DateRange start ({self.start.date()}) must not be after end ({self.end.date()})"
            )

    def __repr__(self) -> str:
        return f"{self.start.date()} to {self.end.date()}"

    def overlaps(self, other: DateRange) -> bool:
        """Check if this range overlaps with another."""
        return self.start <= other.end and self.end >= other.start

    def intersection(self, other: DateRange) -> Optional[DateRange]:
        """Get intersection with another range."""
        if not self.overlaps(other):
            return None

        return DateRange(
            start=max(self.start, other.start),
            end=min(self.end, other.end),
        )

    def to_label(self, format: Literal["date", "year", "yearmonth"]) -> str:
        """
        Convert to label string.

        Args:
            format: - "date" shows date_range (YYYY-MM-DD-YYYY-MM-DD)
                    - "year" shows year (YYYY). For single year files, as convenioned for zarr files
                    -  "yearmonth" (YYYY-MM). For monthly files.

        Returns:
            Formatted date label
        """
        if format == "year":
            return str(self.start.year)

        elif format == "yearmonth":
            start_str = self.start.strftime("%Y-%m")
            end_str = self.end.strftime("%Y-%m")
            return start_str if start_str == end_str else f"{start_str}-{end_str}"

        else:  # "date"
            start_str = self.start.strftime("%Y-%m-%d")
            end_str = self.end.strftime("%Y-%m-%d")
            return start_str if start_str == end_str else f"{start_str}-{end_str}"

    def spans_multiple_years(self) -> bool:
        """Check if range spans multiple years."""
        return self.start.year != self.end.year

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> DateRange:
        """Extract date range from dataset time coordinate."""
        if "time" not in ds.coords:
            raise ValueError(
                f"Dataset missing 'time' coordinate. "
                f"Available: {list(ds.coords.keys())}"
            )

        time = ds["time"]

        start = pd.to_datetime(time.min().compute().item()).normalize()
        end = pd.to_datetime(time.max().compute().item()).normalize()
        return cls(start=start, end=end)

    @classmethod
    def from_polars(cls, df: pl.DataFrame, time_col: str) -> DateRange:
        """Create DateRange from min/max of a Polars DataFrame time column."""
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in Polars DataFrame.")
        start = df[time_col].min()
        end = df[time_col].max()
        if start is None or end is None:
            raise ValueError(f"Column '{time_col}' is empty or all null.")
        return cls(start=to_datetime(start), end=to_datetime(end))

    @classmethod
    def from_polars_lazy(cls, df: pl.LazyFrame, time_col: str = "time") -> DateRange:
        """Create DateRange from min/max of a Polars LazyFrame time column."""
        result = df.select(
            [pl.col(time_col).min().alias("start"), pl.col(time_col).max().alias("end")]
        ).collect(engine="streaming")
        return cls(
            start=to_datetime(result["start"][0]), end=to_datetime(result["end"][0])
        )

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, time_col: str = "time") -> DateRange:
        """Create DateRange from min/max of a Pandas DataFrame time column."""
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in Pandas DataFrame.")
        dates = df[time_col]
        return cls(start=dates.min(), end=dates.max())

    @classmethod
    def from_dataframe(cls, df, time_col: str = "time") -> DateRange:
        if isinstance(df, pl.LazyFrame):
            return cls.from_polars_lazy(df, time_col)
        elif isinstance(df, pl.DataFrame):
            return cls.from_polars(df, time_col)
        elif isinstance(df, pd.DataFrame):
            return cls.from_pandas(df, time_col)
        else:
            raise TypeError(f"Unsupported DataFrame type: {type(df)}")


@dataclass
class BBox:
    """
    Geographic bounding box.

    Used for spatial subsetting, catalog queries, and label generation.

    Attributes:
        xmin: Western longitude
        ymin: Southern latitude
        xmax: Eastern longitude
        ymax: Northern latitude

    Example:
        >>> bbox = BBox(xmin=-10, ymin=30, xmax=20, ymax=40)
        >>> bbox.to_tuple()
        (-10, 30, 20, 40)
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        """
        Validate bounding box format.

        Raises:
            ValueError: If xmin >= xmax
            ValueError: If ymin >= ymax
        """
        if self.xmin >= self.xmax:
            raise ValueError(f"Invalid bbox: xmin ({self.xmin}) >= xmax ({self.xmax})")

        if self.ymin >= self.ymax:
            raise ValueError(f"Invalid bbox: ymin ({self.ymin}) >= ymax ({self.ymax})")

    def __repr__(self) -> str:
        return f"BBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})"

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert to tuple (xmin, ymin, xmax, ymax)."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def to_label(self) -> str:
        """
        Convert to geographic label for filenames.

        Returns:
            String like "10W-20E-30N-40N"
        """
        xmin_str = f"{round(abs(self.xmin))}{'W' if self.xmin < 0 else 'E'}"
        xmax_str = f"{round(abs(self.xmax))}{'W' if self.xmax < 0 else 'E'}"
        ymin_str = f"{round(abs(self.ymin))}{'S' if self.ymin < 0 else 'N'}"
        ymax_str = f"{round(abs(self.ymax))}{'S' if self.ymax < 0 else 'N'}"

        return f"{xmin_str}-{xmax_str}-{ymin_str}-{ymax_str}"

    def contains(self, lon: float, lat: float) -> bool:
        """Check if point is within bounding box."""
        return self.xmin <= lon <= self.xmax and self.ymin <= lat <= self.ymax

    def overlaps(self, other: BBox) -> bool:
        """Check if this bbox overlaps with another."""
        return (
            self.xmin <= other.xmax
            and self.xmax >= other.xmin
            and self.ymin <= other.ymax
            and self.ymax >= other.ymin
        )

    def area(self) -> float:
        """Calculate area in square degrees."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    @classmethod
    def from_tuple(cls, bbox: Sequence[float]) -> BBox:
        """Create from tuple (xmin, ymin, xmax, ymax)."""
        if len(bbox) != 4:
            raise ValueError(f"BBox requires 4 values, got {len(bbox)}")
        xmin, ymin, xmax, ymax = bbox
        return cls(xmin, ymin, xmax, ymax)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> BBox:
        """Extract bounding box from dataset coordinates."""
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        lat_name = "lat" if "lat" in ds.coords else "latitude"

        if lon_name not in ds.coords or lat_name not in ds.coords:
            raise ValueError(
                f"Dataset missing coordinates. Available: {list(ds.coords.keys())}"
            )

        lon = ds[lon_name]
        lat = ds[lat_name]

        return cls(
            xmin=float(lon.min().compute().item()),
            xmax=float(lon.max().compute().item()),
            ymin=float(lat.min().compute().item()),
            ymax=float(lat.max().compute().item()),
        )

    @classmethod
    def from_polars_lazy(cls, df: pl.LazyFrame, lon_col: str, lat_col: str) -> BBox:
        df_cols = df.collect_schema().names()
        if lon_col not in df_cols or lat_col not in df_cols:
            raise ValueError(f"{lon_col} or {lat_col} not found in Polars LazyFrame")
        result = df.select(
            [
                pl.col(lon_col).min().alias("min_lon"),
                pl.col(lat_col).min().alias("min_lat"),
                pl.col(lon_col).max().alias("max_lon"),
                pl.col(lat_col).max().alias("max_lat"),
            ]
        ).collect(engine="streaming")

        return cls(
            xmin=float(result["min_lon"][0]),
            ymin=float(result["min_lat"][0]),
            xmax=float(result["max_lon"][0]),
            ymax=float(result["max_lat"][0]),
        )

    @classmethod
    def from_polars(cls, df: pl.DataFrame, lon_col: str, lat_col: str) -> BBox:
        if lon_col not in df.columns or lat_col not in df.columns:
            raise ValueError(
                f"'{lon_col}' or '{lat_col}' not found in Polars DataFrame."
            )
        min_lon = df[lon_col].min()
        max_lon = df[lon_col].max()
        min_lat = df[lat_col].min()
        max_lat = df[lat_col].max()

        if any(v is None for v in [min_lon, max_lon, min_lat, max_lat]):
            raise ValueError(f"'{lon_col}' or '{lat_col}' is empty or all null.")

        return cls(
            xmin=float(cast(float, min_lon)),
            ymin=float(cast(float, min_lat)),
            xmax=float(cast(float, max_lon)),
            ymax=float(cast(float, max_lat)),
        )

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, lon_col: str, lat_col: str) -> BBox:
        if lon_col not in df.columns or lat_col not in df.columns:
            raise ValueError(
                f"'{lon_col}' or '{lat_col}' not found in Pandas DataFrame."
            )
        return cls(
            xmin=float(df[lon_col].min()),
            ymin=float(df[lat_col].min()),
            xmax=float(df[lon_col].max()),
            ymax=float(df[lat_col].max()),
        )

    @classmethod
    def from_dataframe(cls, df, lon_col: str, lat_col: str) -> BBox:
        if isinstance(df, pl.LazyFrame):
            return cls.from_polars_lazy(df, lon_col, lat_col)
        elif isinstance(df, pl.DataFrame):
            return cls.from_polars(df, lon_col, lat_col)
        elif isinstance(df, pd.DataFrame):
            return cls.from_pandas(df, lon_col, lat_col)
        else:
            raise TypeError(f"Unsupported DataFrame type: {type(df)}")


@dataclass
class DownloadTask:
    """Represents a single download task."""

    dataset_id: str
    date_range: DateRange
    dataset_type: Literal["rep", "nrt"]  # reprocessed or near-real-time

    def __repr__(self) -> str:
        return (
            f"DownloadTask(dataset={self.dataset_id}, "
            f"type={self.dataset_type}, {self.date_range})"
        )


@dataclass
class FTPDownloadTask:
    """Represents a single FTP file download task (used by AVISODownloader)."""

    filepath: str
    source: Literal["rep", "nrt"]  # reprocessed or near-real-time

    def __repr__(self) -> str:
        return f"FTPDownloadTask(filepath={self.filepath}, source={self.source})"
