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

#: Which store a var_key is read from. ``native`` is its own per-variable Zarr,
#: ``compiled`` is the h2ds every var_key is merged into, and ``auto`` picks per
#: var_key. Lives here rather than beside its users because both the extraction
#: side (``processing.extractor``) and the read side
#: (``storage.var_routing``) name it, and neither may import the other.
ReadFrom = Literal["auto", "native", "compiled"]

#: How a variable is put on the compile base grid. ``auto`` compares the native
#: and target resolutions and picks ``linear`` or ``conservative``; the others
#: pin the choice. Lives here rather than beside the regridder because the
#: config model (``models``) validates it too, and must not pull in the
#: regridder's dependencies to do so.
RegridMethod = Literal["auto", "linear", "nearest", "conservative"]

#: Where a grid's values sit relative to whole degrees. ``center`` puts them at
#: cell centres (``xmin + (k + 0.5)·step``), ``node`` on the step's own
#: multiples (``xmin + k·step``). Independent of the step: both are valid grids
#: at any resolution. Which one a store wants depends on its sources — see
#: "Regridding" in docs/configuration.md.
GridRegistration = Literal["center", "node"]


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


class FilePeriod(str, Enum):
    """
    How much time one Zarr file on disk covers — one per year, or one per month.

    A property of the *storage layout*, not of the data: it says nothing about
    how far apart the steps inside a file are. That is
    :class:`h2mare.models.TimeStep`, and the two are independent — an hourly
    store written one file per year is ``TimeStep.HOURLY`` and
    ``FilePeriod.YEAR`` at the same time.

    Named ``TimeResolution`` until it acquired an hourly sibling to be confused
    with; "resolution" reads as the cadence of the data, which this is not.
    """

    YEAR = "year"
    MONTH = "month"


#: Deprecated alias for :class:`FilePeriod`. Nothing in h2mare uses it — it is
#: here so an import in a downstream repo does not break on the rename. Safe to
#: delete once those have caught up.
TimeResolution = FilePeriod


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
        # NaT reaches here intact: it is a datetime subclass, so to_datetime
        # passes it through and .replace() returns NaT again. Every comparison
        # against it is False, so the ordering check below waves it through and
        # a range of NaT..NaT travels on as if it named real dates. Rejecting it
        # at the one point every construction path goes through also covers the
        # from_* classmethods — from_pandas has no emptiness check of its own,
        # so an empty frame reaches here as NaT to NaT.
        if pd.isna(self.start) or pd.isna(self.end):
            raise ValueError(
                f"DateRange bounds must be real dates, got start={self.start!r}, "
                f"end={self.end!r} — usually an empty or all-null time column."
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
        start, end = result["start"][0], result["end"][0]
        # Same guard the eager from_polars has. Without it an empty frame fell
        # through to to_datetime(None) and raised "Cannot convert <class
        # 'NoneType'> to datetime", which does not say which column was empty.
        if start is None or end is None:
            raise ValueError(f"Column '{time_col}' is empty or all null.")
        return cls(start=to_datetime(start), end=to_datetime(end))

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, time_col: str = "time") -> DateRange:
        """Create DateRange from min/max of a Pandas DataFrame time column."""
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in Pandas DataFrame.")
        dates = df[time_col]
        start, end = dates.min(), dates.max()
        # Parity with from_polars. min()/max() on an empty or all-null column
        # give NaT, which the constructor rejects — but only by value, so it
        # cannot name the column the caller actually passed.
        if pd.isna(start) or pd.isna(end):
            raise ValueError(f"Column '{time_col}' is empty or all null.")
        return cls(start=start, end=end)

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
            ValueError: If any bound is NaN
            ValueError: If xmin >= xmax
            ValueError: If ymin >= ymax
        """
        # Checked before the ordering rules, which cannot see it: every
        # comparison against NaN is False, so `xmin >= xmax` passes and an
        # all-NaN box is accepted as valid. from_pandas produced exactly that
        # for an empty frame — min()/max() give NaN and float(NaN) is NaN —
        # while the polars twins already raised on the same input.
        if any(v != v for v in (self.xmin, self.ymin, self.xmax, self.ymax)):
            raise ValueError(
                f"BBox bounds must be real numbers, got "
                f"({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}) — "
                "usually an empty or all-null lon/lat column."
            )

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

        bounds = [
            result["min_lon"][0],
            result["min_lat"][0],
            result["max_lon"][0],
            result["max_lat"][0],
        ]
        # Same guard the eager from_polars has — an empty frame otherwise
        # reached float(None) and raised a TypeError naming neither column.
        if any(v is None for v in bounds):
            raise ValueError(f"'{lon_col}' or '{lat_col}' is empty or all null.")
        xmin, ymin, xmax, ymax = (float(cast(float, v)) for v in bounds)
        return cls(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

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
        bounds = (
            df[lon_col].min(),
            df[lat_col].min(),
            df[lon_col].max(),
            df[lat_col].max(),
        )
        # Parity with from_polars: an empty frame gives NaN throughout, which
        # the constructor now rejects — this names the column while doing it.
        if any(pd.isna(v) for v in bounds):
            raise ValueError(f"'{lon_col}' or '{lat_col}' is empty or all null.")
        xmin, ymin, xmax, ymax = (float(cast(float, v)) for v in bounds)
        return cls(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

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
