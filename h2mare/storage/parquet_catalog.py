"""Read-oriented query interface for the Hive-partitioned Parquet store."""

from __future__ import annotations

from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import polars as pl
from loguru import logger

from h2mare.types import BBox, DateRange
from h2mare.utils.datetime_utils import to_datetime

if TYPE_CHECKING:
    from h2mare.storage.parquet_plotter import ParquetPlotter
    from h2mare.storage.parquet_store import ParquetStore


class ParquetCatalog:
    """
    Read-oriented wrapper around ``ParquetStore``.

    Provides ``scan``, ``load``, and coverage-query methods backed by the
    underlying store. The store handles all writes; this class exposes them
    as a clean query interface to external consumers.
    """

    def __init__(self, store: "ParquetStore") -> None:
        self._store = store

    # --- Delegated attributes (keep callers working without store access) ---

    @property
    def parquet_root(self) -> Path:
        return self._store.parquet_root

    @property
    def time_col(self) -> str:
        return self._store.time_col

    @property
    def lon_col(self) -> str:
        return self._store.lon_col

    @property
    def lat_col(self) -> str:
        return self._store.lat_col

    @property
    def physical_schema(self):
        return self._store.physical_schema

    @property
    def physical_cols(self) -> set[str]:
        return self._store.physical_cols

    @property
    def partition_cols(self) -> set[str]:
        return self._store.partition_cols

    # --- Delegated coverage queries ---

    def get_schema(self) -> dict:
        return self._store.get_schema()

    def get_time_coverage(self) -> DateRange | None:
        return self._store.get_time_coverage()

    def get_var_coverage(
        self, columns: list[str] | None = None
    ) -> dict[str, DateRange]:
        return self._store.get_var_coverage(columns)

    def get_geoextent(self) -> BBox | None:
        return self._store.get_geoextent()

    # --- File resolution ---

    def _resolve_files(self, dates: Optional[Union[list, tuple]]) -> list[Path]:
        """
        Select parquet files efficiently based on dates.
        Uses partition directory shortcuts when year/month are partition keys;
        falls back to returning all files (scan() LazyFrame filter handles the rest).
        """
        parquet_root = self._store.parquet_root
        _partition_by = self._store._partition_by
        all_files = sorted(parquet_root.rglob("*.parquet"))

        if dates is None:
            return all_files

        has_year = "year" in _partition_by
        has_month = "month" in _partition_by

        if isinstance(dates, tuple) and len(dates) == 2:
            start, end = map(to_datetime, dates)

            if has_year and has_month:
                valid_partitions: set[tuple[int, int]] = set()
                y, m = start.year, start.month
                while (y, m) <= (end.year, end.month):
                    valid_partitions.add((y, m))
                    m += 1
                    if m > 12:
                        m = 1
                        y += 1
                return [
                    f
                    for f in all_files
                    if any(
                        f"year={y}/month={mo}" in f.as_posix()
                        for y, mo in valid_partitions
                    )
                ]
            elif has_year:
                valid_years = set(range(start.year, end.year + 1))
                return [
                    f
                    for f in all_files
                    if any(f"year={y}" in f.as_posix() for y in valid_years)
                ]
            else:
                return all_files

        elif isinstance(dates, list):
            if has_year and has_month:
                result: set[Path] = set()
                for d in dates:
                    try:
                        dt = to_datetime(d)
                        year, month = dt.year, dt.month
                        patterns = (
                            f"year={year}/month={month}",
                            f"{year}/{month:02d}",
                            f"{year}-{month:02d}",
                        )
                        for pattern in patterns:
                            result.update(parquet_root.rglob(f"*{pattern}*/*.parquet"))
                    except Exception as e:
                        logger.exception(f"Failed to parse date '{d}': {e}")
                        continue
                return sorted(result) or all_files
            else:
                return all_files

        else:
            raise ValueError("`dates` must be list or (start, end) tuple")

    # --- Scan / load ---

    def scan(
        self,
        dates: Optional[Union[list, tuple]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        columns: Optional[Union[str, list[str]]] = None,
    ) -> pl.LazyFrame:
        """
        Return a LazyFrame with optional date range, spatial filter, and column subset.

        Parameters
        ----------
        dates : list[str] or (str, str), optional
            Discrete list of dates or (start, end) for range filtering.
        bbox : (xmin, ymin, xmax, ymax), optional
            Spatial subset for lon/lat columns.
        columns : str or list[str], optional
            Columns to select (in addition to time/lon/lat).
        """
        if self._store.physical_schema is None:
            raise RuntimeError("No data in parquet store. Call add_data() first.")

        time_col = self._store.time_col
        lon_col = self._store.lon_col
        lat_col = self._store.lat_col

        parquet_files = self._resolve_files(dates)
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {self._store.parquet_root}"
            )

        lf = pl.scan_parquet(parquet_files).with_columns(pl.col(time_col).cast(pl.Date))

        if dates is not None:
            if isinstance(dates, tuple) and len(dates) == 2:
                start, end = map(to_datetime, dates)
                lf = lf.filter((pl.col(time_col) >= start) & (pl.col(time_col) <= end))
            elif isinstance(dates, list):
                normalized = [to_datetime(d) for d in dates]
                lf = lf.filter(
                    pl.any_horizontal(
                        [
                            (pl.col(time_col) >= d)
                            & (pl.col(time_col) < d + timedelta(days=1))
                            for d in normalized
                        ]
                    )
                )
            else:
                raise ValueError("`dates` must be list[str] or (start, end) tuple")

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            lf = lf.filter(
                (pl.col(lon_col) >= xmin)
                & (pl.col(lon_col) <= xmax)
                & (pl.col(lat_col) >= ymin)
                & (pl.col(lat_col) <= ymax)
            )

        if columns:
            columns = [columns] if isinstance(columns, str) else columns
            mandatory = {time_col, lon_col, lat_col}
            cols = list(mandatory.union(columns))
            existing_cols = [
                c for c in cols if c in list(self._store.physical_schema.keys())
            ]
            lf = lf.select(existing_cols)

        return lf

    def load(
        self,
        dates: Optional[Union[list, tuple]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        columns: Optional[Union[str, list[str]]] = None,
    ) -> pl.DataFrame:
        """Return a collected DataFrame."""
        return self.scan(dates=dates, bbox=bbox, columns=columns).collect()

    # --- Visualization ---

    @cached_property
    def plot(self) -> "ParquetPlotter":
        """Visualization accessor. Use ``catalog.plot.time_series(...)``."""
        from h2mare.storage.parquet_plotter import ParquetPlotter

        return ParquetPlotter(self)

    def _clear_plot_cache(self) -> None:
        if "plot" in self.__dict__:
            self.plot.clear_cache()
