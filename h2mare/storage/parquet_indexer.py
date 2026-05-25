"""
Parquet files handling and manipulation.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

if TYPE_CHECKING:
    from h2mare.storage.parquet_plotter import ParquetPlotter

import shutil
from datetime import timedelta

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from loguru import logger

from h2mare.types import BBox, DateRange
from h2mare.utils.datetime_utils import to_datetime

from .parquet_helpers import polars_float64_to_float32

_TIME_COMPONENTS = {"year", "month", "day"}


def _coerce_partition_value(s: str) -> int | str:
    try:
        return int(s)
    except ValueError:
        return s


class ParquetIndexer:
    def __init__(
        self,
        parquet_root: str | Path,
        *,
        time_col: str = "time",
        lon_col: str = "lon",
        lat_col: str = "lat",
        target_file_mb: int = 256,
        partition_by: list[str] = ["year", "month"],
    ):
        """
        Parquet data indexer.

        Args:
            parquet_root (str | Path): Root directory for parquet data
            time_col (str, optional): Time column name. Defaults to "time".
            lon_col (str, optional): Longitude column name. Defaults to "lon".
            lat_col (str, optional): Latitude column name. Defaults to "lat".
            target_file_mb (int, optional): Target size per Parquet file in MB. Defaults to 256.
            partition_by (list[str], optional): Hive partition column names. Temporal components
                ("year", "month", "day") are auto-derived from the time column; all other columns
                must be present in the DataFrame passed to add_data(). Defaults to ["year", "month"].

        Raises:
            ValueError: If time, lat, lon cols not in data.
        """
        self.parquet_root = Path(parquet_root)
        self.time_col = time_col
        self.lon_col = lon_col
        self.lat_col = lat_col
        self._target_file_mb = target_file_mb
        self._partition_by = list(partition_by)

        # partition_cols excludes these from the physical schema
        self.partition_cols = set(partition_by)

        # Initialize before any method calls that may reference these attributes
        self.physical_schema = None
        self.physical_cols: set[str] = set()

        # Set metadata
        self._init_dataset_metadata()

        # ---- No data in parquet_root ----
        if not self.parquet_root.exists() or not any(
            self.parquet_root.rglob("*.parquet")
        ):
            logger.warning(f"No data in {self.parquet_root}. Creating directory.")
            self.parquet_root.mkdir(parents=True, exist_ok=True)

        # ---- Data exists ----
        else:
            all_present = set([self.time_col, self.lon_col, self.lat_col]).issubset(
                set(self.get_schema().keys())
            )
            if not all_present:
                raise ValueError(
                    f"{self.time_col}, {self.lon_col} or {self.lat_col} not present in dataset."
                )
            self.physical_schema = self.get_schema()
            self.physical_cols = set(self.physical_schema.keys())

    def __repr__(self) -> str:

        if self.physical_schema is None:
            return ""

        time_cov = self.get_time_coverage()
        bbox = self.get_geoextent()

        return (
            f"ParquetIndexer(\n"
            f"  path={self.parquet_root},\n"
            f"  coverage={time_cov if time_cov is not None else None},\n"
            f"  bbox={bbox.to_label() if bbox is not None else None},\n"
            f"  n_columns={len(self.get_schema().keys())},\n"
            f")"
        )

    # ======================  METADATA ========================
    def _get_partition_level_values(self, col_name: str, parent: Path) -> list:
        prefix = f"{col_name}="
        return sorted(
            _coerce_partition_value(p.name.split("=", 1)[1])
            for p in parent.iterdir()
            if p.is_dir() and p.name.startswith(prefix)
        )

    def _build_partition_schema(self, df: pl.DataFrame) -> pa.Schema:
        return pa.schema(
            [
                pa.field(col, df[col].to_arrow().type, nullable=False)
                for col in self._partition_by
            ]
        )

    def _partition_path(self, partition: tuple) -> Path:
        path = self.parquet_root
        for col, val in zip(self._partition_by, partition):
            path = path / f"{col}={val}"
        return path

    def _partition_glob(self) -> str:
        parts = "/".join(f"{col}=*" for col in self._partition_by)
        return str(self.parquet_root / parts / "*.parquet").replace("\\", "/")

    def _partition_filter_sql(self, pairs: list[tuple]) -> str:
        clauses = []
        for vals in pairs:
            parts = [
                f"{col} = '{v}'" if isinstance(v, str) else f"{col} = {v}"
                for col, v in zip(self._partition_by, vals)
            ]
            clauses.append(f"({' AND '.join(parts)})")
        return " OR ".join(clauses)

    def _partition_filter_expr(self, partition: tuple) -> pl.Expr:
        exprs = [pl.col(col) == val for col, val in zip(self._partition_by, partition)]
        return exprs[0] if len(exprs) == 1 else pl.all_horizontal(exprs)

    def _get_time_coverage(self) -> DateRange:
        """Time coverage extraction. Uses partition directory shortcuts when year is a partition key."""
        if "year" in self._partition_by:
            years = self._get_partition_level_values("year", self.parquet_root)
            y0_path = self.parquet_root / f"year={years[0]}"
            yn_path = self.parquet_root / f"year={years[-1]}"
            if "month" in self._partition_by:
                first_dir = (
                    y0_path
                    / f"month={self._get_partition_level_values('month', y0_path)[0]}"
                )
                last_dir = (
                    yn_path
                    / f"month={self._get_partition_level_values('month', yn_path)[-1]}"
                )
            else:
                first_dir, last_dir = y0_path, yn_path
            first_file = sorted(first_dir.rglob("*.parquet"))[0]
            last_file = sorted(last_dir.rglob("*.parquet"))[-1]
            lf_min = pl.scan_parquet(first_file).select(pl.col(self.time_col).min())
            lf_max = pl.scan_parquet(last_file).select(pl.col(self.time_col).max())
            dt_min, dt_max = (r.item() for r in pl.collect_all([lf_min, lf_max]))
        else:
            all_files = list(self.parquet_root.rglob("*.parquet"))
            row = (
                pl.scan_parquet(all_files)
                .select(
                    [
                        pl.col(self.time_col).min().alias("mn"),
                        pl.col(self.time_col).max().alias("mx"),
                    ]
                )
                .collect()
            )
            dt_min, dt_max = row["mn"][0], row["mx"][0]

        return DateRange(dt_min, dt_max)

    def _init_dataset_metadata(self) -> None:
        """
        Initialize dataset-level metadata from parquet_root.
        Must be called whenever data becomes available.
        """
        if not any(self.parquet_root.rglob("*.parquet")):
            self._time_range = None
            self._geoextent = None
            self._dataset_meta_initialized = False
            return

        self._time_range = self._get_time_coverage()
        assert self._time_range.start <= self._time_range.end

        # Get geoextent from first file
        if "year" in self._partition_by:
            years = self._get_partition_level_values("year", self.parquet_root)
            y0_path = self.parquet_root / f"year={years[0]}"
            if "month" in self._partition_by:
                m0 = self._get_partition_level_values("month", y0_path)[0]
                first_dir = y0_path / f"month={m0}"
            else:
                first_dir = y0_path
        else:
            first_dir = self.parquet_root

        first_file = next(first_dir.rglob("*.parquet"))

        scan = pl.scan_parquet(first_file)

        self._geoextent = BBox.from_dataframe(
            scan, lon_col=self.lon_col, lat_col=self.lat_col
        )
        self._dataset_meta_initialized = True

    def _update_physical_schema(self, df: pl.DataFrame) -> None:
        """Updates physycal schema with new varaibles if present

        Args:
            df (pl.DataFrame): input dataframe
        """
        if self.physical_schema is None:
            return

        candidate_cols = set(df.columns) - self.partition_cols
        new_cols = candidate_cols - set(self.physical_schema.keys())
        if not new_cols:
            return

        logger.info(f"Extending physical schema with: {new_cols}")

        for col in new_cols:
            self.physical_schema[col] = df.schema[col]

        self.physical_cols = set(self.physical_schema.keys())

    def _init_physical_schema(self, df: pl.DataFrame) -> None:
        """When no data exists in parquet_root, collects schema from input df.
        Applies float64→float32 conversion to match what _prepare_df will write."""
        physical_df = df.select([c for c in df.columns if c not in self.partition_cols])
        self.physical_schema = dict(polars_float64_to_float32(physical_df).schema)
        self.physical_cols = set(self.physical_schema.keys())

    def _align_to_schema(
        self, df: pl.DataFrame, include_partitions: bool = True
    ) -> pl.DataFrame:
        """
        Align dataframe to physical schema, adding missing columns with nulls and reordering.

        Args:
            df (pl.DataFrame): Input dataframe
            include_partitions (bool): Whether to include partition columns in the output. Defaults to True.
        """
        physical_cols = set(self.physical_schema.keys())  # type: ignore
        partition_cols = set(self.partition_cols)

        # Split dataframe
        df_partitions = df.select([c for c in df.columns if c in partition_cols])
        df_physical = df.select([c for c in df.columns if c not in partition_cols])

        # Fails if schema update was skipped
        extra = set(df.columns) - physical_cols - self.partition_cols
        if extra:
            raise RuntimeError(
                f"New columns {extra} detected but physical schema was not updated"
            )

        # Missing physical columns → fill with nulls (exclude partition cols which are handled separately)
        missing = (physical_cols - self.partition_cols) - set(df_physical.columns)
        if missing:
            logger.warning(f"Missing variables in new data: {missing}")
            df_physical = df_physical.with_columns(
                [
                    pl.lit(None).cast(self.physical_schema[col]).alias(col)  # type: ignore
                    for col in missing
                ]
            )

        # Reorder columns to match physical schema (skip partition cols — not present in df_physical)
        df_physical = df_physical.select(
            [
                pl.col(col).cast(dtype)
                for col, dtype in self.physical_schema.items()  # type: ignore
                if col not in self.partition_cols
            ]
        )

        if not include_partitions:
            return df_physical

        # Reattach partition columns
        return pl.concat([df_physical, df_partitions], how="horizontal")

    # ========================  I/O  =========================

    def _resolve_time_col(
        self,
        df: pl.DataFrame,
        time_mode: Literal["date", "datetime"] = "date",
        fmt: str | None = None,
    ) -> pl.DataFrame:
        """Resolve time column type before saving.

        Args:
            df (pl.DataFrame): input dataframe
            time_mode (Literal['date', 'datetime'], optional): 'date' if daily dates (e.g. YYYY-MM-DD) or datetime (e.g. YYYY-MM-DD HH:MM:SS). Defaults to 'date'.
            fmt (str | None, optional): String format to parse time column. Defaults to None (auto-detect in to_datetime function).

        Raises:
            ValueError: `fmt` is only valid when time column is Utf8
            ValueError: time_mode must be 'date' or 'datetime'

        Returns:
            pl.DataFrame: _description_
        """
        dtype = df[self.time_col].dtype
        expr = pl.col(self.time_col)

        if dtype == pl.Utf8:
            if fmt is not None:
                expr = expr.str.to_datetime(format=fmt)
            else:
                expr = expr.cast(pl.Datetime, strict=False)
        else:
            if fmt is not None:
                raise ValueError("`fmt` is only valid when time column is Utf8")

            expr = expr.cast(pl.Datetime, strict=False)

        if time_mode == "date":
            expr = expr.dt.date()
        elif time_mode == "datetime":
            pass
        else:
            raise ValueError("time_mode must be 'date' or 'datetime'")

        return df.with_columns(expr.alias(self.time_col))

    def _max_rows_per_file(self, df: pl.DataFrame) -> int:
        bytes_per_row = df.estimated_size("b") / len(df)
        return max(1, int((self._target_file_mb * 1024**2) / bytes_per_row))

    def add_data(
        self,
        df: pl.DataFrame,
        time_mode: Literal["date", "datetime"] = "date",
        fmt: str | None = None,
    ) -> None:
        """
        Write ``df`` into the Hive-partitioned store, using one of three paths
        depending on how the new data relates to what is already on disk:

        1. **No existing data** — first write; partitions are created from scratch.
        2. **Non-overlapping dates** — new partitions are appended vertically; existing
           partitions are not touched.
        3. **Overlapping dates** — a coordinate-aligned horizontal merge is performed
           via a DuckDB ``FULL OUTER JOIN`` on ``(time, lon, lat)``:

           - *New columns only* (e.g. adding CHL where SST already exists): the new
             columns are joined onto the existing rows. Each ``(time, lon, lat)`` point
             ends up with both variables in one row; no rows are duplicated.
           - *Existing columns* (same variable, same dates): values in the new ``df``
             overwrite the stored values at matching coordinates.
           - *Mixed* (new columns + updated values): both effects apply in one pass.

        .. note::
            Overlap detection is coordinate-based, not partition-based. Two DataFrames
            that share a date but cover different spatial extents are appended, not merged.

        Args:
            df: DataFrame containing at least the time, lon, and lat columns. Any
                non-temporal partition columns (see ``partition_by``) must also be present.
            time_mode: ``'date'`` for daily resolution (``YYYY-MM-DD``), ``'datetime'``
                for sub-daily. Defaults to ``'date'``.
            fmt: strptime format string, only valid when the time column is a plain string.
                Leave ``None`` to auto-detect.
        """
        logger.info(f"Saving partitioned parquet to {self.parquet_root}")

        # Resolve time column
        df = self._resolve_time_col(df, time_mode=time_mode, fmt=fmt)

        # Strict validation: non-temporal partition columns must already be in the DataFrame
        custom_cols = set(self._partition_by) - _TIME_COMPONENTS
        missing_partition_cols = custom_cols - set(df.columns)
        if missing_partition_cols:
            raise ValueError(
                f"Partition columns {missing_partition_cols} not found in DataFrame. "
                "Non-temporal partition columns must be present in the data."
            )

        first_write = not self._dataset_meta_initialized

        if self.physical_schema is None:
            self._init_physical_schema(df)

        df = self._prepare_df(df)  # Adds auto-derived partition columns

        if any(self.parquet_root.rglob("*.parquet")):
            is_resolved = self.resolve_dims_overlap(df)
            if is_resolved:
                logger.success("Overlap resolved. Data added.")
                if "plot" in self.__dict__:
                    self.plot.clear_cache()
                return
            else:
                logger.info("Appending non-overlapping data.")
                df = self._align_to_schema(df)
        else:
            logger.info("Creating new parquet dataset.")

        max_rows = self._max_rows_per_file(df)
        ds.write_dataset(
            df.to_arrow(),
            base_dir=str(self.parquet_root),
            format="parquet",
            partitioning=ds.partitioning(
                self._build_partition_schema(df), flavor="hive"
            ),
            existing_data_behavior="overwrite_or_ignore",
            max_rows_per_file=max_rows,
            max_rows_per_group=max_rows,
        )

        # get metadata from first write
        if first_write:
            self._init_dataset_metadata()

        # invalidate plot cache if it has been created
        if "plot" in self.__dict__:
            self.plot.clear_cache()

    def resolve_dims_overlap(self, df: pl.DataFrame) -> bool | None:
        """
        Resolves spatial, temporal and column names overlap between existing and new data (df).
        If no temporal or vars overlap, returns None, else merges data and replaces partitions
        atomically to avoid memory issues. Returns True when overlap is resolved.

        Existing partitions are read in one parallel DuckDB query instead of N sequential
        Polars scans, then joined once. The write loop remains per-partition for atomicity.

        Args:
            df (pl.DataFrame): New data to be added to parquet dir

        Raises:
            ValueError: If spatial coordinates of new data are outside of existing data bbox.
        """
        import duckdb

        # ---- metadata from Existing data ----
        store_time_cov = self.get_time_coverage()
        store_bbox = self.get_geoextent()

        # ---- metadata from New data ----
        df_time_cov = DateRange.from_dataframe(df, time_col=self.time_col)
        df_bbox = BBox.from_dataframe(df, lon_col=self.lon_col, lat_col=self.lat_col)
        n_cols = set(df.columns)

        # Check spatial, temporal and vars overlap
        if store_bbox is None or df_bbox is None:
            return None
        if not store_bbox.overlaps(df_bbox):
            raise ValueError(
                "No spatial overlap between existing and new parquet data."
            )

        if store_time_cov is None or df_time_cov is None:
            return None
        if not store_time_cov.overlaps(df_time_cov):
            return None

        new_cols = n_cols - self.physical_cols
        duplicated_cols = self.physical_cols.intersection(n_cols) - {
            self.time_col,
            self.lat_col,
            self.lon_col,
        }

        if not duplicated_cols and not new_cols:
            return None

        self._update_physical_schema(df)

        # Recompute after schema update so new columns are included in exclude_cols
        # below. Without this, new cols present in partially-written partitions (e.g.
        # from an interrupted prior run) would appear in both the existing CTE and
        # df_new, producing duplicate column names in the DuckDB join.
        # Exclude new_cols: they were never written to any existing parquet file so
        # DuckDB would raise a BinderException if they appear in the EXCLUDE list.
        duplicated_cols = (
            self.physical_cols.intersection(n_cols)
            - {
                self.time_col,
                self.lat_col,
                self.lon_col,
            }
            - new_cols
        )

        # Classify partition columns: time-derived ones are dropped from df_new and
        # re-derived after the JOIN; custom ones are kept in df_new and added to the key.
        time_part_cols = [c for c in self._partition_by if c in _TIME_COMPONENTS]
        custom_part_cols = [c for c in self._partition_by if c not in _TIME_COMPONENTS]

        affected = df.select(self._partition_by).unique().rows()

        # Separate partitions that already exist from genuinely new ones
        existing_pairs = [
            p for p in affected if any(self._partition_path(p).rglob("*.parquet"))
        ]
        new_pairs = [p for p in affected if p not in set(existing_pairs)]

        # ---------- READ ALL EXISTING PARTITIONS IN ONE DUCKDB QUERY + JOIN ----------
        if existing_pairs:
            conn = duckdb.connect()

            # Drop only time-derived partition cols from df_new; custom cols stay as join keys
            conn.register("df_new", df.drop(time_part_cols) if time_part_cols else df)

            parquet_glob = self._partition_glob()

            # Detect new_cols that were partially written by a previous interrupted run.
            # DuckDB schema-unifies the glob so any column present in *any* file appears here.
            # Such columns must be excluded from the existing CTE to avoid a name clash in
            # the FULL OUTER JOIN (both sides would have the column → Polars renames to _1).
            partially_written: set[str] = set()
            if new_cols:
                try:
                    file_col_rows = conn.execute(
                        f"DESCRIBE SELECT * FROM read_parquet('{parquet_glob}', "
                        f"hive_partitioning=true) LIMIT 0"
                    ).fetchall()
                    actual_file_cols = {row[0] for row in file_col_rows}
                    # Exclude partition cols: DESCRIBE on a hive-partitioned parquet glob
                    # includes virtual year/month columns that aren't physical data columns.
                    partially_written = (
                        new_cols & actual_file_cols
                    ) - self.partition_cols
                    if partially_written:
                        logger.debug(
                            f"Partially-written columns detected (excluded from existing CTE): {partially_written}"
                        )
                except Exception as e:
                    logger.debug(f"Could not detect partially-written columns: {e}")

            # Exclude time-derived partition cols from existing CTE (re-derived after JOIN);
            # custom partition cols remain so they survive the FULL OUTER JOIN correctly.
            # Also exclude partially_written cols to prevent name clashes in the JOIN.
            exclude_cols = set(time_part_cols) | duplicated_cols | partially_written
            exclude_sql = ", ".join(exclude_cols)

            filter_sql = self._partition_filter_sql(existing_pairs)
            key_cols = ", ".join(
                [self.time_col, self.lon_col, self.lat_col] + custom_part_cols
            )

            merged = conn.execute(
                f"""
                WITH existing AS (
                    SELECT * EXCLUDE ({exclude_sql})
                    FROM read_parquet('{parquet_glob}', hive_partitioning = true)
                    WHERE {filter_sql}
                )
                SELECT * FROM existing
                FULL OUTER JOIN df_new USING ({key_cols})
            """
            ).pl()

            conn.close()

            # Re-derive time-component partition cols from the merged time column
            _time_rederive: dict[str, pl.Expr] = {
                "year": pl.col(self.time_col).dt.year().cast(pl.Int32),
                "month": pl.col(self.time_col).dt.month().cast(pl.Int32),
                "day": pl.col(self.time_col).dt.day().cast(pl.Int32),
            }
            rederive = [_time_rederive[c].alias(c) for c in time_part_cols]
            if rederive:
                merged = merged.with_columns(rederive)

            for partition in existing_pairs:
                partition_data = merged.filter(self._partition_filter_expr(partition))
                partition_data = self._align_to_schema(
                    partition_data, include_partitions=False
                )
                self.atomic_partition_write(partition_data, partition)

        # ---------- WRITE GENUINELY NEW PARTITIONS DIRECTLY ----------
        for partition in new_pairs:
            df_write = self._align_to_schema(
                df.filter(self._partition_filter_expr(partition)),
                include_partitions=False,
            )
            self.atomic_partition_write(df_write, partition)

        return True

    def atomic_partition_write(self, df: pl.DataFrame, partition: tuple) -> None:
        """
        Atomically replace a Hive-style partition directory by writing to a temp path first,
        then renaming into place.
        """
        final_path = self._partition_path(partition)
        tmp_path = (
            self.parquet_root / f".tmp_write_{'_'.join(str(v) for v in partition)}"
        )
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        tmp_path.mkdir(parents=True, exist_ok=True)

        max_rows = self._max_rows_per_file(df)
        ds.write_dataset(
            df.to_arrow(),
            base_dir=str(tmp_path),
            format="parquet",
            max_rows_per_file=max_rows,
            max_rows_per_group=max_rows,
        )
        # Remove existing partition safely
        if final_path.exists():
            shutil.rmtree(final_path, ignore_errors=True)

        final_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path.rename(final_path)

    def _resolve_files(self, dates: Optional[Union[list, tuple]]) -> list[Path]:
        """
        Select parquet files efficiently based on dates.
        Uses partition directory shortcuts when year/month are partition keys;
        falls back to returning all files (scan() LazyFrame filter handles the rest).
        """
        all_files = sorted(self.parquet_root.rglob("*.parquet"))
        if dates is None:
            return all_files

        has_year = "year" in self._partition_by
        has_month = "month" in self._partition_by

        # ---- range of dates ----
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

        # ---- Sparse list of dates ----
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
                            result.update(
                                self.parquet_root.rglob(f"*{pattern}*/*.parquet")
                            )
                    except Exception as e:
                        logger.exception(f"Failed to parse date '{d}': {e}")
                        continue
                return sorted(result) or all_files
            else:
                return all_files

        else:
            raise ValueError("`dates` must be list or (start, end) tuple")

    def scan(
        self,
        dates: Optional[Union[list, tuple]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        columns: Optional[str | list[str]] = None,
    ) -> pl.LazyFrame:
        """
        Returns a lazyframe (not loaded) with optional date range, spatial filter, and column subset.

        Parameters
        ----------
        dates : list[str] or (str, str), optional
            Discrete list of dates or (start, end) for range filtering.
        bbox : (xmin, ymin, xmax, ymax), optional
            Spatial subset for lon/lat columns.
        columns : list[str], optional
            Columns to select (in addition to date/lon/lat if needed).
        """
        if self.physical_schema is None:
            raise RuntimeError("No data in parquet store. Call add_data() first.")

        time_col = self.time_col
        lon_col = self.lon_col
        lat_col = self.lat_col

        parquet_files = self._resolve_files(dates)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {self.parquet_root}")

        lf = pl.scan_parquet(parquet_files).with_columns(pl.col(time_col).cast(pl.Date))

        # ---- Filter by date ----
        if dates is not None:
            # Range type: (start, end)
            if isinstance(dates, tuple) and len(dates) == 2:
                start, end = map(to_datetime, dates)
                lf = lf.filter((pl.col(time_col) >= start) & (pl.col(time_col) <= end))

            #  Discrete sample of dates
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

        # ---- Filter by bounding box ----
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            lf = lf.filter(
                (pl.col(lon_col) >= xmin)
                & (pl.col(lon_col) <= xmax)
                & (pl.col(lat_col) >= ymin)
                & (pl.col(lat_col) <= ymax)
            )

        # ---- Select columns ----
        if columns:
            columns = [columns] if isinstance(columns, str) else columns
            # Make sure we include columns needed for filters
            mandatory = {time_col, lon_col, lat_col}
            cols = list(mandatory.union(columns))
            existing_cols = [c for c in cols if c in list(self.physical_schema.keys())]
            lf = lf.select(existing_cols)

        return lf

    def load(
        self,
        dates: Optional[Union[list, tuple]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        columns: Optional[str | list[str]] = None,
    ) -> pl.DataFrame:
        """
        Returns a loaded dataframe.

        Args:
            dates (Optional[Union[list, tuple]], optional): Discrete list of dates or (start, end) for range filtering. Defaults to None, using the whole data.
            bbox (Optional[tuple[float, float, float, float]], optional): Spatial subset for lon/lat columns. Defaults to None, using the whole data.
            columns (Optional[list[str]], optional): Columns to select (in addition to time/lon/lat). Defaults to None, using the whole data.

        Returns:
            pl.DataFrame: Loaded dataframe
        """
        return self.scan(dates=dates, bbox=bbox, columns=columns).collect()

    # ========================= HELPERS =======================

    def get_time_coverage(self) -> DateRange | None:
        """Get time range from a parquet data."""
        if not self._dataset_meta_initialized:
            raise RuntimeError("Dataset metadata not initialized yet")
        return self._time_range

    def get_geoextent(self) -> BBox | None:
        """Get geographical extent from parquet data."""
        if not self._dataset_meta_initialized:
            raise RuntimeError("Dataset metadata not initialized yet")
        return self._geoextent

    def get_schema(self) -> dict[str, pl.DataType]:
        """Get schema (columns names and dtypes).
        Returns cached physical_schema when available, otherwise scans all parquet files
        to produce the union schema (handles partitions with different columns).
        """
        if self.physical_schema is not None:
            return self.physical_schema
        all_files = list(self.parquet_root.rglob("*.parquet"))
        if len(all_files) == 1:
            return dict(pl.read_parquet_schema(all_files[0]))
        # One file per partition directory is enough — schema is uniform within a partition
        rep_files = {f.parent: f for f in all_files}.values()
        schema: dict = {}
        for f in rep_files:
            schema.update(pl.read_parquet_schema(f))
        return schema

    def _prepare_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add partition columns for Hive partitioning and downcast float64→float32."""
        _time_exprs: dict[str, pl.Expr] = {
            "year": pl.col(self.time_col).dt.year().cast(pl.Int32),
            "month": pl.col(self.time_col).dt.month().cast(pl.Int32),
            "day": pl.col(self.time_col).dt.day().cast(pl.Int32),
        }
        derive = [
            expr.alias(col)
            for col, expr in _time_exprs.items()
            if col in self._partition_by and col not in df.columns
        ]
        if derive:
            df = df.with_columns(derive)
        return df.pipe(polars_float64_to_float32)

    # ----------------------------------------
    #                  PLOTS
    # ----------------------------------------
    @cached_property
    def plot(self) -> "ParquetPlotter":
        """Visualization accessor. Use ``indexer.plot.time_series(...)`` or ``indexer.plot.spatial_maps(...)``."""
        from h2mare.storage.parquet_plotter import ParquetPlotter

        return ParquetPlotter(self)
