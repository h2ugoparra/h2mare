"""Hive-partitioned Parquet write layer with DuckDB overlap resolution."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Literal

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from loguru import logger

from h2mare.types import BBox, DateRange

from .parquet_helpers import polars_float64_to_float32

_TIME_COMPONENTS = {"year", "month", "day"}


def _coerce_partition_value(s: str) -> int | str:
    try:
        return int(s)
    except ValueError:
        return s


class ParquetStore:
    """
    Low-level Hive-partitioned Parquet store.

    Handles all filesystem I/O: atomic partition writes, DuckDB-based overlap
    resolution, and dataset metadata. Query semantics (``scan`` / ``load``)
    live in ``ParquetCatalog``, which wraps this class.
    """

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
        self.parquet_root = Path(parquet_root)
        self.time_col = time_col
        self.lon_col = lon_col
        self.lat_col = lat_col
        self._target_file_mb = target_file_mb
        self._partition_by = list(partition_by)

        self.partition_cols = set(partition_by)
        self.physical_schema = None
        self.physical_cols: set[str] = set()

        self._init_dataset_metadata()

        if not self.parquet_root.exists() or not any(
            self.parquet_root.rglob("*.parquet")
        ):
            if not self.parquet_root.exists() and sys.stdin.isatty():
                answer = (
                    input(
                        f"Directory '{self.parquet_root}' does not exist. Create it? [y/N] "
                    )
                    .strip()
                    .lower()
                )
                if answer != "y":
                    raise FileNotFoundError(
                        f"Aborted: '{self.parquet_root}' was not created."
                    )
            logger.debug(f"No data in {self.parquet_root}. Creating directory.")
            self.parquet_root.mkdir(parents=True, exist_ok=True)
        else:
            all_present = {self.time_col, self.lon_col, self.lat_col}.issubset(
                set(self.get_schema().keys())
            )
            if not all_present:
                raise ValueError(
                    f"{self.time_col}, {self.lon_col} or {self.lat_col} not present in dataset."
                )
            self.physical_schema = self.get_schema()
            self.physical_cols = set(self.physical_schema.keys())

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
        """Initialize dataset-level metadata from parquet_root."""
        if not self.parquet_root.exists() or not any(self.parquet_root.rglob("*.parquet")):
            self._time_range = None
            self._geoextent = None
            self._dataset_meta_initialized = False
            return

        self._time_range = self._get_time_coverage()
        assert self._time_range.start <= self._time_range.end

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
        physical_df = df.select([c for c in df.columns if c not in self.partition_cols])
        self.physical_schema = dict(polars_float64_to_float32(physical_df).schema)
        self.physical_cols = set(self.physical_schema.keys())

    def _align_to_schema(
        self, df: pl.DataFrame, include_partitions: bool = True
    ) -> pl.DataFrame:
        physical_cols = set(self.physical_schema.keys())  # type: ignore
        partition_cols = set(self.partition_cols)

        df_partitions = df.select([c for c in df.columns if c in partition_cols])
        df_physical = df.select([c for c in df.columns if c not in partition_cols])

        extra = set(df.columns) - physical_cols - self.partition_cols
        if extra:
            raise RuntimeError(
                f"New columns {extra} detected but physical schema was not updated"
            )

        missing = (physical_cols - self.partition_cols) - set(df_physical.columns)
        if missing:
            logger.warning(f"Missing variables in new data: {missing}")
            df_physical = df_physical.with_columns(
                [
                    pl.lit(None).cast(self.physical_schema[col]).alias(col)  # type: ignore
                    for col in missing
                ]
            )

        df_physical = df_physical.select(
            [
                pl.col(col).cast(dtype)
                for col, dtype in self.physical_schema.items()  # type: ignore
                if col not in self.partition_cols
            ]
        )

        if not include_partitions:
            return df_physical

        return pl.concat([df_physical, df_partitions], how="horizontal")

    # ========================  I/O  =========================

    def _resolve_time_col(
        self,
        df: pl.DataFrame,
        time_mode: Literal["date", "datetime"] = "date",
        fmt: str | None = None,
    ) -> pl.DataFrame:
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

    def _max_rows_per_file(self, df: pl.DataFrame) -> tuple[int, int]:
        bytes_per_row = df.estimated_size("b") / len(df)
        max_file = max(1, int((self._target_file_mb * 1024**2) / bytes_per_row))
        max_group = max(1, int(((self._target_file_mb // 4) * 1024**2) / bytes_per_row))
        return max_file, max_group

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
           via a DuckDB ``FULL OUTER JOIN`` on ``(time, lon, lat)``.
        """
        logger.info(f"Saving partitioned parquet to {self.parquet_root}")

        df = self._resolve_time_col(df, time_mode=time_mode, fmt=fmt)

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

        df = self._prepare_df(df)

        if any(self.parquet_root.rglob("*.parquet")):
            is_resolved = self.resolve_dims_overlap(df)
            if is_resolved:
                logger.success("Overlap resolved. Data added.")
                return
            else:
                logger.info("Appending non-overlapping data.")
                df = self._align_to_schema(df)
        else:
            logger.info("Creating new parquet dataset.")

        max_file, max_group = self._max_rows_per_file(df)
        ds.write_dataset(
            df.to_arrow(),
            base_dir=str(self.parquet_root),
            format="parquet",
            partitioning=ds.partitioning(
                self._build_partition_schema(df), flavor="hive"
            ),
            existing_data_behavior="overwrite_or_ignore",
            max_rows_per_file=max_file,
            max_rows_per_group=max_group,
        )

        if first_write:
            self._init_dataset_metadata()

    def resolve_dims_overlap(self, df: pl.DataFrame) -> bool | None:
        """
        Resolve spatial, temporal and column-name overlap between existing and new data.

        Returns ``True`` when overlap is detected and partitions are rewritten,
        ``None`` when no overlap requires merging (caller should append instead).

        Raises:
            ValueError: If the new data has no spatial overlap with the store.
        """
        import duckdb

        store_time_cov = self.get_time_coverage()
        store_bbox = self.get_geoextent()

        df_time_cov = DateRange.from_dataframe(df, time_col=self.time_col)
        df_bbox = BBox.from_dataframe(df, lon_col=self.lon_col, lat_col=self.lat_col)
        n_cols = set(df.columns)

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

        existing_pairs = [
            p for p in affected if any(self._partition_path(p).rglob("*.parquet"))
        ]
        new_pairs = [p for p in affected if p not in set(existing_pairs)]

        if existing_pairs:
            conn = duckdb.connect()

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

        max_file, max_group = self._max_rows_per_file(df)
        ds.write_dataset(
            df.to_arrow(),
            base_dir=str(tmp_path),
            format="parquet",
            max_rows_per_file=max_file,
            max_rows_per_group=max_group,
        )
        if final_path.exists():
            shutil.rmtree(final_path, ignore_errors=True)

        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.rename(final_path)

    # ========================= QUERIES =======================

    def get_time_coverage(self) -> DateRange | None:
        """Return the time range covered by the store."""
        if not self._dataset_meta_initialized:
            raise RuntimeError("Dataset metadata not initialized yet")
        return self._time_range

    def get_geoextent(self) -> BBox | None:
        """Return the geographic bounding box of the store."""
        if not self._dataset_meta_initialized:
            raise RuntimeError("Dataset metadata not initialized yet")
        return self._geoextent

    def get_schema(self) -> dict[str, pl.DataType]:
        """Return the union schema across all partitions."""
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
