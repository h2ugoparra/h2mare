# ParquetIndexer

`ParquetIndexer` manages the Hive-partitioned Parquet store (`year=YYYY/month=MM/`). It is used by `Zarr2Parquet` to persist h2ds data and can be used directly for analysis.

```python
from h2mare.storage.parquet_indexer import ParquetIndexer

idx = ParquetIndexer("data/processed/parquet")
```

---

## Constructor

```python
ParquetIndexer(
    parquet_root,
    *,
    time_col="time",
    lon_col="lon",
    lat_col="lat",
    target_file_mb=256,
    partition_by=None,
    column_groups=None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `parquet_root` | — | Root directory for the Parquet store |
| `time_col` | `"time"` | Name of the time column |
| `lon_col` | `"lon"` | Name of the longitude column |
| `lat_col` | `"lat"` | Name of the latitude column |
| `target_file_mb` | `256` | Target size of each Parquet file, in MB |
| `partition_by` | `["year", "month"]` | Hive partition columns. The temporal ones (`"year"`, `"month"`, `"day"`) are derived from the time column; any other must be a column of the frame passed to `add_data()` (e.g. `["species", "year"]`) |
| `column_groups` | `None` | Maps a column to whatever produces it (a var_key, in the pipeline). Only used to report missing columns by producer rather than one by one |

Everything after `parquet_root` is keyword-only.

---

## Writing

### `add_data(df, time_mode="date", fmt=None)`

Add or replace data in the store. Handles first writes, schema evolution, and overlap resolution automatically.

```python
idx.add_data(df)
```

| Parameter | Default | Description |
|---|---|---|
| `df` | — | `pl.DataFrame` to write |
| `time_mode` | `"date"` | `"date"` for daily dates, `"datetime"` for sub-daily |
| `fmt` | `None` | strptime format string, only used when the time column is a string |

Key behaviours:
- **Atomic writes** — each partition is written to `.tmp_write_YYYY_MM` then renamed into place.
- **Overlap resolution** — when new data overlaps existing partitions in time or columns, all affected partitions are merged with a single DuckDB `FULL OUTER JOIN` and rewritten atomically.
- **Schema evolution** — new columns are detected and added; missing columns in existing partitions are backfilled with nulls.
- **Float32 storage** — Float64 columns are downcast to Float32 on write.

---

## Reading

### `scan(dates=None, bbox=None, columns=None) → pl.LazyFrame`

Return a lazy frame with optional filters. Does not load data into memory.

```python
lf = idx.scan(dates=("2021-01-01", "2021-12-31"), bbox=(-10, 30, 20, 50))
df = lf.collect()
```

| Parameter | Description |
|---|---|
| `dates` | `(start, end)` tuple for a range, or `list[str]` for discrete dates |
| `bbox` | `(xmin, ymin, xmax, ymax)` spatial filter |
| `columns` | Column name or list; `time`, `lon`, `lat` are always included |

### `load(dates=None, bbox=None, columns=None) → pl.DataFrame`

Same as `scan()` but collects and returns a `pl.DataFrame`.

---

## Metadata

| Method | Returns | Description |
|---|---|---|
| `get_schema()` | `dict[str, pl.DataType]` | Column names and dtypes |
| `get_time_coverage()` | `DateRange \| None` | First and last date of the stored rows. A frontier, not a completeness check |
| `get_var_coverage(columns=None)` | `dict[str, DateRange]` | Per column, the span of its *non-null* values — spots a lagging variable whose trailing rows exist but hold only nulls |
| `get_var_coverage_end(columns)` | `dict[str, datetime]` | Last non-null date per column, read newest partition first and stopping once every column is found (usually one partition). Columns never non-null are omitted |
| `get_var_backfill_start(columns, not_before=None)` | `dict[str, datetime]` | Earliest date each column has rows but only nulls — an append that landed while the column's source lagged. Drives the incremental backfill of `h2mare parquet` |
| `get_geoextent()` | `BBox \| None` | Spatial extent of stored data |

Properties: `parquet_root`, `time_col`, `lon_col`, `lat_col`, `physical_schema` (column → dtype as written, excluding Hive partition columns; `None` before the first write), `physical_cols` (columns stored in the files) and `partition_cols` (columns encoded in the directory names).

`ParquetIndexer` is a facade over `ParquetStore` (writes) and `ParquetCatalog` (reads), both in `h2mare.storage`; construct one of those directly when only one side is needed.

---

## Visualization

```python
idx.plot.time_series("sst", agg_by="month")
idx.plot.spatial_maps("sst", agg_by="season")
```

`indexer.plot` is a `cached_property` returning a [`ParquetPlotter`](parquet_plotter.md). The cache is invalidated automatically after each `add_data()` call.
