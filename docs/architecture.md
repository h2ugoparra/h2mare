# Architecture

H2MARE is a five-stage pipeline: **Download → Convert → Compile → Index → Visualize**, with an optional **Extract** step for point or geometry inputs.

---

## Pipeline overview

```
CLI (h2mare/cli/main.py)
  └── PipelineManager
        ├── Downloader (CMEMSDownloader | AVISODownloader | CDSDownloader)
        ├── Netcdf2Zarr
        │     └── ZarrCatalog        (metadata index per variable)
        └── Compiler

Zarr2Parquet                         (uses ParquetIndexer)
Extractor                            (uses ParquetIndexer)
parquet2csv                          (reads Parquet directly)

ParquetIndexer  (facade)
  ├── ParquetStore   (writes: add_data, overlap resolution, atomic I/O)
  ├── ParquetCatalog (reads: scan, load, coverage queries)
  └── ParquetPlotter (via catalog.plot)
        ├── time_series()
        └── spatial_maps()
```

---

## Stage 1 — Download

**Classes:** `CMEMSDownloader`, `AVISODownloader`, `CDSDownloader`  
**Output:** raw NetCDF or GRIB files in `data/raw/downloads/<local_folder>/`

Each downloader resolves the date range to fetch (explicit or inferred from the existing store), splits it into yearly or monthly tasks (`DownloadTask`), and calls the provider API. For CMEMS variables the downloader automatically switches from the reprocessed (`rep`) dataset to the near-real-time (`nrt`) dataset at the appropriate date boundary.

---

## Stage 2 — Convert

**Class:** `Netcdf2Zarr` (`format_converters/netcdf2zarr.py`)  
**Output:** Zarr stores in `$STORE_ROOT/<local_folder>/`

Raw files are opened with xarray, regridded to a daily time axis, and written (or appended) as chunked Zarr stores. `ZarrCatalog` updates its Parquet index after each write so subsequent runs can resume from where they left off.

---

## Stage 3 — Compile

**Class:** `Compiler` (`processing/compiler.py`)  
**Output:** unified `h2ds` Zarr in `$STORE_ROOT/h2ds/`

All per-variable Zarr stores are opened, interpolated to the common 0.25° × 0.25° daily grid defined in `config.yaml`, and merged into a single dataset. Variables without data for a given period are skipped gracefully. The compiled dataset is also backed up to a local copy for fast access.

Special variables handled outside the general path:

- **`bathy`** — read from a static NetCDF file, no time dimension
- **`moon`** — computed on the fly from the `ephem` library
- **`o2`** — depth-sliced before interpolation

---

## ZarrCatalog

`ZarrCatalog` maintains metadata for each variable key. It tracks:

- File paths, modification times, and scan timestamps
- Temporal coverage (`start_date`, `end_date`) and provenance per source dataset
- Spatial extent and variable names

The catalog is used by `open_dataset` for efficient range queries without opening every Zarr file. It auto-detects stale entries by comparing disk file names and modification times against the index on each cold-start load.

---

## Extractor

`Extractor` reads h2ds Zarr stores and extracts time series at:

- **Point locations** — from a CSV file with `lat`/`lon`/`time` columns
- **Geometries** — from a SHP file (polygons or lines)

Geometry extraction uses `rioxarray.rio.clip()` and is parallelised with `ThreadPoolExecutor`. Point extraction vectorises coords with a module-level cached `KDTree` for spatial lookup and `numpy.searchsorted` for time, then selects with `isel()` (faster than coordinate-based `sel()`).

---

## ParquetIndexer / ParquetStore / ParquetCatalog

`ParquetIndexer` (`storage/parquet_indexer.py`) is the primary interface for the Hive-partitioned Parquet store (`year=YYYY/month=MM/`). It is used by `Zarr2Parquet` to persist h2ds data and can be used directly for analysis.

Internally it is a thin facade over two focused classes that can also be used directly:

| Class | Module | Responsibility |
|---|---|---|
| `ParquetStore` | `storage/parquet_store.py` | All filesystem I/O — `add_data`, atomic partition writes, DuckDB overlap resolution, schema management |
| `ParquetCatalog` | `storage/parquet_catalog.py` | Read interface — `scan`, `load`, coverage queries; wraps a `ParquetStore` |
| `ParquetIndexer` | `storage/parquet_indexer.py` | Facade combining both; preserves the original API for existing call-sites |

Key behaviors (implemented in `ParquetStore`):

- **Atomic writes** — each partition is written to a `.tmp_write_YYYY_MM` directory and renamed into place, preventing corrupt reads during a write.
- **Overlap resolution** — when new data overlaps existing partitions in time or columns, `resolve_dims_overlap()` merges them with a single DuckDB `FULL OUTER JOIN` across all affected files, then rewrites each partition atomically.
- **Schema evolution** — new columns in incoming data are detected and added to the physical schema; missing columns in existing partitions are backfilled with nulls.
- **Float32 storage** — float64 columns are downcast to float32 on write to reduce file size.

Key behaviors (implemented in `ParquetCatalog`):

- **Lazy scanning** — `scan()` returns a Polars `LazyFrame` filtered by date range and/or bounding box without loading the full dataset; `load()` collects it.

`ParquetIndexer` exposes a `plot` cached property (backed by `ParquetCatalog`) that returns a `ParquetPlotter` instance. The cache is invalidated automatically after each `add_data()` call.

---

## ParquetPlotter

`ParquetPlotter` (`storage/parquet_plotter.py`) is the visualization accessor for `ParquetIndexer`. It is accessed via `indexer.plot` — do not instantiate it directly.

| Method | Description |
|---|---|
| `time_series(var, agg_by)` | Interactive Plotly line chart aggregated over space and time (`day`, `week`, `month`, `season`, `year`) |
| `spatial_maps(var, agg_by)` | Climatological panel maps — 12 panels for `month`, 4 for `season` — showing the long-term mean at each grid cell |

Aggregation results are cached internally and cleared when new data is written.

---

## Key types

| Type | Description |
|---|---|
| `DateRange` | Dataclass with `start` / `end` datetime fields and overlap helpers |
| `BBox` | Dataclass with `xmin, ymin, xmax, ymax`; spatial overlap and label helpers |
| `DownloadTask` | Single download unit: `dataset_id`, `date_range`, `dataset_type` |
| `TimeResolution` | `YEAR` or `MONTH` enum controlling Zarr file granularity |
