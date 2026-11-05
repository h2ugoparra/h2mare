# Architecture

H2MARE is a five-stage pipeline: **Download → Convert → Compile → Index → Visualize**, with an optional **Extract** step for point or geometry inputs.

---

## Pipeline overview

```
CLI (h2mare/cli/)  — run · convert · compile · parquet · parquet2zarr · catalog · audit
  └── PipelineManager                     (`h2mare run`)
        ├── Downloader (CMEMSDownloader | AVISODownloader | CDSDownloader)
        │     └── DOWNLOADER_REGISTRY  (downloader/registry.py)
        ├── Netcdf2Zarr : BaseConverter
        │     └── ZarrCatalog  (facade)
        │           ├── ZarrIndex   (catalog persistence + queries)
        │           │     └── ZarrDirectoryScanner  (filesystem I/O + metadata)
        │           └── ZarrReader  (open_dataset: bbox, variables, time)
        ├── Compiler
        └── Zarr2Parquet : BaseConverter  (writes through ParquetIndexer)

Extractor                            (reads Zarr through ZarrCatalog — native stores or h2ds)
convert_parquet_to_csv               (one-way export; reads Parquet directly)
storage.audit                        (`h2mare audit` — interior gaps in the stores)

ParquetIndexer  (facade)
  ├── ParquetStore   (writes: add_data, overlap resolution, atomic I/O)
  ├── ParquetCatalog (reads: scan, load, coverage queries)
  └── ParquetPlotter (via indexer.plot)
        ├── time_series()
        ├── stats_summary()
        └── spatial_maps()
```

---

## Stage 1 — Download

**Classes:** `CMEMSDownloader`, `AVISODownloader`, `CDSDownloader`  
**Registry:** `DOWNLOADER_REGISTRY` (`downloader/registry.py`) — maps source keys (`"cmems"`, `"aviso"`, `"cds"`) to their downloader classes; passed into `PipelineManager` at startup.  
**Output:** raw NetCDF or GRIB files in `data/raw/downloads/<local_folder>/`

Each downloader resolves the date range to fetch (explicit or inferred from the existing store via `resolve_date_range`), splits it into yearly or monthly tasks (`DownloadTask`), and calls the provider API. For CMEMS variables the downloader automatically switches from the reprocessed (`rep`) dataset to the near-real-time (`nrt`) dataset at the appropriate date boundary.

---

## Stage 2 — Convert

**Class:** `Netcdf2Zarr` (`format_converters/netcdf2zarr.py`) — extends `BaseConverter`  
**Output:** Zarr stores in `$STORE_ROOT/<local_folder>/`

Raw files are opened with xarray and taken through the per-variable steps in this order, then written (or appended) as chunked Zarr stores at the variable's own resolution and cadence (`time_step: daily` or `hourly`):

1. `source_renames` — source names → published names (`analysed_sst` → `sst`)
2. the var_key's convert-time processor (`processing/registry.py`; unregistered var_keys pass through)
3. `boa_fronts` and `front_layers` — front layers (the shipped config uses `front_layers` only; see [Front Layers](front-layers.md))
4. `derived_vars` — rolling std / kinetic energy layers
5. `apply_cf_attrs` — CF metadata for variables and coordinates

`ZarrCatalog` updates its Parquet index after each write so subsequent runs can resume from where they left off.

---

## Stage 3 — Compile

**Class:** `Compiler` (`processing/compiler.py`)  
**Output:** unified `h2ds` Zarr in `$STORE_ROOT/h2ds/`

All per-variable Zarr stores are opened, regridded to the daily grid the `h2ds` entry defines in `config.yaml` (0.25° shipped), and merged into a single dataset. How each variable is regridded follows from its own resolution: a store finer than the grid is aggregated by an area-weighted mean, one at the grid's resolution or coarser is interpolated linearly, and a column a mean would destroy — an eddy track ID — is pinned to `nearest` in config. See [Regridding](api/compiler.md#regridding). Variables without data for a given period are skipped gracefully. With `zarr_backup=True` the compiled store is also mirrored to a local backup.

Special variables handled outside the general path:

- **`bathy`** — read from a static NetCDF file, no time dimension
- **`moon`** — computed on the fly from the `ephem` library
- **3-D variables** (`o2`, `thetao`, any var_key with `depth_levels`) — each listed variable is sliced at its configured depths before regridding; chosen by config rather than by a registry entry

The `h2ds` store is chunked for **extraction** (time-contiguous: long time block per small spatial tile), which makes point/geometry time series cheap but full-grid single-date reads costly. For interactive visualization, `export_map_zarr` produces a separate **map-optimized** sibling store (`h2ds_map`) chunked the other way — space-contiguous with a small time chunk — so one date's full field reads as a single small chunk. See [Map export](api/map_export.md).

---

## Stage 4 — Index (Parquet)

**Class:** `Zarr2Parquet` (`format_converters/zarr2parquet.py`) — extends `BaseConverter`  
**Output:** Hive-partitioned Parquet in `PARQUET_DIR` (`$STORE_ROOT/parquet/`, or `data/processed/parquet/` when `STORE_ROOT` is unset)

The compiled `h2ds` store is flattened to long-format rows (`time`, `lat`, `lon`, one column per variable) and written through `ParquetIndexer.add_data`. With no dates it runs incrementally: it appends what h2ds holds past the Parquet store's end, and backfills any column that was appended while its source lagged (`get_var_backfill_start`). `h2mare run` runs this step after compile unless `--no-parquet` (implied by `--no-compile` / `--no-convert`); `h2mare parquet` runs it alone.

---

## Stage 5 — Visualize

`ParquetIndexer.plot` (a [`ParquetPlotter`](api/parquet_plotter.md)) draws time series and climatological maps from the Parquet store; [`h2mare.utils.plot`](api/plotting.md) holds standalone helpers for DataFrames and Zarr fields. Both need the `viz` extra.

---

## ZarrCatalog and its collaborators

`ZarrCatalog` (`storage/zarr_catalog.py`) is the entry point for a variable's Zarr store. It builds store paths and composes two halves: a `ZarrIndex` that answers "which files, covering what", and a `ZarrReader` that turns an answer into an open dataset. The index in turn delegates filesystem I/O to a `ZarrDirectoryScanner`. This mirrors the `ParquetIndexer` facade on the Parquet side.

| Class | Module | Responsibility |
|---|---|---|
| `ZarrDirectoryScanner` | `storage/zarr_scanner.py` | Filesystem I/O — mtime snapshots, change detection, zarr metadata extraction |
| `ZarrIndex` | `storage/zarr_index.py` | Catalog persistence and refresh lifecycle, path and coverage queries |
| `ZarrReader` | `storage/zarr_reader.py` | `open_dataset` — sparse/range dispatch, variable selection, bbox subsetting, time normalisation |
| `ZarrCatalog` | `storage/zarr_catalog.py` | Facade — `build_file_path`, `summary`, delegation to the index and reader |

The dependency runs one way: `ZarrReader` reads the index, and the index knows nothing about the reader. `ZarrReader` holds no state of its own.

`ZarrIndex` owns the catalog DataFrame (`_df_cache`) and is the only writer of the catalog Parquet file. Note the cost of `auto_refresh=True`: every `.df` access re-stats the store directory, so callers reading the catalog in a loop should snapshot `.df` once rather than accessing it per iteration.

The catalog tracks per-variable:

- File paths, modification times, and scan timestamps
- Temporal coverage (`start_date`, `end_date`) and provenance per source dataset
- Spatial extent and variable names

`ZarrDirectoryScanner` detects stale catalog entries by comparing disk file names and modification times against the cached state on each access. `ZarrCatalog.refresh()` delegates to the scanner and rewrites the Parquet index only when changes are detected.

---

## BaseConverter

`BaseConverter` (`format_converters/base.py`) is the shared ABC for `Netcdf2Zarr` and `Zarr2Parquet`. It enforces an abstract `run() -> bool` method and a default no-op `validate()` hook, providing a stable contract for any future converter (e.g. `GRIB2Zarr`).

For files that are **not** registered in `config.yaml`, two config-free module
functions — `convert_netcdf_to_zarr` and `convert_zarr_to_parquet` — run the same
engine (generic transform + overlap-resolving write) driven by arguments instead
of a `var_key`. See [Ad-hoc converters](api/adhoc_converters.md).

---

## Extractor

`Extractor` (`processing/extractor.py`) is a standalone analysis tool, not a pipeline stage. It reads each var_key's own Zarr store through `ZarrCatalog` — or the compiled `h2ds` for hourly var_keys queried at daily cadence, or for any var_key with `read_from="compiled"` (see [Cadence](api/extractor.md#cadence)) — and extracts time series at:

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
| `time_series(var, agg_by)` | Interactive Plotly line chart of up to 4 variables, aggregated over space and time (`day`, `week`, `month`, `season`, `year`) |
| `stats_summary(var, agg_by)` | One variable's mean, ±1σ band, min/max and LOWESS trends over time |
| `spatial_maps(var, agg_by=...)` | Climatological panel maps — 12 panels for `month`, 4 for `season` — showing the long-term mean at each grid cell |
| `show(fig)` | Display a Plotly figure with the toolbar pinned |

Aggregation results are cached internally and cleared when new data is written.

---

## Key types

| Type | Description |
|---|---|
| `DateRange` | Dataclass with `start` / `end` datetime fields and overlap helpers |
| `BBox` | Dataclass with `xmin, ymin, xmax, ymax`; spatial overlap and label helpers |
| `DownloadTask` | Single download unit: `dataset_id`, `date_range`, `dataset_type` |
| `FilePeriod` | `YEAR` or `MONTH` enum controlling Zarr file granularity |
