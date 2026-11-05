# Package reference

Everything the `h2mare` packages export (their `__all__`) that does not have a page
of its own, plus the pipeline classes the CLI drives. The pages in this section
cover the main entry points in depth; this one is the index of the rest, one line
each — read the docstring for the details.

Module-level names that are not exported (e.g. the `processing.core.cds` steps the
registered processors call) are internal and may change without notice.

---

## Pipeline classes

| Class | Module | Description |
|---|---|---|
| `PipelineManager(app_config, registry, store_root, dry_run=False, start_date=None, end_date=None, no_convert=False, no_compile=False, no_parquet=False, h2ds_zarr_backup=False, h2ds_parquet_backup=False, zarr_backup_dir=None, parquet_backup_dir=None)` | `h2mare.pipeline_manager` | What `h2mare run` builds. `run(variables=None) -> bool` downloads, converts, compiles and writes Parquet for each var_key, returning `False` if any step failed. `registry` is normally `DOWNLOADER_REGISTRY` |
| `Netcdf2Zarr(var_key, *, app_config=None, store_root=None, download_root=None, file_period=FilePeriod.YEAR, date_format="year")` | `h2mare.format_converters` | What `h2mare convert` runs. `run(start_date=None, end_date=None) -> bool` converts the var_key's downloaded raw files; omitting the dates converts every matching raw file. The steps it applies are listed in [Architecture](../architecture.md#stage-2-convert) |
| `Zarr2Parquet(var_key, parquet_root, store_root=None)` | `h2mare.format_converters` | What `h2mare parquet` runs. `run(start_date=None, end_date=None, file_period=FilePeriod.MONTH, depth=None, variables=None) -> bool` has three modes: add-var (`variables`, no dates), explicit dates, and incremental append + per-column backfill. `sync_data(remote_root=None)` mirrors the store to a backup |
| `BaseConverter` | `h2mare.format_converters` | ABC of the two converters: abstract `run() -> bool`, no-op `validate()` |
| `BaseDownloader` | `h2mare.downloader` | ABC of the downloaders; see [Downloaders](downloader.md) |
| `ZarrIndex`, `ZarrReader`, `ZarrDirectoryScanner` | `h2mare.storage` | The halves behind the [`ZarrCatalog`](zarr_catalog.md) facade; see [Architecture](../architecture.md#zarrcatalog-and-its-collaborators) |
| `ParquetStore`, `ParquetCatalog` | `h2mare.storage` | The write and read halves behind [`ParquetIndexer`](parquet_indexer.md) |

---

## Settings and configuration — `h2mare`

| Name | Description |
|---|---|
| `get_settings()` | The process-wide `Settings`: resolved paths (`STORE_ROOT`, `PARQUET_DIR`, `DOWNLOADS_DIR`, `INTERIM_DIR`, `METADATA_DIR`, `FALLBACK_STORE_ROOT`, `ZARR_BACKUP_DIR`, `PARQUET_BACKUP_DIR`, `CLIMATOLOGY_DIR`, …) and the loaded `app_config` |
| `Settings.describe_paths()` | Every path grouped as stores / local workspace / local backups, with where the root and `STORE_ROOT` came from. Also the settings' `repr` |
| `Settings.override_store_root(path)` | Point `STORE_ROOT` elsewhere for the rest of the process (what `--store-path` does) |
| `Settings.load_app_config(config_path=None)` | Load and validate a `config.yaml` into an `AppConfig` — e.g. another project's, to pass as `app_config=` |
| `Settings.get_available_var_keys()` / `get_var_info(name)` | The configured var_keys; a variable's `variable_attrs` entry (`{}` if absent) |
| `Settings.ensure_directories()` | Create the project's `data/` tree up front (otherwise each directory appears on first write) |
| `Settings.ZARR_DIR` | **Deprecated** (warns): use `FALLBACK_STORE_ROOT` for where stores go without `STORE_ROOT`, or `ZARR_BACKUP_DIR` for the backup root |
| `AppConfig` | The whole validated `config.yaml` (`variables`, `global_attrs`, `variable_attrs`, `native_attr_overrides`) |
| `KeyVarConfigEntry` | One `variables:` entry; its fields are the [configuration reference](../configuration.md#variable-entries) |
| `VariablesConfig` | Alias of `dict[str, KeyVarConfigEntry]` |
| `SYSTEM_VAR_KEYS` | `{"h2ds", "bathy", "moon"}` — var_keys produced inside the pipeline, never downloaded |
| `validate_var_key(var_key, config)` / `validate_var_keys(var_keys, config)` | Raise if a var_key is not configured |
| `validate_file_period(period)` | Coerce `"year"`/`"month"` to `FilePeriod`, raising otherwise |

## Types — `h2mare`

| Name | Description |
|---|---|
| `DateLike` | `str \| pd.Timestamp \| datetime \| date` — what every date argument accepts |
| `DateRange` | `start` / `end` with `intersection`, `to_label`, `spans_multiple_years`, and `from_dataset` / `from_dataframe` / `from_polars` / `from_pandas` constructors |
| `BBox` | `xmin, ymin, xmax, ymax` with `to_tuple`, `to_label`, and `from_tuple` / `from_dataset` / `from_dataframe` constructors |
| `FilePeriod` | `YEAR` / `MONTH`: how much time one Zarr file covers (storage layout, not data cadence) |
| `DownloadTask` / `FTPDownloadTask` | One planned download: a dataset and date range (CMEMS/CDS), or one FTP file path and its source (`rep`/`nrt`, AVISO) |

---

## Storage helpers — `h2mare.storage`

| Name | Description |
|---|---|
| `catalog_for_var(var, var_key, *, read_from="auto", app_config=None)` | The `ZarrCatalog` that holds `var` — the var_key's own store or h2ds, by the same routing the `Extractor` uses |
| `coverage_for_var(var, var_key, *, read_from="auto", app_config=None)` | That catalog plus the dates it really holds `var` for |
| `compiled_var_key(app_config)` | The var_key of the compiled store (the entry with `source: h2mare`) |
| `get_store_coverage(var_key)` | Time span of a var_key's Zarr store, or `None` |
| `split_time_range(date_range, split)` | Split a `DateRange` into per-year or per-month pieces |
| `write_append_zarr(var_key, ds, path, encoding=None)` | Write a dataset, or append it to an existing store with the pipeline's overlap semantics |
| `chunk_dataset(ds, target_mb=32, time_dim="time", spatial_chunk=256, layout="timeseries", map_time_chunk=14)` | Downcast to float32 and chunk for storage; see [Map export](map_export.md#chunk_dataset-layout-argument) |
| `unified_time_chunk(ds, target_mb=32, time_dim="time")` | A time chunk size bringing chunks near `target_mb` |
| `rename_dims(ds)` | `longitude` / `latitude` / `valid_time` → `lon` / `lat` / `time` |
| `rename_source_vars(ds, renames, var_key="")` | Apply a `source_renames` map |
| `convert360_180(ds)` | 0–360 longitudes to −180–180 |
| `xr_float64_to_float32(ds)` (alias `ds_float64_to_float32`) / `polars_float64_to_float32(df)` | Downcast float64 columns |
| `aggregate_by_time(lf, vars, *, agg_by)` / `aggregate_by_time_stats(lf, var, *, agg_by)` | Spatial-mean time series (`day` … `year`), and its mean/std/min/max variant — what `time_series` / `stats_summary` plot |
| `aggregate_by_space_time(lf, vars, *, agg_by)` | Per-cell means by calendar month or season across years — what `spatial_maps` plots |
| `iter_store_parquet_files(root)` | Every committed Parquet file under a store, skipping interrupted `.tmp_write_*` partitions |
| `recover_zarr_store(store_root)` / `recover_parquet_store(parquet_root, partition_by)` | Reconcile what an interrupted write left (`*.zarr.bak` / `*.zarr.tmp`, `.tmp_write_*`). Called automatically by convert, compile and `ParquetStore` |

## Utilities — `h2mare.utils`

| Name | Description |
|---|---|
| `store_root_for(var_config, default_root=None)` | The root above a variable's `local_folder`: `--store-path` > its `store_root` > `default_root` > `STORE_ROOT` > `FALLBACK_STORE_ROOT` |
| `resolve_store_path(var_config, store_root=None, warn_if_missing=True)` | `store_root_for(...) / local_folder` |
| `resolve_download_path(var_config, download_root=None, warn_if_missing=True)` | The variable's raw-download directory |
| `static_layer_path(var_config, layer, store_dir)` | File of a static `layers` entry (bathy) |
| `create_filename_label(bbox, date_format, date_range=None)` / `create_label_from_dataset(ds, date_format)` | The extent/date label used in store file names |
| `GridBuilder(bbox, dx, dy, attributes=None, values_at="cell_center")` | Build a regular lat/lon grid (`generate_grid()`); used with `regrid_to`, see [Compiler](compiler.md#calling-it-yourself) |
| `haversine_min_distance_kdtree(coords1, coords2)` | Great-circle km from each point to its nearest neighbour in another set |
| `clip_land_data(ds)` | Mask land cells |
| `normalize_date(d)` / `normalize_dates(ds)` / `to_datetime(v)` | Coerce dates to midnight `pd.Timestamp`s / a stdlib `datetime` |
| `more_than_one_year(a, b)` | Whether two dates are more than a year apart |
| `safe_move_files(paths, dest_dir)` / `safe_rmtree(path)` | Move / delete with retries (Windows file locks) |
| `log_time` | Decorator logging a function's run time |

Plotting helpers live in `h2mare.utils.plot`, not re-exported here: see [Plotting utilities](plotting.md).

---

## Processors — `h2mare.processing`

The convert-time processors registered in `processing/registry.py`, exported so a
config-free conversion can reuse them (see [Reusing a registry
processor](adhoc_converters.md#reusing-a-registry-processor)). Each takes
`(ds, var_config, var_key)` and returns the processed dataset.

| Name | var_key | What it does |
|---|---|---|
| `process_sst` | `sst` | Kelvin → °C |
| `process_ssh` | `ssh` | Prepares the fields the `derived_vars` layers read |
| `process_chl` | `chl` | Prepares chlorophyll for its front layers |
| `process_fsle` | `fsle` | Converts the globally downloaded FSLE to −180–180 longitudes and subsets it to `bbox` |
| `process_atm_instante`, `process_atm_accum_avg`, `process_radiation`, `process_waves` | ERA5 var_keys | Prepare the ERA5 fields; reduce to daily only when the store is daily (`time_step`) — an hourly store is reduced at compile time instead |
| `EDDIESProcessor` | `eddies` | Rasterises the trajectory atlas onto the store grid (`trajectory_format`) |
| `apply_front_layers(ds, specs, owner, read_masks=None)` | any | The `front_layers` step; see [Front Layers](../front-layers.md) |
| `apply_boa_fronts(ds, fronts, owner)` / `FrontProcessor` | any | The `boa_fronts` step (Belkin–O'Reilly distances); kept for configs that declare it — the shipped one does not |

Also exported: `Extractor` and `ensure_row_id` ([Extractor](extractor.md)) and
`Compiler` ([Compiler](compiler.md)).
