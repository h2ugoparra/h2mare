# API Reference

The main classes you interact with directly.

| Class | Module | Description |
|---|---|---|
| `CMEMSDownloader` | `h2mare.downloader.cmems_downloader` | Download CMEMS datasets |
| `AVISODownloader` | `h2mare.downloader.aviso_downloader` | Download AVISO FTP datasets |
| `CDSDownloader` | `h2mare.downloader.cds_downloader` | Download ERA5 / CDS datasets |
| `Compiler` | `h2mare.processing.compiler` | Merge variables into the h2ds grid |
| `ZarrCatalog` | `h2mare.storage.zarr_catalog` | Query and manage Zarr stores |
| `Extractor` | `h2mare.processing.extractor` | Extract time series at points or geometries |
| `ensure_row_id` | `h2mare.processing.extractor` | Establish a unique merge key before extraction |
| `PipelineManager` | `h2mare.pipeline_manager` | Orchestrate the full download → convert → compile → Parquet run (`h2mare run`) |
| `ParquetIndexer` | `h2mare.storage.parquet_indexer` | Hive-partitioned Parquet store — write, read, and query |
| `ParquetPlotter` | `h2mare.storage.parquet_plotter` | Visualization accessor for `ParquetIndexer` (via `indexer.plot`) |
| `convert_netcdf_to_zarr` | `h2mare.format_converters.netcdf2zarr` | Config-free NetCDF/GRIB → Zarr for unregistered files |
| `convert_zarr_to_parquet` | `h2mare.format_converters.zarr2parquet` | Config-free Zarr → Parquet for unregistered stores |
| `convert_parquet_to_zarr` | `h2mare.format_converters.parquet2zarr` | Config-free Parquet → Zarr; rebuild per-period stores from Parquet |
| `convert_parquet_to_csv` | `h2mare.format_converters.parquet2csv` | Export Parquet data to day / month / year CSV files (one-way; alias `parquet2csv`) |
| `export_map_zarr` | `h2mare.format_converters.zarr_map_export` | Rewrite a store into a map-optimized (display) sibling, e.g. `h2ds` → `h2ds_map` |
| [Plotting utilities](plotting.md) | `h2mare.utils.plot` | Standalone plot helpers: `plot_maps`, `plot_snapshot`, `animate_vars`, `plot_interactive_map`, `plot_records_on_field` |

Everything else the packages export — settings, types, `Netcdf2Zarr` / `Zarr2Parquet`, storage and utility helpers, the convert-time processors — is indexed in the [Package reference](reference.md).
