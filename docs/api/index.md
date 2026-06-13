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
| `PipelineManager` | `h2mare.pipeline_manager` | Orchestrate the full download → convert pipeline |
| `ParquetIndexer` | `h2mare.storage.parquet_indexer` | Hive-partitioned Parquet store — write, read, and query |
| `ParquetPlotter` | `h2mare.storage.parquet_plotter` | Visualization accessor for `ParquetIndexer` (via `indexer.plot`) |
| `parquet2csv` | `h2mare.format_converters.parquet2csv` | Export Parquet data to day / month / year CSV files |
| `convert_netcdf_to_zarr` | `h2mare.format_converters.netcdf2zarr` | Config-free NetCDF/GRIB → Zarr for unregistered files |
| `convert_zarr_to_parquet` | `h2mare.format_converters.zarr2parquet` | Config-free Zarr → Parquet for unregistered stores |
