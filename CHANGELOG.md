# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - unreleased

### Breaking

- Renamed per-variable config fields in `config.yaml` / `KeyVarConfigEntry`:
  `variables` → `source_vars` and `variables_to_compile` → `compiled_vars`.
  Update existing `config.yaml` files accordingly.
- Removed the module-level `settings` singleton alias (`h2mare.settings` /
  `h2mare.config.settings`). Use `get_settings()` instead — it returns the same
  cached `Settings` instance and is reset-aware (`get_settings.cache_clear()`).
- Renamed the `STORE_DIR` environment variable / setting to `STORE_ROOT`.

### Added

- Multi-variable `time_series` plot and shared plotter options across
  `ParquetPlotter` methods.
- `ParquetPlotter.stats_summary()` with LOWESS trend lines.
- `parquet --add-var` flag for column-wise merges of an already-compiled
  variable into the existing h2ds Parquet store without reprocessing.
- Per-variable incremental compilation, so a lagging variable backfills
  independently when its source advances; plus catalog verbosity controls.
- Config-driven behaviour flags replacing hardcoded var_key checks:
  `compile_depth_slices`, `extract_depth_slices`, `filename_date_range`,
  `trajectory_format`, `rename_lonlat`, and `data_file` / `data_file_hires`.
- `Settings.CLIMATOLOGY_DIR` path.
- Exponential-backoff retry across all downloaders.
- Plotting options: `cmap` in `spatial_maps` / `plot_maps` / `plot_panel`,
  `grid_shape` in `spatial_maps`, extent-derived `figsize`, and accepting a
  `(lon, lat)` point in `time_series`.

### Changed

- Replaced the `settings` singleton with a cached `get_settings()` factory.
- Parquet collects now use the Polars streaming engine.
- Backups are opt-in: `--no-sync` replaced by `--no-backup` /
  `--no-zarr-backup` / `--no-parquet-backup`.
- Parquet writes now target multiple ~64 MB row groups per file.
- Split `ParquetIndexer` into `ParquetStore` + `ParquetCatalog`; extracted
  `ZarrDirectoryScanner` from `ZarrCatalog`, a `BaseConverter` ABC for format
  converters, and a dedicated `DOWNLOADER_REGISTRY` module.
- `Compiler` dispatch is now registry-driven instead of an if/elif chain.
- `PipelineManager.run()` returns a bool and the CLI exits with code 1 on
  failure.
- Switched tooling from black/isort to ruff for formatting and linting.

### Fixed

- `Settings` no longer pollutes consumer projects with `data/` and `logs/`
  directories on import.
- Compile is now a clean no-op when all variables are already up to date.
- Numerous correctness fixes in the fronts processor, FSLE processing
  (bbox handling), extraction (NaN coordinates), and Parquet schema unioning.

[0.2.0]: https://github.com/h2ugoparra/h2mare/compare/v0.1.1...HEAD
