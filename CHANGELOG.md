# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking (config):** the bathy entry's `data_file` / `data_file_hires` are
  replaced by named `layers` (`15s`, `60s`, `0.25deg` → file name) plus
  `compile_layer` and `extract_layer`. An old config fails at load. Config is
  now the only place a layer's file name lives.
- Bathy extraction reads one layer for points and geometries alike, chosen by
  `extract_layer` or `Extractor(bathy_layer=...)`, instead of the input type
  deciding the grid (0.25° for csv, 15s for shp).
- A geometry's `bathy_std` is now the polygon mean of the layer's stored std,
  the same estimator as for points and as `sst_std`/`adt_std`, rather than a
  std of depth within the polygon.

### Added

- `scripts/bathymetry.py` builds a 60s layer next to the 15s one, both as tiled
  Zarr (`etopo2022_<res>_…_bathy-std.zarr`) holding `bathy` and a 3×3 rolling
  `bathy_std` (the entry's `derived_vars`), with the ETOPO source's global and
  `z` attributes carried over. `--layers` builds a subset.

## [0.8.1] - 2026-09-14

### Fixed

- An append to an `int16` store whose values fall outside the store's frozen
  scale no longer fails the run. The store is **repacked** first: rewritten
  with that variable's `scale_factor`/`add_offset` re-derived over the stored
  data and the incoming range together, then appended to. Previously the
  append was refused with an instruction to delete the store and re-convert —
  which, with `archive_raw: false`, meant re-downloading months of hourly ERA5.
  This is routine for zero-bounded, heavy-tailed fields: a yearly store built
  from January–July carries `tp`'s range from those months, and August's storm
  season exceeds it (hit on `atm-accum-avg` 2026).
  - The new scale spans the stored data's *actual* range, not the old
    representable window, so the headroom is not compounded on every repack.
  - Variables that still fit keep their encoding and round-trip exactly;
    existing values of a repacked variable move by at most half a quantisation
    step.
  - The rewrite is atomic (tmp → backup-swap), and costs one rewrite of the
    yearly file, logged as a warning.
  - A variable that cannot be repacked (not `int16`, or with no range to scale
    over) is still refused before anything is written.

## [0.8.0] - 2026-09-03

### Breaking

- `TimeResolution` is renamed `FilePeriod`, and `time_resolution` is
  `file_period` throughout. The old name described how often a file *sampled*
  time, which stopped being true once a store could hold hourly data in monthly
  files; it names the period a file spans. `h2mare.types.TimeResolution`
  remains as an alias, but the package root now exports `FilePeriod` — `from
  h2mare import TimeResolution` no longer resolves.
- `validate_time_resolution` is renamed `validate_file_period`, with no alias.
- A variable's `units` is no longer read from config. Variable and coordinate
  attributes now come from one table in `storage/xarray_helpers.py`, so a
  `units` left in config.yaml is inert — and, since unrecognised per-variable
  keys are now warned about, noisy.

### Added

- **Hourly stores.** A variable declares `time_step: hourly` and converts at its
  native cadence instead of being collapsed to daily on the way in. Four
  var_keys now keep hourly Zarr stores — the three ERA5 ones (`atm-instante`,
  `atm-accum-avg`, `radiation`) back to 1998, plus `waves` — and
  `compile_default` reduces any hourly store to daily for h2ds. Gap checks run
  at each store's own cadence.
- `store_dtype: int16` opts a variable's store into scale/offset encoding,
  roughly halving it against float32. The scale is computed once from the
  source and frozen; an append the frozen scale cannot represent is refused
  rather than silently clipped.
- **`h2mare audit`** finds days missing from a store's interior — the gaps a
  resumable pipeline will not notice, because coverage only tracks its ends.
  `--all` sweeps every var_key on the time axis in about a minute; `--values`
  also reads data to catch days present but empty. Exits non-zero on findings.
- `known_gaps` records days a provider never published (`chl` has 11), so the
  audit stays credible instead of reporting the same known holes every run.
  Suppressed days are listed, not just counted.
- **CF/ACDD metadata.** `apply_cf_attrs` applies variable and coordinate
  attributes from one table on both write paths, with `native_attr_overrides`
  for the cases where a native store and h2ds legitimately differ (`msl` is Pa
  natively, hPa in h2ds). Root attributes declare `Conventions` and compute
  ACDD extents per file. Coordinate attributes are load-bearing, not cosmetic:
  without them `rio.clip` cannot resolve lon/lat and geometry extraction
  returns all-NaN. An append never rewrites attributes, so the existing stores
  were backfilled once out of band; the one-off script has since been removed.
- **Provenance** now means dates *covered* rather than dates requested, carries
  per-source rep/nrt lineage through to h2ds, and survives appends instead of
  being wiped by them. `refresh_provenance` repairs records narrower than the
  file they describe.
- A variable can declare its own `store_root`, resolved by
  `utils/paths.py::store_root_for` as `--store-path` > the variable's
  `store_root` > `STORE_ROOT` > `ZARR_DIR`. Declaring none anywhere resolves
  exactly as before.
- **Cadence-aware extraction.** `Extractor` takes `time_cadence`
  (`auto`/`daily`/`hourly`) for how `time_col` is read and `read_from`
  (`auto`/`native`/`compiled`) for which store answers — two independent axes,
  because an hourly store can serve sub-daily input while a date-only query
  against the same var_key belongs to h2ds.
- An incremental compile can backfill an interior hole rather than only
  extending the ends.
- `h2mare catalog` reports the store bbox; the end-of-run tally reports the
  share of rows null.
- A minimal example config to start from, and docstrings on the public surface
  people meet first.

### Changed

- `expect_daily` was renamed `expect_contiguous_time` and then removed
  outright — it was an escape hatch nothing ever used.
- Unrecognised per-variable config keys are warned about instead of ignored.
- `ZarrCatalog`'s repr says which cadence it is reporting.
- Runtime dependencies that were only ever installed transitively are now
  declared; the docs tooling is no longer shipped to consumers.
- CI checks formatting, lints `scripts/`, tests on 3.12 and 3.13, fails when
  `uv.lock` has drifted from `pyproject.toml`, and runs pyright as an
  informational job so the finding count cannot grow unnoticed.
- `copernicusmarine` bumped to 2.4.1, dropping the `sparse` constraint.

### Fixed

- Zarr stores with drifted coordinate axes are readable again. The same axis
  written on different occasions can disagree in the last float bits, which
  `open_mfdataset(combine="by_coords")` compares exactly; agreeing axes are
  snapped onto the earliest file's, with per-coordinate tolerances because the
  units differ. Ragged variable sets are padded to the union so a store whose
  files carry different variables still combines as one group.
  `scripts/repair_axis_drift.py` rewrites drifted coordinates in place.
- A disjoint backfill no longer duplicates the whole Zarr file.
- A stale backup is cleared before the swap, so a failed write can actually
  roll back; a failed h2ds backup copy is no longer reported as a success.
- Write-path overlap boundaries resolve by instant rather than by whole day, an
  overlapping rewrite against a sub-daily axis is refused, and whole calendar
  days are read so an hourly store is not truncated.
- ERA5 radiation accumulations that are already hourly are no longer
  differenced again, and the store stops advertising rates as accumulations.
- Wave direction is averaged and interpolated as a circle, not a line.
- The Ekman climatology aligns to the calendar rather than to `dayofyear`,
  rolling features seed with 20 days, the curl stencil keeps lat/lon
  contiguous, and upwelling events are not counted before their window is full.
- CMEMS hourly subsets request the whole final day, a trailing day the provider
  is still publishing is held back, and four date-bounded time slices cover the
  whole last day.
- Extraction: a checkpoint written for a different input is discarded rather
  than replayed onto it; an extraction interrupted mid-checkpoint recovers;
  every depth level a 3-D variable publishes is extracted, and those levels are
  no longer reported as a gap; `vars=[]` reads as "everything"; the CRS is set
  after renaming lon/lat, not before; a store lacking compile-derived variables
  says so instead of returning nulls.
- `--store-path` reaches every pipeline step, `Extractor`'s `store_root` reaches
  the reads, source coverage is read from the root the compile reads from, and
  a catalog sidecar belonging to another store root triggers a rescan.
- A variable is routed to the store that actually holds it and dated per
  variable, rather than per source.
- Appending outside the store's date range registers new columns.
- A Parquet string time column is parsed instead of cast to null; the CLI exits
  non-zero when conversion fails; a Parquet null finding names the file it
  refers to.
- Partial AVISO downloads are surfaced instead of swallowed, the FTP connection
  is closed rather than leaked, and the spent eddies download manifest is
  deleted after staging.
- One-day downloads are converted instead of silently discarded, and the
  downloads folder is only removed once a run has consumed it.
- The raw dataset is closed so `archive_raw` can move files on Windows, and the
  raw-archive period folder is built with a portable separator.
- Importing h2mare no longer creates `data/` and `logs/` trees or suppresses
  every warning process-wide.
- `UnicodeEncodeError` on non-UTF-8 consoles.
- `join="exact"` and `data_vars` are pinned on the reader's `open_mfdataset`, so
  xarray's changing defaults stay no-ops.
- Missing dates and coordinates are rejected at the boundary instead of passed
  on as `NaT`/`NaN`.
- Two failures that hid their real cause now report it; the Zarr → Parquet step
  is named in the log and reports how long it took.

### Performance

- The hourly ekman source is reduced in slabs over the store rather than
  materialised whole.
- int16 encoding ranges are found in one pass over the source.

## [0.7.0] - 2026-08-05

### Breaking

- `h2mare.utils` no longer re-exports the plotting helpers (`plot_maps`,
  `plot_snapshot`, `animate_vars`, `plot_interactive_map`,
  `plot_records_on_field`). Import them from `h2mare.utils.plot` instead. The
  package-level re-export forced `utils` to import `storage`, and importing
  anything from `h2mare.utils` pulled in matplotlib, cartopy and geopandas.
- `resolve_date_range` moved from `h2mare.utils.date_range` to
  `h2mare.storage.coverage`, beside `get_store_coverage` and
  `split_time_range`. `h2mare/utils/date_range.py` is removed.

### Changed

- `utils/` is a leaf package again: it imports `storage` nowhere, at module
  scope or inside a function. `storage` may import `utils`, not the reverse.
  The column check shared by both (`_required_columns`) moved to
  `validators.py` as `validate_columns`.
- The Zarr → Parquet step logs one labelled line per window naming its regime
  (append, backfill, add-var, explicit range) instead of announcing each window
  twice with the same dates. The pre-append store end — the pivot separating
  the append from the backfills — is logged once, so the windows below it can
  be interpreted.
- A lagging source is reported by `var_key` rather than by every compiled
  column it owns, and only once per producer per run.

### Fixed

- `xr.concat` calls pass `join` explicitly. xarray is changing the default from
  `"outer"` to `"exact"`, which would have turned a coordinate mismatch into a
  `ValueError` on a routine dependency bump.
- The missing-columns warning no longer re-fires when the gap *shrinks* (a
  source partly catching up) or when two writes miss different columns of the
  same producer.

## [0.6.0] - 2026-07-31

### Added

- `raw_include` config field — a regex restricting which raw files a variable
  converts. Needed for AVISO eddies, whose download directory holds `long`,
  `short` and `untracked` META3.2 trajectory variants side by side when only
  the long ones belong in the store.
- `h2mare convert` accepts `--start-date` / `--end-date`, so a single period
  can be re-converted from raw files already on disk without re-downloading.

### Changed

- `ZarrCatalog` is now a facade over two extracted collaborators, `ZarrIndex`
  (resume index) and `ZarrReader` (dataset opening). Its public API is
  unchanged.

### Fixed

- AVISO downloads write a `h2mare_manifest.json`, and the eddies converter —
  which bypasses the generic `Netcdf2Zarr` path — now stamps `source_datasets`
  from it. Previously no provenance was written at all, so `ZarrCatalog` fell
  back to `dataset_id_rep` and labelled near-real-time data as delayed-time,
  producing rep/nrt ranges that overlapped in `h2mare catalog` even though the
  FTP directories are disjoint.
- The eddies grid is read from a single canonical Zarr file instead of a union
  across the whole store. Combining files whose axes differ only in the last
  floating-point bits produced a doubled axis of near-duplicate points, and
  writing that back made each run read a worse grid than the last.
- Eddies conversion prefers the reprocessed dataset over near-real-time where
  both cover a date, and resolves its conversion window per file rather than
  per eddy type.
- Raw staging no longer deletes the files it is meant to move.

### Removed

- `scripts/rechunk_stores.py` and `scripts/profile_extract_chunking.py` (both
  added in 0.3.0), and `scripts/ekman_derived_vars.py`. All three were one-time
  repairs for data written by older code, and the defects they fixed are now
  prevented at write time: `chunk_dataset` tiles spatial dims on write, and
  `add_engineered_ekman` computes the Ekman derived variables during
  conversion. Two further repair tools added since v0.5.0
  (`repair_aviso_provenance.py`, `standardize_zarr_filenames.py`) were removed
  in the same window and so never appeared in a release.
- The Task Scheduler wrapper's `.gitignore` entry; the wrapper now lives in the
  runtime root (`H2MARE_ROOT/scripts/`) beside the config and data it drives,
  rather than in the checkout.

## [0.5.0] - 2026-06-18

### Breaking

- `Extractor(...)` requires an `index_col`. The positional `__row_id__` key it
  used to generate existed only on the Extractor's side, so merging results
  back onto the caller's dataframe relied on implicit positional alignment,
  which breaks silently under any filter, sort or dedup. New module-level
  `ensure_row_id` establishes the key on the frame the caller keeps.
- `archive_raw` is a required config field on every variable, controlling
  whether raw NetCDF/GRIB files are kept or deleted after conversion. The
  decision was previously hardcoded by source provider inside the converter.

### Added

- `convert_parquet_to_zarr` and a `h2mare parquet2zarr` command — the inverse
  of `convert_zarr_to_parquet`, pivoting long-format Parquet rows back to
  gridded per-period Zarr.
- A `"map"` chunk layout on `chunk_dataset` (alongside the `"timeseries"`
  default) plus `export_map_zarr`, which rewrites a per-period store into a
  map-chunked sibling (`h2ds` → `h2ds_map`) for interactive fields. See
  `docs/api/map_export.md`.
- Standalone plot helpers `plot_interactive_map` and `plot_records_on_field`,
  with a new `docs/api/plotting.md` covering them and the previously
  undocumented `plot_maps` / `plot_snapshot` / `animate_vars`.
- `Extractor.extract_from_dataset`, for extracting from an arbitrary in-memory
  xarray dataset.

### Changed

- `parquet2csv` is now `convert_parquet_to_csv`, matching the
  `convert_<src>_to_<dst>` convention. The old name remains as a deprecated
  alias.
- `pattern` is optional and its capture-group contract with
  `filename_date_range` is documented and validated at config load, turning a
  mid-pipeline "not enough values to unpack" into a clear error.
- `subset` is documented and warned about as CMEMS-only, and dropped from the
  variables where it was a silent no-op.

### Fixed

- Pipeline write paths are crash-consistent and resumable. They were atomic
  against exceptions but not against a hard kill, which could strand temp or
  backup state that nothing reconciled — letting a resumed run trust a partial
  store, drop history, or read orphaned temp files. New `storage/recovery.py`
  reconciles orphaned `*.zarr.bak` / `*.zarr.tmp` and stranded `.tmp_write_*`
  partitions before gap detection reads the store.
- CMEMS re-downloads pass `overwrite=True`, so a re-triggered fetch replaces a
  corrupt or partially written file instead of leaving a `filename_(1).nc`
  duplicate for the convert-step glob.
- All config-free converters are exported from the `format_converters` package
  root, not just `convert_parquet_to_zarr`.
- A circular import between `h2mare.utils.plot` and `h2mare.storage`.
- SHP geometry survives a checkpoint resume as usable shapely objects rather
  than WKB bytes, and is dropped from CSV output where it was dead weight.

## [0.4.0] - 2026-06-12

### Fixed

Several of these caused silent data loss in the Parquet and Zarr write paths;
anyone running 0.3.x or earlier should upgrade.

- A partial-window Parquet merge wiped the column outside the window. Any
  backfill smaller than a partition erased that column's other days in it.
  Incoming data now wins only where it actually has rows, and stored values
  survive everywhere else.
- Appending into an existing Parquet partition destroyed its rows. pyarrow's
  default `part-{i}.parquet` basename restarts at 0 on every write, so combined
  with `overwrite_or_ignore` an append silently overwrote `part-0`. Writes now
  use a per-write unique basename prefix.
- A compile for a subset of variables (`h2mare run -v ssh`) wiped every other
  variable over the appended range, because `xr.concat` NaN-fills variables
  missing from the incoming dataset.
- An explicit compile of a window inside the stored range dropped every date
  after that window.
- Parquet backfill missed holes hidden behind non-null islands, leaving a gap
  permanently unreachable once a later append jumped the window past it.
- Grid labels are snapped to a canonical 4 dp grid. A source reprocessing its
  product can shift the grid by sub-1e-4 float noise (CMEMS seapodym moved
  longitudes ~1.5e-5°), which unioned rather than aligned the axes on merge —
  doubling the axis and NaN-filling each block at the other's phantom cells.
- Time-less variables (`bathy`, `bathy_std`) no longer raise `MergeError` on
  Zarr append.
- Three latent bugs from a codebase audit: `ZarrCatalog.get_bbox` returning
  `None` for an already-`BBox` config value, CLI date validation rejecting
  single-day ranges, and `ParquetStore` misreporting coverage for a partition
  split across multiple files.
- `ParquetStore.__init__` no longer prompts via `input()`, which stalled any
  interactive run that reached it.
- Post-run cleanup prunes nested empty download subfolders, which previously
  survived for eddies' rep/nrt staging dirs and multi-level `local_folder`
  paths.

### Performance

- Clean trailing Zarr appends write only the new chunks via
  `to_zarr(append_dim=...)` instead of rewriting the whole file, with a
  fallback to the rewrite path on any precondition or verification miss.
- Parquet overlap resolution joins one partition at a time rather than
  materializing every affected partition in a single outer join — the
  peak-memory bottleneck on wide backfills.
- Redundant store scans removed across the catalog and Parquet layers.
- The 15s bathymetry layer is written as a spatially tiled Zarr, so a geometry
  reads only the overlapping tiles.

### Changed

- The one-time `backfill_provenance` migration moved out of `ZarrCatalog` into
  `storage/provenance.py`; the method delegates, so the public API is
  unchanged.
- Pipeline log messages deduplicated and clarified.

## [0.3.0] - 2026-06-07

### Changed

- `chunk_dataset` now tiles spatial dims (lat/lon/x/y) to `spatial_chunk`
  (default 256, capped at the dim size) instead of keeping them full-size, so
  point/geometry extraction over a small bbox reads only the overlapping tiles
  instead of decompressing the whole global slice per timestep. The time chunk
  fills the remaining `target_mb` budget. Existing stores keep their layout
  until re-chunked explicitly.
- `Extractor` and `ZarrCatalog` now pad bbox slices by one grid cell, so a
  sub-cell bbox (e.g. a short geometry on a coarse 0.5° grid) falling between
  cell centers still captures surrounding cells instead of yielding an empty
  slice.

### Added

- `scripts/profile_extract_chunking.py` — before/after profiling of the
  spatial-tiling effect on contiguous and sparse (CSV point) reads, without
  modifying the real store.
- `scripts/rechunk_stores.py` — atomically rewrite existing Zarr stores into
  spatial tiles + float32 (dry-run by default; `--apply` to write).

## [0.2.1] - 2026-06-03

### Changed

- Removed the unused `watchfiles` dependency.
- Tuned pipeline logging: dropped noisy per-step DEBUG/INFO lines and added a
  single end-of-run status message (success / dry-run / failure).
- Hoisted invariant base-grid and catalog/grid computation out of per-iteration
  loops in the compile and fronts processing paths.

### Added

- pyright type checking: a `pyright` dev dependency, `pyrightconfig.json`
  targeting the 3.11 floor, and an opt-in `tox -e pyright` environment.

### Fixed

- Several `None`-handling and possibly-unbound-variable bugs caught by pyright:
  `bbox` in the CDS downloader, `bounds` / `data_path` in the extractor, an
  unregistered download source in `PipelineManager`, and a null catalog in
  `compile_default`.
- `KeyError: 'store_root'` from `h2mare catalog` for variables with an empty
  catalog (e.g. `bathy`, `moon`); `ZarrCatalog.summary()` now returns a
  consistent key schema whether or not the catalog has data.

## [0.2.0] - 2026-06-02

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

[0.8.1]: https://github.com/h2ugoparra/h2mare/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/h2ugoparra/h2mare/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/h2ugoparra/h2mare/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/h2ugoparra/h2mare/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/h2ugoparra/h2mare/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/h2ugoparra/h2mare/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/h2ugoparra/h2mare/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/h2ugoparra/h2mare/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/h2ugoparra/h2mare/compare/v0.1.1...v0.2.0
