# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **SDM front layers** (`front_layers`, the successor of `boa_fronts`; design
  and evidence in `plans/front-layers.md`). A Canny-style detector at the scale
  the L4 products resolve: Gaussian smoothing in km, the gradient per km with
  the cos(lat) metric, thinning and hysteresis, chl on log10, and sst masked
  where `analysis_error` > 0.84 K. Writes `{name}_grad` (gradient magnitude) and
  `{name}_ffreq30` (share of the last 30 assessed days with a front within
  12.5 km) to the native store and h2ds, plus the daily mask `{name}_front` to
  the native store only. The frequency is seeded from stored masks across
  periods and recomputed after a rewrite of past days. The shipped config
  declares them for `sst` and `chl`; the stores gain them with the backfill.
  Compile drops a var_key's native-only variables.
- `scripts/backfill_front_layers.py`: adds the front layers to an existing
  store from its own field, file by file in date order so each year's
  frequency is seeded with the previous year's masks (dry run by default).
  Store reads behind the detection retry a transient `OSError` (the `D:`
  drive's intermittent EINVAL).
- `front_layers` entries can declare `persistent_distance: {window, min_frequency}`
  to also write `{name}_pdist{window}`, the great-circle distance (km) to the
  nearest pixel whose front frequency is at least `min_frequency`. Optional and
  off by default; no default threshold. 0.5 is the measured recommendation
  (`plans/front-layers.md` §4.4).

### Removed

- **BOA front distances** (`sst_fdist`, `chl_fdist`): `boa_fronts` is gone
  from the shipped config, and with it the two columns from `compiled_vars`.
  Every stored value predated the 2026-09 distance-metric fix, so the native
  stores were cleared of them (`scripts/drop_variables.py`, which deletes named
  variables from a var_key's store and refuses one the config still produces).
  Breaking for anyone reading them; the front layers replace them. h2ds and the
  Parquet store still hold them until rebuilt. The `boa_fronts` key and its
  code remain for now.

## [0.9.0] - 2026-09-30

### Breaking

- **Breaking (install):** the plotting stack (cartopy, matplotlib, plotly,
  statsmodels, ipython) moved from the core dependencies to a `viz` extra. A
  plain `pip install h2mare` now runs the whole pipeline without it; plotting
  (`h2mare.utils.plot`, `ParquetIndexer.plot`) needs `pip install "h2mare[viz]"`
  and says so when it is missing. `uv sync --dev` still installs it.
- **Breaking (config):** the bathy entry's `data_file` / `data_file_hires` are
  replaced by named `layers` (`15s`, `60s`, `0.25deg` → file name) plus
  `compile_layer` and `extract_layer`. An old config fails at load. Config is
  now the only place a layer's file name lives.
- **Breaking (values): compile regrids by resolution.** Every variable used to be
  put on the h2ds grid by point interpolation, which, onto a coarser grid, reads
  the 2×2 source cells around each target centre and ignores the rest (4 of 25
  for sst at 0.25°). A variable finer than the grid is now the area-weighted mean
  of every source cell in the target cell, skipping NaN, so coastal cells keep a
  value; same-resolution and coarser variables stay linear. Values in h2ds change
  on recompile. `regrid:` per variable overrides the method (`nearest` for the
  eddy identity columns). See `docs/api/compiler.md#regridding`.
- **Breaking (API/config): the compile grid is declared in config.** `cells_per_degree`
  and `values_at` on the `h2ds` entry replace `Compiler.run(dx=..., dy=...)`, which
  is gone; the eddy rasterisation reads the same keys. A store holding another
  grid (or other depth levels) is refused rather than unioned with.
- **Breaking (values): distances are measured on the sphere.**
  `haversine_min_distance_kdtree` indexed (lat, lon) as a plane, so a degree of
  longitude counted as 111 km everywhere: east-west distances were overstated by
  1/cos(lat) (2× at 60°N). Eddy and front distances computed after this release
  differ from stored ones; stores are corrected by recomputing (see Upgrade notes).
- **Breaking (config): BOA front layers are declared in config.** The thresholds
  moved from `fronts.py` into `boa_fronts` (keyed by output name, threshold in
  the source's units). A config without `boa_fronts` writes no front layers; the
  shipped config declares `sst_fdist` and `chl_fdist` as before.

### Added

- **Per-variable depth levels:** `depth_levels: {thetao: [0, 50]}` publishes
  `thetao_0`, `thetao_50`, so a store may mix 2-D and 3-D variables with their own
  levels; extraction can choose levels per request
  (`var_dict={"dyn_rep": {"thetao": [0, 50]}}`). The older `compile_depth_slices` /
  `extract_depth_slices` lists still work.
- **`derived_vars`:** convert-time rolling std and kinetic energy declared in
  config by their inputs (`sst_std`, `adt_std`, `sla_std`, `gke` moved there from
  the processors), optionally at chosen depths only.
- Every command logs, before it does anything, which project root it resolved
  and how (`H2MARE_ROOT`, a config.yaml above the working directory, or the
  library fallback), whether a config.yaml is there, and the `STORE_ROOT`. A
  user-wide `H2MARE_ROOT` silently points any checkout at another project's
  config and stores; now the first line of the log says so.
- `H2MARE_ROOT` set in `.env` is ignored with a warning and kept out of the
  environment. It never moved the process reading the file (the root is chosen
  before `.env` is found), but once loaded, spawn workers inherited it and
  could resolve another project's root than their parent.
- The **process** pools — BOA front detection and the eddy rasterisation — are
  now capped by the host's CPU count, and by an optional `H2MARE_MAX_WORKERS`
  ceiling in `.env`. Front detection defaulted to 10 spawn workers whatever the
  machine had: on a 4-core box (or a CI runner) that started six workers it
  could not run, each re-importing h2mare and receiving its own copy of every
  task. The per-site defaults and each variable's `n_workers` are unchanged —
  they propose, the machine caps. A malformed `H2MARE_MAX_WORKERS` is warned
  about and ignored, since `Settings()` runs on any import. The thread pools
  (AVISO FTP downloads, `parquet2csv`, geometry extraction) are deliberately
  left alone: they wait on the network or the disk, where more threads than
  cores is the point.
- `source_renames` in a variable's config entry maps its source variable names
  onto the names it publishes (`{analysed_sst: sst}`), closing the loop between
  `source_vars` and `compiled_vars` in the one place both are declared. Applied
  at convert time before the registered processor, so processors, `boa_fronts`,
  `derived_vars` and the CF attrs all see config's names. Config load refuses a
  rename onto a name the var_key does not publish, onto one `derived_vars` or
  `boa_fronts` also writes, two sources onto one name, a self-map, and a
  `compiled_vars` still listing a renamed-away source; a rename naming a
  variable the raw files do not hold fails the convert by name.
- `scripts/recompute_fronts.py` rewrites the front-distance layers from the
  stored field, without the raw files; `scripts/rechunk_store.py` rewrites a
  store in the chunk layout the pipeline would choose now.
- Convert, compile, parquet and CMEMS subset downloads log chunk progress (i/n).

### Changed

- Bathy extraction reads one layer for points and geometries alike, chosen by
  `extract_layer` or `Extractor(bathy_layer=...)`, instead of the input type
  deciding the grid (0.25° for csv, 15s for shp).
- A geometry's `bathy_std` is now the polygon mean of the layer's stored std,
  the same estimator as for points and as `sst_std`/`adt_std`, rather than a
  std of depth within the polygon.
- The `sst`, `chl` and `mld` renames moved out of their convert-time processors
  into `source_renames` in config. `process_mld` only renamed `mlotst`, so it
  and its registry entry are gone — a variable needing nothing but a rename now
  needs no Python at all. No store changes: the names on disk are the same, and
  nothing needs re-converting. A caller reusing a registered processor through
  `convert_netcdf_to_zarr` must do the rename itself, as that path is
  config-free.
- `scripts/bathymetry.py` builds a 60s layer next to the 15s one, both as tiled
  Zarr (`etopo2022_<res>_…_bathy-std.zarr`) holding `bathy` and a 3×3 rolling
  `bathy_std` (the entry's `derived_vars`), with the ETOPO source's global and
  `z` attributes carried over. `--layers` builds a subset.
- `archive_raw` is optional and defaults to `false` (delete raw files once
  converted). Entries that set it are unaffected; set `true` where raw files
  are costly to download again (the shipped config does for `fsle`, `eddies`).
- The release workflow runs the test suite on the tagged commit, refuses a
  tag that does not match `pyproject.toml`'s version, and installs the built
  wheel into a clean environment (resolved from its own metadata, no lock) and
  imports it before publishing. Its actions are pinned to commit SHAs, and only
  the publish job may mint a PyPI token. CI's branch-name check reads the branch
  through the environment rather than pasting it into the script.
- Stores keep one depth level per chunk, so reading a level no longer
  decompresses them all (a 23-level store's surface read: 0.44 s → 0.04 s).
- The eddies convert works a month at a time through a staging store, loads the
  raw atlas once, and runs one nearest-eddy search per day: memory is bounded by a
  month instead of a year (the old path ran out of memory), and a year converts in
  ~330 s instead of ~430 s. Its grid is now 1/12° on grid lines (`cells_per_degree:
  12`).
- The conservative regrid applies its weights sparsely (13× faster than a dense
  contraction, bit-identical values).

### Fixed

- **Extraction returned a neighbour's value where it had none.** The point and
  geometry engines took the nearest cell and time step with no limit, so a
  sample outside the grid got the edge cell's value (a point at 25°E read
  22.9 °C off a store ending at 10°E), and a date missing from the store got
  the adjacent day's. Such samples are now `NaN`, with a warning counting them:
  a point must fall inside its cell, and a time within half a step of its own.
  The bathy point path gets the same check.
- **Sub-daily samples against a daily store read the next day after noon.** A
  daily step is stamped at midnight, so 23:30 on June 15 was nearer June 16's
  stamp and took June 16's value — about half the rows of a typical GPS track.
  Against a daily store each sample now takes the day it falls in, as the
  compiled-store path already did. The store's cadence comes from its
  `time_step`; `extract_from_dataset` reads it off the dataset's axis.
- **Every coastline was a front.** BOA filled cells without data with 0, so
  land read as a jump the size of the field (~18 °C for sst) and every sea cell
  touching it was a front at distance 0 — 99% of coast-adjacent sst cells on
  2024-06-15, where the next cell in held 77%. Cells without data are now
  filled from their nearest valid neighbour, and no front is placed on one.
  Offshore detection is unchanged, pixel for pixel.
- **A day with no data wrote 20,015 km everywhere.** With no fronts to measure
  to, every sea cell got half the Earth's circumference; chl's 11 all-null days
  (1998–2002) carry it on disk. It is NaN now — `haversine_min_distance_kdtree`
  returns NaN for an empty target set — and a distance is also NaN wherever the
  source field itself is, rather than wherever `global_land_mask` calls land.
- **An eddy day that failed to rasterise was dropped silently.** The worker
  logged the exception and returned nothing, and the period was written one
  day short, logged as SUCCESS for its full length, with provenance spanning
  the hole — so coverage moved past it and nothing retried it. A failure now
  raises, naming the day and eddy type, and the period is not written. Days the
  atlas has no observations for are still skipped, now by date in a warning.
- **Regridding with `nearest` carried edge values past the source.** A target
  cell beyond the source's extent took the edge cell's value; it is now NaN, as
  under `conservative` and `linear`. The shipped eddies store covers the whole
  compile bbox, so h2ds is unchanged.
- A depth level far past the end of a store's depth axis (more than one level
  spacing beyond it) is still read from the last level, but now logs a warning
  naming the depth actually used; it was labelled with the requested depth and
  said nothing. The shipped `thetao_1000` (read from 902 m) is within that
  spacing and stays silent.
- **A failed AVISO directory listing was treated as the whole dataset.** Listing
  errors were logged and the rest of the tree returned, so a dropped connection
  on one per-year directory ended the REP range a year early (days fetched from
  NRT instead) and left that year's files unqueued, with the run reported as a
  success. A directory that cannot be listed now fails the listing, and so the
  variable's download.
- **A climatology short of the ERA5 grid cropped the Ekman features.** The
  anomaly and upwelling-event counts align with the day-of-year and p90
  climatologies by xarray's default inner join, so a climatology built for a
  smaller bbox (its file name is fixed, whatever `bbox` says) shrank them to
  the overlap without a word, and labels a float apart matched nothing at all.
  Compile now refuses a climatology that does not cover the data's grid, naming
  the file, and puts one that does onto the data's own labels. The shipped
  climatologies match the shipped grid exactly, so nothing changes there.
- Front detection workers start by spawn rather than fork, which deadlocked on
  Linux (CI hung for six hours).
- Staging directories left by an interrupted eddies run are cleared.
- A store carrying a `missing_value` that contradicts its `_FillValue` (chl's
  inherited -999) can be rewritten.
- The REP-updated check, the compiler's catalogs and the Extractor's native reads
  use the caller's own `app_config` and store root, not the process-wide ones, so
  a project reading another project's var_keys works.
- `ZarrCatalog.open_dataset(dates=...)` accepts a `DatetimeIndex`, `Series` or
  array; compile opens Zarr bathy layers.
- `plot_records_on_field` plots depth-sliced columns (`thetao_0`) from the native
  store, so it works in a project that never compiles.
- Downloaders say when a requested range is outside the provider's coverage,
  instead of "already up to date".
- copernicusmarine log lines are no longer printed twice.
- The `compiled_vars` check prints the corrected list when depth columns
  disagree.

### Upgrade notes

- Install `h2mare[viz]` if you plot.
- Move the bathy entry to `layers` / `compile_layer` / `extract_layer`
  (`docs/configuration.md#the-bathy-key`); an old config fails at load.
- Recompile h2ds (`uv run h2mare compile`) and refresh Parquet to pick up the new
  regridding; values change, most near coasts and for fine-grid sources.
- The front-distance layers on disk predate the coastline, empty-day and sphere
  fixes. Rewrite them with `scripts/recompute_fronts.py sst chl --apply` (no raw
  files needed), then compile and parquet. A replacement for the BOA layers is
  designed in `plans/front-layers.md`.
- An eddies store converted before the sphere fix holds planar distances;
  re-convert it.

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
