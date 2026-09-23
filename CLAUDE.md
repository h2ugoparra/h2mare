# Project: h2mare

## Project Overview

A pipeline for downloading and preprocessing multi-source oceanographic and atmospheric data into analysis-ready formats.

## Architecture

**Download → Convert → Compile → Index → Visualize**, with an optional **Extract** step for point or geometry inputs.

```text
h2mare/
  ├── config.py / models.py / types.py    # Settings + msgspec config; runtime types (DateRange, BBox)
  ├── validators.py                        # Shared validation (validate_var_key, …) reused across packages
  ├── pipeline_manager.py                  # Orchestrates Download → Convert → Compile from config + registries
  │
  ├── cli/                  # Typer commands: run, convert, compile, parquet, catalog
  ├── downloader/           # Source fetchers (CMEMS, AVISO, CDS) selected via registry.py → data/raw/downloads/
  ├── format_converters/    # netcdf2zarr (regrid → 0.25°/daily), zarr2parquet, parquet2csv, zarr_map_export
  ├── processing/           # Per-var preprocessing; compiler.py merges → h2ds; core/ holds source transforms
  ├── storage/              # zarr_catalog (facade) / _index (resume index) / _reader (open_dataset); parquet_store (write) / _indexer (API) / _catalog (read); coverage (date-range resolution)
  └── utils/                # spatial (grids/masks), labels, logging, paths, datetime_utils
```

### Registry pattern

Per-variable behavior is selected by `var_key` through three registries; the right way to add a new variable
is to register it, not to branch inside the pipeline:

- `downloader/registry.py` (`DOWNLOADER_REGISTRY`) — source key → downloader class.
- `processing/registry.py` — `var_key` → **convert-time** processor (NetCDF→Zarr step). Unregistered variables pass through unchanged.
- `boa_fronts` (config, applied by `processing/core/fronts.py::apply_boa_fronts` right after that processor and *before* `derived_vars`) — Belkin–O'Reilly front-distance layers keyed by output name (`sst_fdist`, `chl_fdist`), each naming its `source` and a `threshold` **in that source's own units**, so one store can publish a layer per field. Detection is eager: it stages a month at a time under `INTERIM_DIR` (`.{var_key}_*.zarr.stage`) and hands the layer back lazily, which is why `_process_period` clears the staging in its `finally` and `run()` sweeps what a killed run left. Daily stores only. A second algorithm gets its own config key and spec, not a mode flag — the threshold is BOA's.
- `derived_vars` (config, applied by `processing/derived.py` right after that processor) — rolling std / kinetic energy layers named by their inputs, not by var_key (`sst_std`, `adt_std`, `sla_std`, `gke` live there). Prefer it to hardcoding a derived layer in a processor.
- `processing/compiler_registry.py` (`COMPILE_PROCESSORS`) — `var_key` → **compile-time** processor. Unregistered variables use `_compile_depth_var` when their config declares depth levels, else `compile_default` (open catalog, interpolate to base grid; refuses a store that still has a depth axis).
- Depth levels are per store variable (`depth_levels: {thetao: [0, 50]}` → `thetao_0`, `thetao_50`), read only through `models.depth_levels_for`, and sliced by `xarray_helpers.select_depth_levels`. `depth_range` is the unrelated continuous download band. The older `compile_depth_slices`/`extract_depth_slices` lists mean `{var_key: [...]}`. Extraction slices only the variables requested, and a run may choose levels itself (`var_dict={"dyn_rep": {"thetao": [0, 50]}}`) — native stores only; see `docs/api/extractor.md#depth-levels`.

`Extractor` (`processing/extractor.py`) is a standalone analysis tool, not part of the run/convert/compile flow.

### Config & resume

`config.yaml` (variables, dataset IDs, bbox) and `.env` (`STORE_ROOT`, AVISO creds) must both live in the working
directory, or set `H2MARE_ROOT` to point at them. When dates are omitted, the pipeline infers what is missing from
`ZarrCatalog` coverage and only fetches/processes the gap — this is what makes partial runs resumable.

A variable's store is `<root>/<local_folder>/`, and the root is resolved by `utils/paths.py::store_root_for`:
`--store-path` > the variable's own `store_root` in config.yaml > `STORE_ROOT` > `ZARR_DIR`. Declaring no
`store_root` anywhere is the shipped setup and resolves exactly as it always has. Steps holding a root of their
own (`PipelineManager.store_root`, `Compiler.remote_store_root`) pass it as `store_root_for`'s *default*, never
as the answer. Only the Zarr stores follow it — downloads, `STORE_ROOT/parquet` and `Climatology/` do not.

## Tech Stack

Python 3.11+. Key libraries: `xarray`/`dask` (lazy N-D arrays), `zarr` (chunked store), `polars`/`pyarrow`/`duckdb` (columnar data), `geopandas`/`rioxarray`/`cartopy` (geospatial), `copernicusmarine`/`cdsapi` (data sources), `typer` (CLI), `msgspec` (config), `plotly`/`matplotlib` (viz). Dev: `uv`, `ruff`, `pytest`, `tox`.

## Commands

```bash
# Install / sync dependencies
uv sync
uv sync --dev   # include dev dependencies (pytest, ruff)

# Run the pipeline
uv run h2mare run                                                        # all variables; dates inferred from store
uv run h2mare run -v sst --start-date 2021-01-01 --end-date 2021-12-31   # explicit range

# Standalone pipeline steps
uv run h2mare convert -v sst                                             # convert downloaded raw data to zarr (-v required)
uv run h2mare compile                                                    # merge Zarr stores; dates inferred
uv run h2mare parquet                                                    # Zarr → Parquet; dates inferred
uv run h2mare parquet2zarr                                               # rebuild per-period Zarr from Parquet
uv run h2mare catalog sst                                                # inspect ZarrCatalog metadata

# Audit the stores for silently-missing days (exits non-zero on findings)
uv run h2mare audit --all                                                # axis check, whole store, ~1 min
uv run h2mare audit chl --values --since 2020-01-01                      # also read data for empty days (slow)

# Tests
uv run pytest tests/
uv run pytest tests/ -k "test_name"

# Lint / format
uv run ruff check --fix h2mare/
uv run ruff format h2mare/
```

## Pipeline semantics & gotchas

- `run -v X` compiles **only X's columns** into h2ds; other lagging variables catch up on the next full `uv run h2mare compile` (no `-v`).
- Store repair: explicit dates re-read **all** variables and rewrite affected partitions wholesale — `uv run h2mare parquet --start-date ... --end-date ...`. Prefer whole-month windows.
- Write-path merge semantics are deliberate and pinned by regression tests (`tests/test_storage.py`, `tests/test_parquet_store.py`): incoming data wins where it has rows (even when null); stored values survive outside its window; time-less statics (bathy) come from the fresh side; tails and absent variables are preserved. Read those tests before changing `storage.py::_append_data` or `parquet_store.py::resolve_dims_overlap`.
- Extraction cadence: `Extractor` takes two independent args — `time_cadence` (`auto`/`daily`/`hourly`) reads `time_col`; `read_from` (`auto`/`native`/`compiled`) picks the store. Under `auto`: a daily store answers for itself, an **hourly** one answers only sub-daily input and a date-only query goes to **h2ds** — so extracting those at daily cadence needs a current `compile`. Compile-derived vars absent from an hourly store (ekman chain, `wind_*`) are always read from h2ds and broadcast per day, even under `read_from="native"`, because converting the same var_key daily writes them natively. A *daily* store missing what it publishes raises instead — routing is for what the design puts elsewhere, not for holes. Units differ between the two sources (`msl` is Pa native, hPa in h2ds). See `docs/api/extractor.md#cadence`.
- Extraction checkpoint: `INTERIM_DIR/extraction_checkpoint.feather` survives only a *failed* run, and lives at one fixed path. It carries a fingerprint of the input, and a checkpoint written for a different one is discarded rather than resumed — otherwise a same-shape frame had the previous run's rows replayed onto it (`ensure_row_id` keys positionally, so equal-length frames align perfectly). A fingerprint match still cannot tell "resume" from "re-run after a fix", so delete `extraction_checkpoint.*` when the *code or config* changed rather than the input.
- Geometry `_std` columns: the shp engine reduces a clip with `.mean()` only (`extractor.py::_extract_geometry`), so `sst_std`/`adt_std`/`sla_std` are the **polygon-mean of a stored layer**, not a std computed within the polygon — deliberate, since a within-polygon std is size-dependent (a polygon touching one cell gives 0 or NaN) and would not compare across rows. `bathy_std` follows the same rule: each bathy layer (`layers` in config, built by `scripts/bathymetry.py`) stores its own std — a 3×3 rolling std at 15″/60″, the within-cell std of 15″ cells at 0.25° — and `Extractor(bathy_layer=...)` / `extract_layer` picks the layer for csv and shp alike. The layers are not on a common scale either (`sst_std` is a 3×3 window at 0.05°, sub-cell; `adt_std`/`sla_std` 3×3 at 0.125°, *wider* than the 0.25° cell), so don't compare their magnitudes across variables. See `docs/api/extractor.md#standard-deviation-columns`.
- Axis drift: the same axis written on different occasions can disagree in the last float bits. Each file stays monotonic alone, so it only shows up when `open_mfdataset(combine="by_coords")` compares the arrays exactly — as *"does not have monotonic global indexes along dimension lon"*, or a silently doubled axis, for the axis being concatenated along; as *"cannot align objects with join='exact' … 'depth'"* for one that is not. `ZarrReader` snaps agreeing axes onto the earliest file's and warns; anything coarser still raises. Tolerances are per coordinate in `zarr_reader.AXIS_SNAP_TOL` because the units differ — `1e-9` for lat/lon in degrees, `1e-3` for depth in metres (o2's 2023+ files sit one float32 ULP, ~6e-5 m, off the older ones). `scripts/repair_axis_drift.py` imports that map rather than restating it. Repair with `uv run python scripts/repair_axis_drift.py <var_key> --apply` (dry run by default, rewrites coordinate arrays only). `--all` surveys every var_key.
- Ragged variable sets: h2ds files need not all carry the same variables — `run -v X` compiles only X's columns, and a source that has not reached the current year (seapodym, `npp`/`zeu`/`zooc`) leaves that year's file short. `combine_by_coords` groups datasets *by their set of data variables* first, so such a store reached the final merge as two cubes over different periods and `join="exact"` refused them: *"cannot align objects … 'time'"*, naming the axis rather than the variables that split it. `ZarrReader._reference_data_vars` takes the union across the files being opened and `_pad_to_union` fills each short file with all-NaN placeholders, so everything combines as one group; the padded names are warned about once per read. Not a store defect — do not "fix" it by recompiling.
- CF metadata: `apply_cf_attrs` (`storage/xarray_helpers.py`) is the single source of variable and coordinate attributes for *both* write paths — convert (native stores) and compile (h2ds) — reading `variable_attrs` plus, for a native store, that var_key's `native_attr_overrides` (`msl` is Pa natively and hPa in h2ds; `tp` is m vs mm; an hourly field drops the `cell_methods` naming a daily reduction). Coordinate attrs are not cosmetic: without them `rio.clip` cannot resolve lon/lat and geometry extraction returns all-NaN. Root attrs (`Conventions`, extents) are computed per file in `storage/provenance.py`, never config. An append never rewrites coordinates or variable attrs, so a store only picks up a table change when it is rewritten — the stores on disk were backfilled once, out of band, in Aug 2026. `tests/test_cf_compliance.py` validates the table against udunits2 and a vendored CF name snapshot.
- Data quirks: `chl` has legitimate all-null days (~1999/2000 — the raw product never published them; the zarr is null too, so they are not backfillable). `seapodym` covers 2025 only.

## ParquetIndexer

Primary interface for reading and writing the Parquet store (`storage/parquet_indexer.py`).

```python
from h2mare.storage.parquet_indexer import ParquetIndexer

idx = ParquetIndexer("path/to/parquet_root")
idx.add_data(df)                                                           # write; resolves overlap via DuckDB
lf = idx.scan(dates=("2021-01-01", "2021-12-31"), bbox=(-10, 30, 20, 50))  # LazyFrame
df = idx.load(dates=["2021-06-01", "2021-07-01"])                          # DataFrame
idx.get_schema(); idx.get_time_coverage(); idx.get_geoextent()
idx.plot.time_series("sst", agg_by="month")
idx.plot.spatial_maps("sst", agg_by="season")
```

Non-obvious behavior: partition writes are atomic (`.tmp_write_YYYY_MM` → rename); Float64 is downcast to Float32
on write; `idx.plot` is a `cached_property` invalidated after `add_data()`.

## Chunk layouts & map export

`chunk_dataset` (`storage/xarray_helpers.py`) takes `layout`: `"timeseries"` (default — time-contiguous, small
spatial tiles; what extraction/`Extractor` reads) or `"map"` (space-contiguous, time pinned to `map_time_chunk`,
default 14; what interactive maps read). Both fill the 32 MB budget along the axis read *contiguously* and minimize
the axis indexed *into* — the asymmetry is deliberate (don't budget-fill time in `map`, it would force a single-day
viz read to decompress many unwanted days). It logs the resulting layout/shape/size on every call.

`export_map_zarr` (`format_converters/zarr_map_export.py`) rewrites a per-period store into a map-chunked **sibling**
(`h2ds` → `h2ds_map`; or any `var_key` / config-free `source_root`). It's a pure projection: lazy split-rechunk,
atomic temp-dir swap, source never modified. The `_map` store is a full duplicate (~h2ds size) — a derived,
rebuildable artifact. `h2ds_map` is written here and read by the **external** interactive-viz app; nothing in
h2mare consumes it, so don't go looking for a reader in this repo (in particular, `plot_interactive_map` in
`utils/plot.py` plots a DataFrame the caller passes in and opens no store). The canonical `h2ds` stays
extraction-chunked. See `docs/api/map_export.md`.

## Git workflow

Follows the global Git Workflow verbatim (see `~/.claude/CLAUDE.md`) — branches, PR-only merges,
conventional commits, branch protection. Only these two are additional here:

- Merging requires 5 green checks on `dev` (`branch-name`, `commit-lint`, `quality`, `tests (3.12)`, `tests (3.13)`) **and** an up-to-date branch: `gh pr update-branch <#> --rebase`, wait for checks, then `gh pr merge <#> --merge --delete-branch`. `main` requires the same minus `branch-name`, which only runs for PRs into `dev`.
- `typecheck (informational)` runs pyright but is not required — it reports the finding count so it cannot grow unnoticed. Promote it once that count reaches zero.
- Bump `pyproject.toml` version + `uv lock` via a `chore/` PR into `dev` *before* the release PR.

## Coding Rules

- **Logging** — use `loguru` (`from loguru import logger`), not stdlib `logging`
- **Paths** — always access paths via `settings.*`; never hardcode
- **`.env`** — `STORE_ROOT` (required); `AVISO_FTP_SERVER`, `AVISO_USERNAME`, `AVISO_PASSWORD` (required for AVISO variables); `H2MARE_ROOT` (optional, overrides project root detection)
- **Types** — use `DateRange`, `BBox`, `DateLike` from `h2mare/types.py`; no raw tuples. Accept plain tuples in public APIs and construct the named type internally.
- **Regression tests** — must fail on unfixed code; verify with `git stash push <src-file>` → run test → `git stash pop`.
- **Test helpers** — `tests/conftest.py:make_grid_df` builds time×lon×lat Polars frames for parquet-layer tests; `_make_ds` helpers in `tests/test_storage.py` build zarr-ready datasets.
