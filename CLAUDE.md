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
  ├── format_converters/    # netcdf2zarr (regrid → 0.25°/daily), zarr2parquet, parquet2csv
  ├── processing/           # Per-var preprocessing; compiler.py merges → h2ds; core/ holds source transforms
  ├── storage/              # zarr_catalog (resume index); parquet_store (write) / _indexer (API) / _catalog (read)
  └── utils/                # date_range, spatial (grids/masks), labels, logging, paths
```

The pipeline flows left-to-right through these packages: downloader/ fetches raw files → format_converters/ +
processing/ regrid and preprocess into per-variable Zarr → processing/compiler.py merges into the unified h2ds →
storage/ indexes it and exposes Parquet for analysis/visualization, all orchestrated by pipeline_manager.py and
driven from cli/.

### Registry pattern

Per-variable behavior is selected by `var_key` through three registries; the right way to add a new variable
is to register it, not to branch inside the pipeline:

- `downloader/registry.py` (`DOWNLOADER_REGISTRY`) — source key → downloader class.
- `processing/registry.py` — `var_key` → **convert-time** processor (NetCDF→Zarr step). Unregistered variables pass through unchanged.
- `processing/compiler_registry.py` (`COMPILE_PROCESSORS`) — `var_key` → **compile-time** processor. Unregistered variables use `compile_default` (open catalog, interpolate to base grid).

`Extractor` (`processing/extractor.py`) is a standalone analysis tool, not part of the run/convert/compile flow.

### Config & resume

`config.yaml` (variables, dataset IDs, bbox) and `.env` (`STORE_ROOT`, AVISO creds) must both live in the working
directory, or set `H2MARE_ROOT` to point at them. When dates are omitted, the pipeline infers what is missing from
`ZarrCatalog` coverage and only fetches/processes the gap — this is what makes partial runs resumable.

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
uv run h2mare convert                                                    # convert downloaded raw data to zarr
uv run h2mare compile                                                    # merge Zarr stores; dates inferred
uv run h2mare parquet                                                    # Zarr → Parquet; dates inferred
uv run h2mare catalog sst                                                # inspect ZarrCatalog metadata

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

## Git workflow

Follows the global Git Workflow (see `~/.claude/CLAUDE.md`).

**Branches**
- `main` — production; stable, deployable, never commit directly.
- `dev` — integration/staging; features land here first.
- `feature/<name>` — one per feature, branched from `dev`, merged back to `dev`.

**Per feature/fix**
- Branch fresh: `git checkout dev && git pull origin dev && git checkout -b feature/<name>`.
- Work, then `git add <file>` → `git commit -m "feat: ..."` → `git push -u origin feature/<name>`.
- Resuming or dev moved? `git pull origin dev` *before* coding more — avoids most conflicts.

**Merge via PR on GitHub (never local merge)**
- Open PR (base `dev`), review the diff, approve, merge.
- Clean up: `git checkout dev && git pull origin dev && git branch -d feature/<name>` and delete the remote branch.

**Release**
- PR `dev` → `main`, review, merge; then `git checkout main && git pull origin main`.
- Optional tag: `git tag -a v1.0.0 -m "..." && git push origin v1.0.0`.
- Bump `pyproject.toml` version + `uv lock` via a `chore/` PR into `dev` *before* the release PR.

**Rules**
- Always branch from the latest `dev` — *pull before you branch*.
- Even small fixes get a branch (`fix/<name>`) — never merge directly.
- Commit with a type: `feat:` / `fix:` / `docs:` / `chore:` / `perf:` / `refactor:`; say what changed and why.
- Branch with a matching prefix: `feature/` / `fix/` / `docs/` / `chore/` / `perf/` / `refactor/`.
- Protect `main` and `dev` (require PR review).
- Merging requires 3 green checks (`branch-name`, `commit-lint`, `quality`) **and** an up-to-date branch: `gh pr update-branch <#> --rebase`, wait for checks, then `gh pr merge <#> --merge --delete-branch`.

## Coding Rules

- **Logging** — use `loguru` (`from loguru import logger`), not stdlib `logging`
- **Paths** — always access paths via `settings.*`; never hardcode
- **`.env`** — `STORE_ROOT` (required); `AVISO_FTP_SERVER`, `AVISO_USERNAME`, `AVISO_PASSWORD` (required for AVISO variables); `H2MARE_ROOT` (optional, overrides project root detection)
- **Types** — use `DateRange`, `BBox`, `DateLike` from `h2mare/types.py`; no raw tuples. Accept plain tuples in public APIs and construct the named type internally.
- **Regression tests** — must fail on unfixed code; verify with `git stash push <src-file>` → run test → `git stash pop`.
- **Test helpers** — `tests/conftest.py:make_grid_df` builds time×lon×lat Polars frames for parquet-layer tests; `_make_ds` helpers in `tests/test_storage.py` build zarr-ready datasets.
