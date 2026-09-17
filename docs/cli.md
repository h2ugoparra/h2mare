# CLI Reference

All commands are run via `uv run h2mare <command> [options]`.

---

## `h2mare run`

Download raw data and convert it to Zarr for one or more variable keys.

```
uv run h2mare run [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `-v, --vars` | text (repeatable) | all keys | Variable key(s) to process |
| `--start-date` | YYYY-MM-DD | inferred | Start of date range. Must be paired with `--end-date` |
| `--end-date` | YYYY-MM-DD | inferred | End of date range. Must be paired with `--start-date` |
| `--store-path` | path | `STORE_ROOT` | Override the Zarr store root for the whole run, including variables that declare their own `store_root` in `config.yaml` |
| `--no-convert` | flag | false | Download raw files only; skip Zarr conversion and compile |
| `--no-compile` | flag | false | Convert to Zarr but skip the h2ds compile step |
| `--no-parquet` | flag | false | Skip the Zarr → Parquet conversion step |
| `--dry-run` | flag | false | Plan tasks and log without downloading anything |
| `--h2ds-zarr-backup` | flag | false | Copy the compiled h2ds Zarr files to the local backup store |
| `--h2ds-parquet-backup` | flag | false | Copy the h2ds Parquet output to the remote store |
| `--h2ds-zarr-backup-dir` | path | `local_store_root` | Override destination for the Zarr backup (only used with `--h2ds-zarr-backup`) |
| `--h2ds-parquet-backup-dir` | path | `STORE_ROOT/parquet` | Override destination for the Parquet backup (only used with `--h2ds-parquet-backup`) |

When `--start-date` / `--end-date` are omitted the pipeline infers the missing date range from the existing store.

The command exits with code `0` if all steps succeed and code `1` if any download, conversion, compile, or Parquet step fails. Errors are logged but the run continues across variables, so a non-zero exit code means at least one step failed — check the log for details.

**Examples**

```bash
# First-time download with explicit dates
uv run h2mare run -v sst --start-date 2021-01-01 --end-date 2021-12-31

# Update an existing store (dates inferred automatically)
uv run h2mare run -v sst

# Multiple variables at once
uv run h2mare run -v sst -v ssh -v mld

# Download only, skip Zarr conversion
uv run h2mare run -v sst --no-convert

# Skip the compile step after conversion
uv run h2mare run -v sst --no-compile

# Validate configuration without downloading
uv run h2mare run -v sst --dry-run

# Back up the compiled h2ds outputs (off by default)
uv run h2mare run -v sst --h2ds-zarr-backup --h2ds-parquet-backup

# Process all configured variables
uv run h2mare run
```

---

## `h2mare compile`

Merge per-variable Zarr stores into the unified h2ds compiled dataset.

```
uv run h2mare compile [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `-v, --vars` | text (repeatable) | all keys | Variable key(s) to include |
| `--start-date` | YYYY-MM-DD | inferred | Start of date range |
| `--end-date` | YYYY-MM-DD | inferred | End of date range |
| `--store-path` | path | `STORE_ROOT` | Override the Zarr store root for the whole run, including variables that declare their own `store_root` in `config.yaml` |
| `--zarr-backup` | flag | false | Copy compiled Zarr files to the local backup store |
| `--zarr-backup-dir` | path | `local_store_root` | Override destination for the Zarr backup (only used with `--zarr-backup`) |

**Examples**

```bash
# Compile all variables (dates inferred)
uv run h2mare compile

# Compile a subset of variables over a specific period
uv run h2mare compile -v sst -v ssh -v mld --start-date 2024-01-01 --end-date 2024-12-31

# Use a custom store path
uv run h2mare compile --store-path /data/h2mare

# Compile and back up to local store (off by default)
uv run h2mare compile --zarr-backup
```

---

## `h2mare convert`

Convert already-downloaded raw files to Zarr without re-downloading.

```
uv run h2mare convert [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `-v, --vars` | text (repeatable) | **required** | Variable key(s) to convert |
| `--in-dir` | path | `DOWNLOADS_DIR` | Override the input directory containing raw files |
| `--start-date` | text | all raw files | Restrict the conversion to this window (`YYYY-MM-DD`). Must be given with `--end-date` |
| `--end-date` | text | all raw files | End of the conversion window (`YYYY-MM-DD`). Must be given with `--start-date` |

Unlike `run`, `compile` and `parquet`, `-v` is required here — `convert` never
defaults to all configured variables. Omitting the dates converts every
downloaded raw file the variable's `pattern` (and `raw_include`) matches.

**Examples**

```bash
# Convert downloaded files for sst and ssh
uv run h2mare convert -v sst -v ssh

# Convert from a custom input directory
uv run h2mare convert -v sst --in-dir /data/raw/CMEMS_SST

# Re-convert a single period from raw files already on disk
uv run h2mare convert -v eddies --start-date 2024-01-01 --end-date 2024-12-31
```

---

## `h2mare parquet`

Convert compiled Zarr stores to Hive-partitioned Parquet.

```
uv run h2mare parquet [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `-v, --vars` | text (repeatable) | `h2ds` | Variable key(s) to convert. Cannot be combined with `--add-var` |
| `--add-var` | text (repeatable) | — | Merge new columns into the existing h2ds Parquet store (see below) |
| `--start-date` | YYYY-MM-DD | inferred | Start of date range |
| `--end-date` | YYYY-MM-DD | inferred | End of date range |
| `--out-dir` | path | `PARQUET_DIR` | Root directory for Parquet output |
| `--store-path` | path | `STORE_ROOT` | Override the Zarr store root for the whole run, including variables that declare their own `store_root` in `config.yaml` |
| `--depth` | float | — | Depth level in metres for depth-aware variables (e.g. `thetao`, `o2`) |
| `--parquet-backup` | flag | false | Copy the Parquet output to the remote store |
| `--parquet-backup-dir` | path | `STORE_ROOT/parquet` | Override destination for the Parquet backup (only used with `--parquet-backup`) |

**Examples**

```bash
# Convert the compiled h2ds store (dates inferred)
uv run h2mare parquet

# Convert a specific date range
uv run h2mare parquet --start-date 1998-01-01 --end-date 1998-12-31

# Convert and back up to remote store (off by default)
uv run h2mare parquet --parquet-backup

# Write to a custom output directory
uv run h2mare parquet --out-dir D:/parquet_store

# Add new variable columns to an existing h2ds Parquet store
uv run h2mare parquet --add-var thetao
uv run h2mare parquet --add-var thetao --add-var o2
```

**Adding variables to an existing Parquet store (`--add-var`)**

When the h2ds Parquet store already exists and a new variable has been compiled into h2ds, `--add-var` merges that variable's columns into every existing monthly partition without reprocessing all other variables.

```bash
# thetao was compiled into h2ds; add thetao_100…thetao_1000 to the Parquet store
uv run h2mare parquet --add-var thetao
```

Each var_key passed to `--add-var` is resolved to its `compiled_vars` list in `config.yaml` (e.g. `thetao` → `[thetao_100, thetao_200, thetao_500, thetao_1000]`). Only those columns are read from the h2ds Zarr, and a `FULL OUTER JOIN` on `(time, lat, lon)` adds them to each partition. Cannot be combined with `-v`.

---

## `h2mare parquet2zarr`

Rebuild per-period Zarr files from a Hive-partitioned Parquet store — the inverse of `h2mare parquet`. Reads the long-format Parquet rows, pivots each time chunk back to a gridded dataset, and writes per-period `.zarr` files (one per year by default) using the pipeline's standard snap → chunk → append path. This is the config-free [`convert_parquet_to_zarr`](api/adhoc_converters.md#convert_parquet_to_zarr) function — no `var_key` or `config.yaml` is consulted.

```
uv run h2mare parquet2zarr PARQUET_ROOT OUT_DIR [OPTIONS]
```

| Argument / Option | Type | Default | Description |
|---|---|---|---|
| `PARQUET_ROOT` | path | — | Root of the Parquet store to read (required) |
| `OUT_DIR` | path | — | Directory to write per-period `.zarr` files into (required) |
| `--name` | text | `data` | Identity label used in the output filename (`{name}_{label}.zarr`) |
| `--start-date` | YYYY-MM-DD | store start | Start of date range (must be paired with `--end-date`) |
| `--end-date` | YYYY-MM-DD | store end | End of date range (must be paired with `--start-date`) |
| `-v, --vars` | text (repeatable) | all | Variable column(s) to read |
| `--file-period` | `month` \| `year` | `month` | Read/append chunk granularity (memory control). `--time-resolution` is accepted as the former name |
| `--date-format` | `year` \| `yearmonth` \| `date` | `year` | Output file granularity |
| `--layout` | `timeseries` \| `map` | `timeseries` | Zarr chunk layout — extraction vs interactive display |

**Examples**

```bash
# Rebuild the whole store into per-year Zarr files
uv run h2mare parquet2zarr D:/parquet_store/h2ds D:/rebuilt_zarr --name h2ds

# Restrict to a date range and a subset of variables
uv run h2mare parquet2zarr D:/parquet_store/h2ds D:/rebuilt_zarr \
    --name h2ds --start-date 2021-01-01 --end-date 2021-12-31 -v sst -v ssh

# Write one file per month instead of per year
uv run h2mare parquet2zarr D:/parquet_store/h2ds D:/out --date-format yearmonth
```

The rebuilt store is the faithful inverse of what was written to Parquet (`time × lat × lon`, Float32, midnight-normalized time), not necessarily byte-identical to an original NetCDF-derived store. Re-running over an existing output appends/merges via the standard overlap resolution.

---

## `h2mare catalog`

Inspect `ZarrCatalog` metadata for one or more variable keys without opening any Zarr files.

```
uv run h2mare catalog [VAR_KEY] [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `VAR_KEY` | text | — | Variable key to inspect (e.g. `sst`, `ssh`) |
| `-a, --all` | flag | false | Show summary for all configured variables |
| `-r, --rows` | flag | false | Print individual catalog rows (filename, dataset, dates, timesteps, extent) |

The `BBox` line is the extent the files actually hold, unioned across the store
— cell centres, so it sits half a grid cell inside the requested edges. A
`BBox (cfg)` line appears only when the bbox configured for the variable differs
by more than that (or when the store is empty, where it is all there is).

**Examples**

```bash
# Summary for SST
uv run h2mare catalog sst

# Summary for all configured variables
uv run h2mare catalog --all

# Show individual catalog rows
uv run h2mare catalog sst --rows
```

---

## `h2mare audit`

Report days that are missing from the **middle** of a store's own time span.

Every other coverage mechanism is a frontier — a start and an end. A year
holding January 1st and December 31st reports itself complete however much is
missing in between, which is how `AVISO_FSLE` 1999 shipped with 128 of 365 days
and passed every check.

The default check reads time **coordinates** only, never data: the full
production store takes about a minute, so there is no cache and no incremental
mode. Exits non-zero when anything is found, so it can gate a scheduled run.

```
uv run h2mare audit [VAR_KEY] [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `VAR_KEY` | text | — | Variable key to audit (e.g. `sst`, `fsle`) |
| `-a, --all` | flag | false | Audit every configured variable |
| `--values` | flag | false | Also report present-but-unusable slices (empty or single-valued). Reads data — much slower; pair with `--since`. Shows a per-file progress bar |
| `--parquet` | flag | false | Check the Parquet store for wholly-null columns, from footer statistics. Reads no data |
| `--show-ok` | flag | false | List variables that passed too |
| `--known` | flag | false | List the days excluded via each variable's `known_gaps` config entry, rather than only counting them |
| `--since` | date | — | Bound `--values` to dates on or after this. The value scan is disk-bound over the whole store — `chl` alone is 97 GB — so auditing more than one variable without it is an hours-long job |

**What it does and does not flag**

A day *absent from the time axis* is a pipeline defect and is reported. A day
*present but entirely null* is a genuine source gap — `chl` has three in 1999
alone — and is only reported under `--values`. Keeping those separate is what
lets the default check stay enabled: one that flagged `chl` every year would be
switched off, and then it would protect nothing.

A store whose tail stops short of today is ordinary provider lag and is never
flagged; only the interior of a store's own span is checked.

**What the verdict claims**

The closing line names its own scope, because "no gaps found" would otherwise
read as a statement about the whole store when it may have covered one variable,
or one variable since 2020:

```
Auditing 16 key variable(s) — axis check
  [SKIP] moon             no store directory (nothing downloads this variable)

Checked 15 of 16 key variable(s), axis check — no gaps found.
```

*Key* variables: a `var_key` such as `eddies` expands to ~15 columns, so a bare
count would read as those.

A variable that could not be checked is never one of the passes. `moon` is
computed at compile time and has no store of its own, so it is skipped and does
not gate. A **downloaded** variable with no store is a finding — that is what an
unmounted drive or a wrong `STORE_ROOT` looks like, and it would otherwise let
the command exit 0 having opened nothing at all. So is a variable whose audit
raised: the check did not run, and nobody knows why.

Days the provider never published are a third case: they leave an axis hole
that no amount of re-running can fill. Record them in `known_gaps` (see
[Configuration](configuration.md)) and the audit excludes them, reporting how
many it suppressed so the list stays visible rather than silently growing.
`--known` prints the dates themselves:

```
uv run h2mare audit --all --known

  [OK]   fsle             29 file(s)  (1 known source gap(s) excluded)
           known source gaps (never published upstream):
             2025-06-02
```

A variable with suppressed days prints under `--known` even without
`--show-ok`, so the list is visible on an otherwise clean store.

**Cost of `--values`**

The axis check reads coordinates and takes about a minute for the whole store.
`--values` reads every cell, and the stores are far too large for page cache —
`chl` alone is 97 GB across 29 files, roughly 9 minutes on its own. Bound it:

```bash
# 35 seconds instead of 9 minutes
uv run h2mare audit chl --values --since 2025-01-01
```

A day absent from the axis and a day present-but-empty need opposite responses,
so the summary gives advice per finding type: re-download for the first,
confirm-then-`known_gaps` for the second. Re-running never fills a day the
provider did not publish.

**Examples**

```bash
# Every configured variable
uv run h2mare audit --all

# One variable
uv run h2mare audit fsle

# Also look for present-but-unusable slices
uv run h2mare audit fsle --values

# Parity check on the Parquet store
uv run h2mare audit --parquet
```

Repair a reported gap through the normal path — re-run the download and convert
for those dates, then `compile`, then rewrite the affected Parquet months:

```bash
uv run h2mare run -v fsle --start-date 2025-06-02 --end-date 2025-06-02
uv run h2mare compile
uv run h2mare parquet --start-date 2025-06-01 --end-date 2025-06-30
```

---

## Variable keys

Valid values for `-v / --vars`:

`sst` `ssh` `mld` `chl` `seapodym` `o2` `fsle` `eddies` `atm-instante` `atm-accum-avg` `radiation` `waves`

See [Variables](variables.md) for descriptions and source details.
