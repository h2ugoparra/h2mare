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
| `--store-path` | path | `STORE_ROOT` | Override the Zarr store root |
| `--no-convert` | flag | false | Download raw files only; skip Zarr conversion and compile |
| `--no-compile` | flag | false | Convert to Zarr but skip the h2ds compile step |
| `--no-parquet` | flag | false | Skip the Zarr → Parquet conversion step |
| `--dry-run` | flag | false | Plan tasks and log without downloading anything |
| `--no-backup` | flag | false | Skip both backup steps (zarr local copy and Parquet remote copy) |
| `--no-zarr-backup` | flag | false | Skip copying compiled Zarr files to the local backup store |
| `--no-parquet-backup` | flag | false | Skip copying the Parquet output to the remote store |
| `--zarr-backup-dir` | path | `local_store_root` | Override destination for the Zarr backup |
| `--parquet-backup-dir` | path | `STORE_ROOT/parquet` | Override destination for the Parquet backup |

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
| `--store-path` | path | `STORE_ROOT` | Override the Zarr store root |
| `--no-zarr-backup` | flag | false | Skip copying compiled Zarr files to the local backup store |
| `--zarr-backup-dir` | path | `local_store_root` | Override destination for the Zarr backup |

**Examples**

```bash
# Compile all variables (dates inferred)
uv run h2mare compile

# Compile a subset of variables over a specific period
uv run h2mare compile -v sst -v ssh -v mld --start-date 2024-01-01 --end-date 2024-12-31

# Use a custom store path
uv run h2mare compile --store-path D:/GlobalData

# Compile without backing up to local store
uv run h2mare compile --no-zarr-backup
```

---

## `h2mare convert`

Convert already-downloaded raw files to Zarr without re-downloading.

```
uv run h2mare convert [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `-v, --vars` | text (repeatable) | all keys | Variable key(s) to convert |
| `--in-dir` | path | `DOWNLOADS_DIR` | Override the input directory containing raw files |

**Examples**

```bash
# Convert downloaded files for sst and ssh
uv run h2mare convert -v sst -v ssh

# Convert from a custom input directory
uv run h2mare convert -v sst --in-dir /data/raw/CMEMS_SST
```

---

## `h2mare parquet`

Convert compiled Zarr stores to Hive-partitioned Parquet.

```
uv run h2mare parquet [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `-v, --vars` | text (repeatable) | `h2ds` | Variable key(s) to convert |
| `--start-date` | YYYY-MM-DD | inferred | Start of date range |
| `--end-date` | YYYY-MM-DD | inferred | End of date range |
| `--out-dir` | path | `PARQUET_DIR` | Root directory for Parquet output |
| `--store-path` | path | `STORE_ROOT` | Override the Zarr store root |
| `--depth` | float | — | Depth level in metres for depth-aware variables (e.g. `thetao`, `o2`) |
| `--no-parquet-backup` | flag | false | Skip copying the Parquet output to the remote store |
| `--parquet-backup-dir` | path | `STORE_ROOT/parquet` | Override destination for the Parquet backup |

**Examples**

```bash
# Convert the compiled h2ds store (dates inferred)
uv run h2mare parquet

# Convert a specific date range
uv run h2mare parquet --start-date 1998-01-01 --end-date 1998-12-31

# Convert without backing up to remote store
uv run h2mare parquet --no-parquet-backup

# Write to a custom output directory
uv run h2mare parquet --out-dir D:/parquet_store
```

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
| `-r, --rows` | flag | false | Print individual catalog rows (filename, dataset, dates, timesteps) |

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

## Variable keys

Valid values for `-v / --vars`:

`sst` `ssh` `mld` `chl` `seapodym` `o2` `fsle` `eddies` `atm-instante` `atm-accum-avg` `radiation` `waves`

See [Variables](variables.md) for descriptions and source details.
