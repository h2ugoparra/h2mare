# Ad-hoc converters

Two config-free functions convert arbitrary files through the pipeline's engine
**without a configured `var_key`**. Use them to process files that are not
registered in `config.yaml` — a one-off download, a manually placed store, or an
externally produced dataset.

They are the standalone counterparts to the `Netcdf2Zarr` and `Zarr2Parquet`
classes: same generic transform and overlap-resolving write path, but driven by
arguments instead of `config.yaml`. The classes remain the right tool for
configured, resumable pipeline runs.

| Function | Module | Converts |
|---|---|---|
| `convert_netcdf_to_zarr` | `h2mare.format_converters.netcdf2zarr` | NetCDF / GRIB → Zarr |
| `convert_zarr_to_parquet` | `h2mare.format_converters.zarr2parquet` | Zarr → Hive-partitioned Parquet |

---

## convert_netcdf_to_zarr

Convert one or more NetCDF/GRIB files to a single Zarr store.

```python
from h2mare.format_converters.netcdf2zarr import convert_netcdf_to_zarr

convert_netcdf_to_zarr("input.nc", "out.zarr", name="adhoc")
```

Applies the same generic prep the pipeline uses —
open → (optional `rename_dims`) → (optional `processor`) → `snap_grid_coords` →
`chunk_dataset` → `write_append_zarr`.

### Parameters

```python
convert_netcdf_to_zarr(
    paths,                # one path, or an iterable of paths (.nc/.grib)
    out_path,             # destination .zarr (appended if it exists)
    *,
    name="data",          # identity label for overlap/logs; need NOT be in config
    processor=None,       # optional Callable[[Dataset], Dataset] applied after rename
    apply_rename=True,    # rename longitude/latitude/valid_time → lon/lat/time
    open_kwargs=None,     # extra kwargs forwarded to xr.open_mfdataset
)
```

| Parameter | Default | Description |
|---|---|---|
| `paths` | — | One path or an iterable of `.nc`/`.grib` files. NetCDF and GRIB may be mixed; the engine is auto-detected from the first file |
| `out_path` | — | Destination `.zarr` path. If it exists, data is appended with the store's standard overlap semantics |
| `name` | `"data"` | Identity label used in write/append logs and overlap resolution. Need not exist in config |
| `processor` | `None` | Optional callable applied after rename, before snap/chunk — the slot a registry processor occupies in `Netcdf2Zarr.process_dataset` |
| `apply_rename` | `True` | Apply `rename_dims`. Set `False` if files already use canonical `lon`/`lat`/`time` names |
| `open_kwargs` | `None` | Extra keyword arguments forwarded to `xr.open_mfdataset` |

Returns the `out_path` written. Raises `FileNotFoundError` if no paths are given.

### Reusing a registry processor

`processor` takes a single-argument callable; wrap a registered processor to
supply its config arguments:

```python
from h2mare.processing.registry import PROCESSORS

convert_netcdf_to_zarr(
    "raw_sst.nc", "sst.zarr", name="sst",
    processor=lambda ds: PROCESSORS["sst"](ds, cfg, "sst"),
)
```

---

## convert_zarr_to_parquet

Convert an arbitrary Zarr store (or list of stores) to a Hive-partitioned
Parquet store.

```python
from h2mare.format_converters.zarr2parquet import convert_zarr_to_parquet

convert_zarr_to_parquet("store.zarr", "data/processed/parquet")
```

Opens the store directly (instead of locating it through a `ZarrCatalog` keyed by
a registered variable), splits the window into memory-sized chunks, and writes
each chunk via `ParquetIndexer.add_data` — the same overlap-resolving write path
the class uses.

### Parameters

```python
convert_zarr_to_parquet(
    zarr_path,                  # one store, or an iterable of stores
    parquet_root,               # destination Parquet store (written directly here)
    *,
    start_date=None,            # defaults to the store's first time step
    end_date=None,              # defaults to the store's last time step
    time_resolution="month",    # "month" | "year" (or TimeResolution); chunk size
    depth=None,                 # required if the store has a depth dim
    variables=None,             # subset of data variables; None = all
    indexer_kwargs=None,        # forwarded to ParquetIndexer (e.g. column names)
    open_kwargs=None,           # forwarded to the xarray open call
)
```

| Parameter | Default | Description |
|---|---|---|
| `zarr_path` | — | One Zarr store path, or an iterable opened together via `xr.open_mfdataset(engine="zarr")` |
| `parquet_root` | — | Destination directory. Unlike the class, no dataset sub-folder is derived — data is written here directly. Existing partitions are appended or JOINed via the indexer's overlap semantics |
| `start_date` | store start | Start of the conversion window (`str` or `pd.Timestamp`) |
| `end_date` | store end | End of the conversion window (`str` or `pd.Timestamp`) |
| `time_resolution` | `"month"` | Granularity of each write batch. Accepts a plain string (`"month"`/`"year"`) or `TimeResolution` |
| `depth` | `None` | Depth level (metres) to select for stores with a `depth` dim; nearest level is chosen. **Required** when the store has a `depth` dim |
| `variables` | `None` | Subset of data variables to read. `None` reads all |
| `indexer_kwargs` | `None` | Extra kwargs for `ParquetIndexer` (e.g. `time_col`/`lon_col`/`lat_col`, `partition_by`) |
| `open_kwargs` | `None` | Extra kwargs forwarded to the xarray open call |

Returns the `parquet_root` written. Raises `ValueError` if the store has a
`depth` dim but `depth` is not given, or if `start_date` is after `end_date`.

!!! note "Incremental backfill is not replicated"
    The class's incremental/backfill mode is inherently config-driven (it walks
    `app_config.variables`, `compiled_vars`, and source coverage to catch up
    lagging columns). The ad-hoc function does explicit/full-range conversion;
    `add_data`'s overlap resolution still handles re-runs and appends correctly.

---

## When to use the class instead

| Use the function | Use the class (`Netcdf2Zarr` / `Zarr2Parquet`) |
|---|---|
| Files not registered in `config.yaml` | A configured `var_key` |
| One-off / external / manually placed data | Resumable pipeline runs with date inference |
| You provide input and output paths | Catalog-driven discovery and output naming |
| No per-variable backfill needed | Incremental append + per-variable backfill |
