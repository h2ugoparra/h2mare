# ZarrCatalog

`ZarrCatalog` maintains a Parquet index for a single variable key, enabling efficient temporal range queries without opening every Zarr file.

```python
from h2mare.storage.zarr_catalog import ZarrCatalog

catalog = ZarrCatalog("sst")
print(catalog)                          # summary: coverage, bbox, file count
ds = catalog.open_dataset(
    start_date="2024-01-01",
    end_date="2024-12-31",
)
```

---

## Constructor

```python
ZarrCatalog(
    var_key,
    *,
    file_period=FilePeriod.YEAR,
    app_config=None,
    store_root=None,
    metadata_root=None,
    auto_refresh=True,
    verbose=False,
    warn_if_missing=True,
)
```

| Parameter | Default | Description |
|---|---|---|
| `var_key` | — | Variable key; must exist in `config.yaml` |
| `file_period` | `YEAR` | Granularity used for the `period` column in the index |
| `app_config` | settings | Configuration to read `var_key` from. Pass another project's `AppConfig` to catalog its stores |
| `store_root` | `STORE_ROOT/<local_folder>` | Directory scanned for `.zarr` files |
| `metadata_root` | `data/processed/metadata/` | Directory for the Parquet catalog file |
| `auto_refresh` | `True` | Check for new/modified files on each `.df` access |
| `verbose` | `False` | Log the catalog's own bookkeeping (scans, rebuilds, cache use) at its level; otherwise it stays quiet |
| `warn_if_missing` | `True` | Log a warning when the store directory does not exist yet. Set `False` where an empty store is expected, e.g. before the first convert |

Everything after `var_key` is keyword-only.

---

## `open_dataset()`

Open one or more Zarr files as a lazy xarray Dataset.

```python
# Date range mode
ds = catalog.open_dataset(
    start_date="2024-01-01",
    end_date="2024-12-31",
    bbox=(-80, 0, 10, 70),        # optional spatial subset
    variables=["analysed_sst"],   # optional variable selection
)

# Sparse dates mode
ds = catalog.open_dataset(
    dates=["2024-06-15", "2024-07-20"],
)
```

`chunks` (default `"auto"`) is forwarded to xarray; pass a dict or `None` to control dask chunking.

Issues a warning (but does not raise) when the requested range extends beyond what the store contains, and clamps to the available period.

---

## Catalog management

| Method | Description |
|---|---|
| `refresh(force=False)` | Reload from disk; rescan if files have changed or `force=True` |
| `reload()` | Force a full rescan unconditionally |
| `get_time_coverage()` | Return `DateRange(min_start, max_end)` across all files |
| `get_var_coverage(var)` | Return the dates *`var` itself* has data for — narrower than the above on a compiled store, where `xr.merge` pads each variable out to the union axis |
| `get_vars_nonnull_start(vars)` / `get_vars_nonnull_end(vars)` | First / last date each variable holds non-null data; one pass for the whole batch |
| `get_var_time_coverage(var)` | Span of the files that hold `var` as a data variable (from the index; reads no data) |
| `get_nonnull_days(window, var_names=None)` | Dates within `window` on which each variable holds non-null data |
| `get_variables()` | Return the set of variable names across all files |
| `get_bbox()` | Return the configured `BBox` for this variable |
| `get_store_bbox()` | The extent the files actually hold, unioned across the store (cell centres) |
| `summary()` | Return a dict with file count, coverage, variables, and paths |
| `get_paths_in_range(start_date, end_date)` | Paths of the files overlapping a date range |
| `map_dates_to_paths(dates)` | Group sparse dates by the file holding them, as path → dates |
| `build_file_path(ds, date_format, store_root=None, name_key=None)` | The path a dataset is written to: `{source}_{name_key}_{geo_extent}_{dates}.zarr` |
| `exists()` / `catalog_path` | Whether the catalog Parquet file exists, and its path |
| `has_changes()` / `get_change_summary()` | Whether files were added, removed or modified since the last scan, and which |
| `df` | The catalog as a pandas DataFrame (re-stats the store on access while `auto_refresh` is on) |
| `refresh_provenance()` | Rebuild each file's `source_datasets` from its own time axis; returns the number of files rewritten |
| `backfill_provenance(rep_end_date)` | Write provenance sidecars for Zarr files that predate tracking |

The module function `get_zarr_time_coverage(var_key)` returns the same span as `ZarrCatalog(var_key).get_time_coverage()` without keeping the catalog around.

---

## Staleness detection

On each cold-start load `ZarrCatalog` compares the set of `.zarr` directory names on disk against those in the Parquet index. If they differ (files added or removed) it rescans automatically. Stale entries caused by in-place appends (same filename, new content) are detected via the stored `file_mtime`.

---

## Catalog schema

Each row in the Parquet index represents one source dataset within one Zarr file.

| Column | Type | Description |
|---|---|---|
| `path` | str | Absolute path to the `.zarr` directory |
| `filename` | str | Basename of the `.zarr` directory |
| `start_date` | datetime | First timestep in this source's period |
| `end_date` | datetime | Last timestep in this source's period |
| `dataset` | str | Source dataset ID (rep or nrt) |
| `variables` | list[str] | Variable names inside the Zarr |
| `xmin/ymin/xmax/ymax` | float | Spatial extent |
| `num_timesteps` | int | Number of time steps in this source's period |
| `file_mtime` | float | File modification time at last scan |
| `scanned_at` | datetime | When this row was written |
