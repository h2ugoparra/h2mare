# Compiler

`Compiler` reads per-variable Zarr stores, interpolates them to a common 0.25° daily grid, and writes the merged `h2ds` dataset.

```python
from h2mare.processing.compiler import Compiler

Compiler().run(start_date="2024-01-01", end_date="2024-12-31")
```

---

## Constructor

```python
Compiler(
    var_key="h2ds",
    app_config=None,
    remote_store_root=None,
    local_store_root=None,
    time_resolution=TimeResolution.YEAR,
    date_format="year",
)
```

| Parameter | Default | Description |
|---|---|---|
| `var_key` | `"h2ds"` | Output variable key (defines the target grid via `config.yaml`) |
| `app_config` | settings | Override the application configuration |
| `remote_store_root` | `STORE_ROOT` | Root directory where source Zarr stores live |
| `local_store_root` | `ZARR_DIR` | Local copy destination for the compiled output |
| `time_resolution` | `YEAR` | Output file granularity: `YEAR` or `MONTH` |
| `date_format` | `"year"` | Output filename date format: `"year"`, `"yearmonth"`, or `"date"` |

---

## `run()`

```python
Compiler().run(
    start_date=None,
    end_date=None,
    var_keys=None,
    dx=0.25,
    dy=0.25,
    zarr_backup=False,
    zarr_backup_dir=None,
)
```

| Parameter | Description |
|---|---|
| `start_date` | Start of compilation period. Inferred from store if `None` |
| `end_date` | End of compilation period. Inferred from store if `None` |
| `var_keys` | List of variable keys to include. Defaults to all keys in `config.yaml` |
| `dx`, `dy` | Output grid resolution in degrees |
| `zarr_backup` | Copy compiled Zarr files to the local backup store. Defaults to `False` |
| `zarr_backup_dir` | Override backup destination. Defaults to `local_store_root` |

The method splits the requested range into yearly (or monthly) chunks, processes each variable independently, merges the results with `xr.merge`, and writes via `write_append_zarr`. After all chunks are written, the compiled files are backed up to `local_store_root` (or `zarr_backup_dir` if provided) only when `zarr_backup=True`.

Variables with no data for a given chunk are skipped with a warning rather than raising an error.

---

## Special variable handling

| Variable | Behaviour |
|---|---|
| `bathy` | Read from the static NetCDF file configured via `data_file` in `config.yaml`; interpolated onto the output grid |
| `moon` | Lunar illumination computed from `ephem` for each day; broadcast to all grid cells |
| `o2` | Depth-sliced at 0, 100, 500, and 1000 m before interpolation |
| `atm-accum-avg` | `dayofyear`, `month`, `quantile` coordinates dropped before merge |
| `sst` | `sst_fdist` clipped to ≥ 0 after interpolation |
