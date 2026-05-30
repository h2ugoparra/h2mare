# Configuration

H2MARE is configured through two files: `config.yaml` (variable definitions and processing parameters) and `.env` (paths and credentials).

---

## config.yaml

### Variable entries

Each key under `variables:` defines one data stream:

```yaml
variables:
  sst:
    local_folder: CMEMS_SST           # subdirectory under STORE_ROOT
    variables: [analysed_sst, ...]    # variable names inside the source file
    dataset_id_rep: <cmems-id>        # reprocessed (multiyear) dataset ID
    dataset_id_nrt: <cmems-id>        # near-real-time dataset ID (optional)
    source: cmems                     # cmems | aviso | cds
    pattern: "(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})"  # filename date pattern
    subset: true                      # spatial subset on download
    bbox: [-80, 0, 10, 70]           # [xmin, ymin, xmax, ymax]
    depth_range: [0.0, 500.0]        # [min_depth, max_depth]
```

| Field | Required | Description |
|---|---|---|
| `local_folder` | yes | Subdirectory under `STORE_ROOT` for this variable's Zarr files |
| `variables` | yes | Variable names to extract from source files |
| `dataset_id_rep` | yes | Reprocessed dataset identifier |
| `dataset_id_nrt` | no | Near-real-time dataset identifier. Omit for reanalysis-only products |
| `source` | yes | Provider: `cmems`, `aviso`, or `cds` |
| `pattern` | yes | Regex for extracting date ranges from raw filenames |
| `subset` | no | Whether to spatially subset on download (default `false`) |
| `merge_time_step` | no | Set to `true` for CDS/ERA5 accumulated or averaged variables whose GRIB files have a 2-D `time × step` coordinate grid instead of a flat `time` axis (e.g. `atm-accum-avg`, `radiation`). Triggers a preprocess step that merges the two dimensions and trims overlapping timestamps at month edges. Default `false`. |
| `filename_date_range` | no | Set to `true` when the `pattern` has two capture groups encoding a `(start, end)` date range (e.g. CMEMS/CDS files named `2021-01-01-2021-01-31.nc`). Leave `false` (default) when the pattern yields a single date (e.g. AVISO FSLE: `_20210115_`). Controls how `Netcdf2Zarr` expands filenames into daily time steps. |
| `bbox` | no | Bounding box for subset. If omitted, the full available extent is downloaded |
| `depth_range` | no | Depth range for 3D variables (e.g. `o2`) |
| `data_file` | no | Filename of the static source file at the configured output resolution (e.g. 0.25°). Used by compile-only variables such as `bathy` |
| `data_file_hires` | no | Filename of the high-resolution static source file. Used by `bathy` when extracting at full native resolution (e.g. from SHP geometries) |
| `trajectory_format` | no | Set to `true` for trajectory-format datasets (e.g. `eddies`) that require spatial binning before they can be stored as a gridded Zarr. The standard `open_mfdataset` pipeline is bypassed entirely. Default `false`. |
| `rename_lonlat` | no | Set to `true` for variables whose Zarr store uses `lon`/`lat` coordinate names that must be renamed to `x`/`y` before `rioxarray` clip during extraction (e.g. AVISO `fsle`, `eddies`). Default `false`. |
| `extract_depth_slices` | no | Depth levels (metres) to extract when slicing a 3-D variable during `Extractor` runs. Each level becomes a separate output column (e.g. `[0, 100, 500]` → `o2_0`, `o2_100`, `o2_500`). Omit for 2-D variables. |

### Validation

h2mare warns at load time if `config.yaml` contains top-level keys other than `variables`, `global_attrs`, and `variable_attrs`. Unknown keys are ignored, but the warning helps catch typos like `varibles:` before they cause a silent misconfiguration.

### The `h2ds` key

The special `h2ds` variable defines the output grid for the compile step:

```yaml
  h2ds:
    local_folder: h2ds
    dataset_id_rep: compiled-data-0.25deg-P1D
    source: h2mare
    bbox: [-80, 0, 10, 70]
```

The `bbox` here sets the spatial extent of the compiled dataset.

---

## .env

| Variable | Required | Description |
|---|---|---|
| `STORE_ROOT` | yes | Root path for Zarr output (can be an external drive) |
| `CMEMS_USERNAME` | CMEMS only | Copernicus Marine account username |
| `CMEMS_PASSWORD` | CMEMS only | Copernicus Marine account password |
| `AVISO_USERNAME` | AVISO only | AVISO account username |
| `AVISO_PASSWORD` | AVISO only | AVISO account password |
| `AVISO_FTP_SERVER` | AVISO only | FTP server hostname |

CDS / ERA5 credentials are handled by the `cdsapi` package and stored in `~/.cdsapirc`.

---

## Adding a new variable

1. Add an entry under `variables:` in `config.yaml` with the correct `source`, `dataset_id_rep`, and `local_folder`.
2. Add `variable_attrs` entries for each output variable name (used to set metadata in compiled Zarr files).
3. If the variable is a CDS/ERA5 accumulated or averaged product (GRIB files with a `time × step` structure), set `merge_time_step: true` in its config entry.
4. If each downloaded file covers a date range encoded in its filename as two groups (e.g. `2021-01-01-2021-01-31.nc`), set `filename_date_range: true`. Leave it unset for variables whose filenames encode a single date (e.g. AVISO FSLE).
5. If the variable is a trajectory dataset that requires spatial binning (observations indexed by `obs`, not a lat/lon/time grid), set `trajectory_format: true`.
6. If the source is new, implement a downloader class inheriting from `BaseDownloader` and register it in `h2mare/cli/main.py`.
