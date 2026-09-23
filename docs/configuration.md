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
    store_root: /mnt/fast_ssd         # optional: this variable's own root
    source_vars: [analysed_sst, ...]  # variable names inside the source file
    dataset_id_rep: <cmems-id>        # reprocessed (multiyear) dataset ID
    dataset_id_nrt: <cmems-id>        # near-real-time dataset ID (optional)
    source: cmems                     # cmems | aviso | cds
    archive_raw: false                # optional, default false: keep raw files in store (true) or delete after convert
    pattern: "(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})"  # filename date pattern
    subset: true                      # CMEMS only: subset() vs get() download API
    bbox: [-80, 0, 10, 70]           # [xmin, ymin, xmax, ymax]
    depth_range: [0.0, 500.0]        # [min_depth, max_depth]

  radiation:
    local_folder: CDS_Radiation
    source: cds
    archive_raw: false
    time_step: hourly                 # keep the source cadence; h2ds stays daily
    store_dtype: int16                # scale/offset packed, ~2/3 the size
    merge_time_step: true             # GRIB time x step grid
    dataset_id_rep: reanalysis-era5-single-levels
```

Both `time_step` and `store_dtype` are properties of the **store**, not of a run:
each takes effect when a Zarr is created and an append inherits it, so changing
either on an existing variable means re-converting it.

| Field | Required | Description |
|---|---|---|
| `local_folder` | yes | Subdirectory under `STORE_ROOT` (or `store_root`) for this variable's Zarr files |
| `store_root` | no | Root holding this variable's `local_folder`, for stores that should not live under `STORE_ROOT` — e.g. the hourly ERA5 stores on one drive and the CMEMS dailies on another. Must be absolute (either `/data/store` or `D:\Data`; a relative path is rejected at config load because it would resolve against the current working directory). Defaults to `STORE_ROOT`. See [Where a variable's store lives](#where-a-variables-store-lives). |
| `source_vars` | yes | Variable names to extract from source files |
| `dataset_id_rep` | yes | Reprocessed dataset identifier |
| `dataset_id_nrt` | no | Near-real-time dataset identifier. Omit for reanalysis-only products |
| `source` | yes | Provider: `cmems`, `aviso`, or `cds` |
| `archive_raw` | no | Whether to keep this variable's raw NetCDF/GRIB files by moving them into the store after conversion (`true`), or delete them per-period once converted (`false`, the default). Set `true` where the raw files are costly to fetch again — the shipped config does for `fsle` and `eddies` — since re-converting a store (for example to change `store_dtype`) re-reads them, and without the archive that means downloading them again. Meaningless for `bathy`, `moon` and `h2ds`, which are never converted. |
| `pattern` | download vars | Regex matched against each raw filename to extract date component(s). Unmatched optional groups are dropped before parsing. Its capture groups must agree with `filename_date_range`: with `true`, up to **2** groups giving `(start, end)` — one group alone is read as a single day, which is why the shipped patterns make the range tail optional (`(\d{4}-\d{2}-\d{2})(?:-(\d{4}-\d{2}-\d{2}))?`); with `false`, the groups are joined with `-` and parsed as a single date (e.g. `(\d{8})` → `20210115`; `(\d{4})(\d{2})(\d{2})` → `2021-01-15`). Omit for derived/system variables (`bathy`, `moon`, `h2ds`) that are never matched against filenames. |
| `subset` | CMEMS only | Chooses the CMEMS download API: `true` (default) uses `copernicusmarine.subset()` (spatial/variable subset honoring `bbox`/`source_vars`); `false` uses `copernicusmarine.get()` to fetch full original files. Ignored for non-CMEMS sources. |
| `merge_time_step` | no | Set to `true` for CDS/ERA5 accumulated or averaged variables whose GRIB files have a 2-D `time × step` coordinate grid instead of a flat `time` axis (e.g. `atm-accum-avg`, `radiation`). Triggers a preprocess step that merges the two dimensions and trims overlapping timestamps at month edges. Default `false`. |
| `filename_date_range` | no | Set to `true` when the `pattern` captures a `(start, end)` date range (e.g. CMEMS/CDS files named `2021-01-01-2021-01-31.nc`). A **one-day** request is named with a single date instead (`2026-07-31.nc`), so make the second group optional and it will be read as that one day — without this a single-day repair download matches nothing and is discarded. Leave `false` (default) when the pattern yields a single date (e.g. AVISO FSLE: `_20210115_`). Controls how `Netcdf2Zarr` expands filenames into daily time steps. |
| `known_gaps` | no | Days the provider never published, so they can never be downloaded, converted or backfilled. Each entry is a date (`2025-06-02`) or a closed interval (`2025-06-02/2025-06-05`). Excluded from the gap checks and from `h2mare audit`, which reports how many were suppressed. Needed because a source shipping one file per day leaves an *axis* hole when it skips one — AVISO has no `fsle` file for 2025-06-02 and its remote listing jumps `20250601` → `20250603` — which is otherwise indistinguishable from data the pipeline lost. Only for gaps confirmed absent at the source; anything else is a defect and belongs fixed, not listed. |
| `time_step` | no | Cadence of this variable's own Zarr: `daily` (default) or `hourly`. An hourly store keeps the source's native axis and moves the daily reduction to compile time, so h2ds stays daily either way. Distinct from `file_period`, which is about the storage layout (one Zarr per year or per month) — a store can be hourly and still written one file per year. The gap checks read this so they compare a store against a calendar at its own resolution; a daily grid cannot see a missing hour, and an hourly grid over a daily store would report 23 phantom gaps a day. Flipping it on an existing store requires re-converting: the store is written at one cadence and the check expects the other, which fails the write verification rather than corrupting anything. It also changes where `Extractor` reads this variable from: with `hourly`, the daily values and the features derived from them are written only to the compiled h2ds, so a date-only extraction is answered from there and needs a current `compile`. Converting the same variable `daily` computes those up front and keeps everything in its own store, where `Extractor` finds them natively. See [Cadence](api/extractor.md#cadence). |
| `store_dtype` | no | On-disk encoding: `float32` (default, byte-identical to what the pipeline has always written) or `int16`, which stores scale/offset-packed integers at roughly two thirds the size. Safe for ERA5, whose GRIB is already ~16-bit packed, so the packing discards quantisation noise rather than signal. The scale spans each variable's own measured range widened for headroom, so the encoding step makes one pass over the data before the first byte is written — expect a silent minutes-long pause on a large store. Applied only when a store is **created**; appends inherit whatever encoding the store already has, so changing this on an existing store does nothing until it is re-converted, and an append carrying values the frozen scale cannot represent is refused rather than wrapped. Safe for some variables and not others — see [Choosing `store_dtype`](#choosing-store_dtype). |
| `raw_include` | no | Regex matched (via `re.search`) against each raw filename; only matching files are converted. Use when a download directory holds files the pipeline must not read — AVISO ships META3.2 eddy trajectories as `long`/`short`/`untracked` variants side by side, and only the long ones belong in the store (the `untracked` files carry no `track` variable at all). Omit (default) to convert every file the date `pattern` matches. |
| `bbox` | no | Bounding box for subset. If omitted, the full available extent is downloaded |
| `depth_range` | no | Continuous depth band `[min, max]` (metres) **downloaded** for 3-D variables (e.g. `o2`). Says nothing about which depths are published — that is `depth_levels`. |
| `depth_levels` | 3-D vars | Discrete depths (metres) each 3-D variable is **published** at, keyed by the variable's name in the store: `{thetao: [0, 50, 100], uo: [0]}`. Each level becomes a column `<variable>_<level>` (`thetao_50`), and these names belong in `compiled_vars` — config load refuses a declared `compiled_vars` that leaves one out. Store variables not listed pass through when they have no depth axis, so one store can mix 2-D and 3-D fields (`zos` beside `thetao`); a variable **with** a depth axis that is not listed is refused rather than averaged over its whole range, as is a listed name the store does not hold. Levels are matched to the store's axis by nearest depth and keep the requested name (`o2_1000` off a 902 m axis). Any var_key with levels is compiled by the depth processor — no registry entry needed. Also the default for extraction, where only the variables requested are sliced. |
| `layers` | `bathy` | Static layers the variable is read from, by name → file under `<store_root>/<local_folder>/`: `{15s: etopo2022_15s_….zarr, 60s: etopo2022_60s_….zarr, 0.25deg: etopo_0.25deg_….nc}`. The only place a layer's file name lives — `scripts/bathymetry.py` writes each file there, compile and extraction read it. The native layers (15s, 60s) hold `bathy` plus the entry's `derived_vars` (`bathy_std`, a 3×3 rolling std) and keep the ETOPO source attributes; the 0.25° layer holds the mean and std of the 15s cells inside each cell. |
| `compile_layer` | `bathy` | Layer compile puts on the base grid — the layer's **name**, a key of `layers`, not its file. |
| `extract_layer` | no | Default layer for extraction — a key of `layers` — for point and geometry input alike; `Extractor(bathy_layer=...)` overrides it for one run. Without either, extracting `bathy` raises. See [The `bathy` key](#the-bathy-key). |
| `trajectory_format` | no | Set to `true` for trajectory-format datasets (e.g. `eddies`) that require spatial binning before they can be stored as a gridded Zarr. The standard `open_mfdataset` pipeline is bypassed entirely. Default `false`. |
| `n_workers` | no | Worker processes for a trajectory variable's rasterisation (`eddies`), one day per task. Defaults to 4, which is also about where it stops paying: profiling a year at 1/12° showed 8 workers no faster than 4, because the limit is the staging write in the main process rather than the search in the workers. Conversion runs a month at a time, so memory follows the month, not this number. |
| `rename_lonlat` | no | **No longer needed.** Geometry extraction now renames `lon`/`lat` to `x`/`y` for every variable, because `rioxarray`'s clip resolves spatial dims by name and only falls back to `lon`/`lat` when they carry CF attributes — which CMEMS and AVISO stores inherit from source but CDS stores and the compiled h2ds do not. Setting it changes nothing; it is kept so existing config files stay valid. Default `false`. |
| `extract_depth_levels` | no | Extraction-only depths, same shape as `depth_levels`: `{thetao: [0]}`. Merged per variable — a variable listed here replaces its `depth_levels` entry for extraction, the others keep theirs — and never reaches compile or `compiled_vars`. **Omit it** and extraction returns exactly the columns the variable publishes, agreeing with h2ds and Parquet. A single run can also choose levels without touching config: see [Depth levels](api/extractor.md#depth-levels). |
| `extract_depth_slices` | no | **Older form of `extract_depth_levels`**, still accepted: a list meaning `{<var_key>: [...]}`. Setting both is refused. |
| `compile_depth_slices` | no | **Older form of `depth_levels`**, still accepted: `compile_depth_slices: [0, 100]` means `depth_levels: {<var_key>: [0, 100]}`, so it only fits a store whose single 3-D variable is named like its var_key (`o2`, `thetao`). Setting both is refused. |
| `derived_vars` | no | Variables computed at convert time and written to the native store, keyed by output name: `{gke: {op: kinetic_energy, source: [ugos, vgos]}}`. `op` is `rolling_std` (one `source`; `window`, an odd number of cells, default 3, gives the square lon/lat box centred on each cell, which uses whatever part of the box holds data) or `kinetic_energy` (`source: [u, v]`, gives `0.5*(u²+v²)`). Sources are named as the var_key's registered processor leaves them (`sst`, not `analysed_sst`). Entries run in order, so one may read an earlier one. Missing sources, a wrong number of them, an even `window` or a misspelt field are refused. By default the result keeps any depth axis its sources have, so a derived 3-D variable needs its own `depth_levels` entry (`ke: [0]` → `ke_0`). To compute and store only some depths, add `depth: [0, 50]`: each level is matched to the nearest source depth (as `depth_levels` does) and written as a 2-D `ke_0`, `ke_50`, so the other levels are never read or stored. Such an entry is listed by those names in `compiled_vars` and must not appear in `depth_levels`; config load refuses that, and refuses two entries or a `depth_levels` column writing the same name. The cost is that changing the levels needs a reconversion, and a per-run `{"ke": [50]}` extraction override does not apply to it (ask for `ke_50` by name). Every output name needs a `variable_attrs` entry like any other. The `sst_std`, `adt_std`, `sla_std` and `gke` layers are declared this way, so removing those entries stops them being written. `bathy` is never converted: its entry (`bathy_std`) is applied by `scripts/bathymetry.py` when it builds the native layers. |
| `compiled_vars` | no | Exact variable names as they appear in the compiled h2ds Zarr for this var_key, accounting for any renames or derived variables produced during the Convert step (e.g. `sst` → `[sst, analysis_error, sst_std, sst_fdist]`). Used by `h2mare parquet --add-var` to select only the relevant columns from the h2ds Zarr without the caller needing to know internal variable names. |
| `cells_per_degree` | no | The grid this var_key's own store is written on, as a whole number of cells per degree: `4` is 0.25°, `8` is 0.125°, `12` is 1/12°, `20` is 0.05°. A count rather than a step so the value is exact — `0.083` is not 1/12°, and lays 1084 cells across a 90° span where 1080 belong. Refused at load if it does not divide the bbox into whole cells. Read by the steps that *create* a grid — the compile (`h2ds`) and the eddy rasterisation — and ignored by a variable that keeps its source files' own grid. Defaults to 4. |
| `values_at` | no | Where that grid's values sit: `cell_center` (the default, and what every existing store uses) puts them at cell centres, half a step inside each bbox edge; `grid_line` puts them where the grid lines cross, on the step's own multiples. Independent of the step — see [Where the values sit](#where-the-values-sit) for which one a given resolution wants. |
| `regrid` | no | How individual columns are put on the compile base grid, keyed by the column name: `{ac_track: nearest}`. Anything not listed uses `auto`, which compares the variable's native resolution with the base grid and picks `linear` when the grid is the same or finer and an area-weighted mean when it is coarser — the right choice for a continuous field either way. Set `nearest` for a column a mean would destroy: an identifier or a class, or a quantity describing something other than the cell itself. `linear` and `conservative` pin the automatic choice. Names are checked against `compiled_vars` at config load, so a typo is refused rather than silently ignored. See [Regridding](#regridding). |

### Regridding

Each variable is measured against the base grid and put on it accordingly, so
neither the direction nor the ratio has to be declared:

| Native vs base grid | Method | Example at 0.25° |
| --- | --- | --- |
| Same or finer base grid | `linear` | ERA5 and `o2` at 0.25°, `waves` at 0.5° |
| Coarser base grid | area-weighted mean | `sst` 0.05° (5×5 source cells per output cell), `chl` 1/24°, `thetao`/`mld` 1/12° |

The mean skips NaN cells and normalises by the valid area, so a cell that is
part land reports the mean of its water — land contributes nothing to the value
rather than dragging it. **Any valid area is enough to give a cell a value**, so
the compiled product reaches as far into the coast as its sources do: on
2024-02-15 that is 74,993 sea cells for `sst`, against 71,743 under a
point-sampled interpolation. A cell touching a single 0.05° pixel of ocean is
therefore real, if noisier than an open-ocean one; `regrid_to`'s `min_coverage`
can require a minimum valid fraction, and the compile leaves it at 0.

`nearest` is never chosen automatically — a mean is right for almost every
field, and the exceptions are only visible to someone who knows what the numbers
mean. The eddy columns are the shipped case: each holds a property of whichever
eddy is nearest, so it is constant across that eddy's neighbourhood and a mean
taken across a boundary describes neither eddy. `ac_track` averaged with its
neighbour names an eddy that does not exist. `ac_dist_km` and `ac_normdist` are
left on `auto` in the same entry, because those two really are continuous
fields.

### Where the values sit

A grid needs two things: the step, and where its values sit. `grid_line` and
`cell_center` are both valid at 0.25° — one puts values at 0.00, 0.25, 0.50, the
other at 0.125, 0.375, 0.625. The step does not pick between them.

What does is your sources. Registration only matters for a variable whose native
step **equals** the target step: aligned, it is copied through exactly; half a
cell off, it is interpolated onto the point between its cells, which averages the
four around it. A coarsened variable is area-averaged over the target cell
whatever its phase, and a refined one is being interpolated anyway.

Measured on 2026-03-01 fields, that half-cell average costs, as a fraction of the
field's own spatial variability:

| Field | RMSE | % of field σ |
| --- | --- | --- |
| `u10` daily mean, 0.25° | 0.36 m/s | 5.9% |
| `o2` surface, 0.25° | 2.4 mmol/m³ | 4.7% |
| `mld`, 1/12° | 13.9 m | 11.2% |
| `adt`, 0.125° | 0.011 m | 2.8% |

It is a resolution loss rather than noise, concentrated where the gradients are.
So the placement worth choosing is the one matching whichever family lands at
ratio 1 for the step you pick:

| Step | At ratio 1 | Its values sit at |
| --- | --- | --- |
| 0.05° | `sst` | cell centres |
| 1/12° | `thetao`, `mld` | grid lines |
| 0.1° | `eddies` (our own raster) | grid lines |
| 0.125° | `ssh` | cell centres |
| 0.25° | ERA5 ×4, `o2` | grid lines |
| 0.5° | `waves` | grid lines |

Observation products tend to put values at cell centres, models and reanalyses
on grid lines. The shipped `h2ds` is 0.25° `cell_center`, so ERA5 and `o2` are
2×2 averaged; a compile logs a warning naming any variable in that position, so
the cost is visible rather than assumed.

Changing either key means a new store — the write path refuses to merge one grid
into another — so give it its own `local_folder` and `dataset_id_rep` and
recompile.

### Choosing `store_dtype`

`int16` packs each variable over 65,000 levels spanning its own measured
min→max, for roughly two thirds the size of `float32`. Nothing validates the
choice, so it is worth knowing when it is free and when it costs.

**It is near-free where the source was already packed at similar precision, and
adds real error where the pipeline computed the value itself.** ERA5's GRIB is
already ~16-bit packed, which is why the CDS variables use it — the packing
reproduces quantisation the data already carried. Several CMEMS products ship
int16-packed netCDF too, recognisable by a `valid_min`/`valid_max` pair that is
plainly an integer range rather than a physical one (`thetao`:
`[-32766, 21306]` "degrees_C"; `analysis_error`: `[0, 32767]` "kelvin").
Anything h2mare derives in float32 has no such floor to hide under.

The cost also depends on the *distribution*, not the source. The scale spans
min→max, so a long tail spends the levels where the data is not:

| variable | step vs median | why |
|---|---|---|
| `adt`, `sst`, `ac_speedrad_km` | <0.01% | bounded and roughly symmetric |
| `sst_std` | 0.17% | derived, small values |
| `fsle_max`, `chl` | 0.5–0.6% | log-distributed; `chl` median 0.16 against max 65 |
| `gke` | **1.5%** | squared quantity — median 0.004 against max 4.2 |

**Never for identity or index fields.** The eddy trajectory ids span a range
wider than 65,000 (`ac_track`: `[176122, 242079]`, giving a step of 1.01;
`c_track`: 1.24), so packing them collapses distinct eddies onto the same value
— a loss of identity, not of precision.

!!! warning "`store_dtype` is ignored for `trajectory_format` variables"

    The trajectory path (`eddies`) writes its Zarr without consulting the
    encoding, so setting `store_dtype: int16` there is accepted by config and
    silently has no effect. Given what packing would do to the track ids, the
    no-op is the safer outcome — but do not read the setting as evidence that
    the store is packed.

!!! warning "A store's scale is fixed for its lifetime"

    The scale is derived when the store is **created**, from the range of
    whatever batch is written first, and every later append inherits it. A
    value outside that range has no encoding — and it does not clip, it
    overflows `int16` and wraps back into the middle of the range, which is
    worse: a wrapped value lands inside the plausible band, so there are no
    outliers to spot and no NaNs to count.

    Two things guard against this. The scale is widened past the observed
    range (`_INT16_HEADROOM`, currently 1.6×, leaving ~60% headroom on each
    side at 1.6× coarser resolution — `msl` ~0.24 Pa, still far inside ERA5's
    own precision). An append that falls outside even that is caught by
    `storage._check_packed_range` before anything is written, and the store is
    **repacked**: rewritten with that variable's scale re-derived over the
    stored data and the incoming range together, then appended to. Existing
    values move by at most half a quantisation step. Variables that still fit
    keep their scale.

    A repack rewrites the whole yearly file, which is expected for zero-bounded,
    heavy-tailed fields: a store first built from January–July carries `tp`'s
    range from those months, and the August–October storm season routinely
    exceeds it. Only a variable with no range to scale over (for example an
    all-NaN store receiving a constant) is still refused, with a pointer to
    re-convert.

Reasonable candidates are the bounded, already-packed fields: `sst`, `thetao`,
`o2`, `mld`, `adt`. Leave `chl`, `gke` and the other derived variables on
`float32`.

### Validation

h2mare warns at load time if `config.yaml` contains top-level keys other than `variables`, `global_attrs`, `variable_attrs`, and `native_attr_overrides`. Unknown keys are ignored, but the warning helps catch typos like `varibles:` before they cause a silent misconfiguration.

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

### The `bathy` key

`bathy` is static (no time axis) and never downloaded or converted. It is read
from named **layers**, files under `<store_root>/<local_folder>/` that
`scripts/bathymetry.py` builds from the ETOPO 2022 v1 source files kept in that
same folder:

```yaml
  bathy:
    local_folder: ETOPO_Topography
    source_vars: [z]
    compiled_vars: [bathy, bathy_std]
    dataset_id_rep: Etopo_v1
    source: noaa
    bbox: [-80, 0, 10, 70]            # extent the layers are built over
    layers:                           # name -> file
      15s: etopo2022_15s_80W-10E-0N-70N_bathy-std.zarr
      60s: etopo2022_60s_80W-10E-0N-70N_bathy-std.zarr
      0.25deg: etopo_0.25deg_80W-10E-0N-70N_mean-std_surface.nc
    compile_layer: 0.25deg            # a name from layers, not a file
    extract_layer: 15s
    derived_vars:
      bathy_std: {op: rolling_std, source: bathy, window: 3}
```

| Layer | Built from | `bathy_std` | Used by |
|---|---|---|---|
| `15s`, `60s` | the 15″ tiles / the global 60″ file, subset to `bbox`; tiled Zarr | `derived_vars`: 3×3 rolling std on the native grid (≈ 1.4 km / ≈ 5.5 km) | extraction |
| `0.25deg` | the `15s` layer, coarsened | std of the 15″ cells inside each 0.25° cell | compile (h2ds) |

The layer names are free-form; `scripts/bathymetry.py` builds `15s`, `60s` and
`0.25deg` (`--layers` picks some, `0.25deg` needs `15s` first), writing each to
the file named here — config is the only place a layer's file name lives. The
native layers keep the ETOPO source's global attributes and `z` attributes, with
the CF attributes from `variable_attrs` and `native_attr_overrides.bathy` over
them. `compile_layer` and `extract_layer` must name a declared layer, which
config load checks. Extraction reads one layer for points and geometries alike:
a point takes the nearest cell, a geometry the polygon mean of `bathy` and
`bathy_std` — see [Standard-deviation columns](api/extractor.md#standard-deviation-columns).

---

## Global attributes

`global_attrs` becomes the root attributes of `h2ds`, following
[ACDD](https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3).
Everything in it is a *choice* — the title, the summary, who to contact, what
may be done with the data.

The facts about a given file are **not** in config and cannot be. `Conventions`,
`product_version`, `history` and the geospatial/time extents are computed by
`provenance.refresh_root_attrs` at compile and read off the store itself; a
per-period store holds a different span in every file, so a value in config
could only ever describe one of them correctly. `time_coverage_resolution` is
inferred from the axis, so a daily store reports `P1DT0H0M0S` and an hourly one
`P0DT1H0M0S` without either being told which it is.

The native per-variable stores get `Conventions`, `product_version`, `history`
and their own extents via `provenance.write_cf_root_attrs`, but **not** the
fixed globals: those describe `h2ds` ("Integrated Geospatial Dataset
Collection"), and a native store is one source's own data at its own cadence.
That call updates rather than replaces, so the `source_datasets` provenance
survives it.

`license` deliberately does not claim the MIT terms the h2mare source carries —
the data is not h2mare's to license, and each source product's terms travel with
the values.

---

## Variable metadata

`variable_attrs` entries become the attributes written onto each variable in the
Zarr stores, and they follow [CF conventions](https://cfconventions.org) so the
output is readable by CF-aware tooling (xarray, rioxarray, THREDDS, `cfchecks`).

| Key | Required | Meaning |
|---|---|---|
| `long_name` | yes | Free-text label. Used by the plotting helpers. |
| `units` | yes, unless the variable is a label | Must parse under **udunits2**. `m.s-1`, `W.m-2` and `mmol.m-3` are valid; a space means multiplication, so `degrees Celsius` parses as *angle × temperature* and is wrong — write `degree_C`. |
| `standard_name` | when one fits | Must be a current entry in the [CF standard name table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html), not a deprecated alias, and its canonical units must be convertible to `units`. Omit it rather than approximate: CF has no name for the front-distance, eddy, FSLE or Ekman-anomaly fields, and a wrong one is worse than none. |
| `cell_methods` | for temporal reductions | How the value was reduced over the cell, e.g. `'time: mean'`, `'time: sum'`, `'time: maximum'`. Quote it — the colon would otherwise start a YAML mapping. |
| `comment` | recommended | Prose description. Named `comment` because that is the attribute CF and ACDD recognise. |
| `short_name` | no | Compact label for plot legends. Not a CF attribute. |
| `product_id*` / `dataset_id*` | no | Provenance, carried through to the store. |

Two entries — `thetao` and `o2` — describe the depth-resolved parent field held
in the native store, which never reaches `h2ds`; the `thetao_*` / `o2_*` entries
describe the depth slices cut from it at compile time.

### Where the attributes are applied

`apply_cf_attrs` (`storage/xarray_helpers.py`) is the single place both write
paths take this metadata from — the convert step for the per-variable native
stores, and the compile step for `h2ds` — so the two cannot drift into
describing the same quantity differently. It also stamps the CF attributes on
`lon`, `lat`, `time` and `depth`. That half is not cosmetic: `rio.clip` resolves
spatial dims by name and only falls back to CF attributes when they are not
called `x`/`y`, so on a store whose coordinates carry nothing it cannot find
them at all, and geometry extraction clips to NaN.

### `native_attr_overrides`

A native store does not always hold what `h2ds` publishes, so where the two
differ the delta lives under `native_attr_overrides`, keyed by var_key and then
by variable name. A `null` value **removes** the attribute instead of setting it.

```yaml
native_attr_overrides:
  atm-instante:
    msl:
      units: Pa           # hPa only after the compile converts
      cell_methods: null  # hourly instantaneous, not a daily mean
```

The hourly CDS stores need entries on two counts: they keep ERA5's own
units, because the conversion happens on the way into `h2ds`, and they keep
ERA5's hourly cadence, so a `cell_methods` naming a daily reduction does not
describe them. `radiation` is deliberately absent — `hourly_radiation` converts
J m⁻² to W m⁻² at both cadences, and each hourly value is a mean over its own
interval, so both the units and `time: mean` still hold.

`bathy` has one too: the shared table describes the 0.25° column compiled into
`h2ds` (std of the 15″ cells inside each cell), while the native 15″/60″ layers
written by `scripts/bathymetry.py` hold a 3×3 rolling std, so their `comment`s
are overridden there.

### Sign conventions

`bathy` keeps the ETOPO sign convention — `positive: up`, so sea-floor values are
negative — and takes `standard_name: altitude` ("geometric height above the
geoid") to match it. CF's `sea_floor_depth_below_geoid` would be the obvious
choice but is defined positive-*down*, and `bedrock_altitude` claims the surface
is bedrock, which the ETOPO *surface* grid does not guarantee over ice. The
source's own `standard_name: height` is looser still: CF defines height as the
distance above *the surface*, not above the geoid.

---

## Where a variable's store lives

A variable's Zarr store is `<root>/<local_folder>/`. The root is chosen in this
order, first match winning:

1. `--store-path` on the command line — relocates the **whole run**, including
   variables that name a root of their own.
2. `store_root` in that variable's `config.yaml` entry.
3. `STORE_ROOT` from `.env`.
4. `data/processed/zarr/` under the project root, when `STORE_ROOT` is unset.

Declaring nothing keeps the historical behaviour: every variable sits under
`STORE_ROOT`. Spreading variables across drives is a matter of adding
`store_root` to the entries that should move:

```yaml
variables:
  sst:
    local_folder: CMEMS_SST         # -> $STORE_ROOT/CMEMS_SST
  radiation:
    local_folder: CDS_Radiation
    store_root: /mnt/bulk           # -> /mnt/bulk/CDS_Radiation
```

Only the Zarr stores follow this setting. Downloads stay under the project's
`data/raw/`, and the Parquet store and `Climatology/` remain single shared trees
under `STORE_ROOT` — they are not per-variable.

Moving an existing store is a file move plus a config edit; nothing rewrites the
data. The catalog sidecar under `data/processed/metadata/` still points at the
old location, but it is detected as belonging to another root and rebuilt on the
next read, so it does not need deleting by hand.

---

## .env

| Variable | Required | Description |
|---|---|---|
| `STORE_ROOT` | yes* | Root path for Zarr output (can be an external drive). *Required unless every variable declares its own `store_root` in `config.yaml`; it is the root for those that do not. |
| `H2MARE_ROOT` | no | The project root. `data/` (raw, interim, processed) and `logs/` hang off it, and `config.yaml` / `.env` are read from it by default — so pointing it elsewhere moves the whole data tree, not just the config. Distinct from `STORE_ROOT`, which only locates the per-variable Zarr stores. Overrides the default auto-detection (walking up from the current working directory looking for `config.yaml`); the path is taken as given, so no `config.yaml` need exist there. Must be a real environment variable — the root is resolved before `.env` is loaded, so setting it inside `.env` has no effect. See [Installation](installation.md#where-to-place-these-files). |
| `AVISO_USERNAME` | AVISO only | AVISO account username |
| `AVISO_PASSWORD` | AVISO only | AVISO account password. **Sent in cleartext** — see the note below the table. |
| `AVISO_FTP_SERVER` | AVISO only | FTP server hostname |

CMEMS credentials take no env var: the `copernicusmarine` CLI stores them itself (`copernicusmarine login`) and h2mare reads none. CDS / ERA5 credentials are handled by the `cdsapi` package and stored in `~/.cdsapirc`.

!!! warning "AVISO traffic is unencrypted"

    `ftp-access.aviso.altimetry.fr` supports plain FTP only. Its `FEAT` reply
    advertises no `AUTH`, and an explicit `AUTH TLS` is refused with
    `500 AUTH not understood` (checked 2026-09-01). `AVISODownloader` therefore
    uses `ftplib.FTP`, and no setting changes that — the username, password and
    the FSLE/eddy transfers all cross the network unencrypted.

    Use a password unique to AVISO, and prefer downloading from a trusted
    network. If AVISO enables AUTH TLS later, the change is small: `FTP_TLS`
    in place of `FTP` in `connect_ftp`, plus `prot_p()` after login.

---

## Adding a new variable

1. Add an entry under `variables:` in `config.yaml` with the correct `source`, `dataset_id_rep`, and `local_folder`. Add `store_root` only if this variable's store should not sit under `STORE_ROOT` — see [Where a variable's store lives](#where-a-variables-store-lives).
2. Add `variable_attrs` entries for each output variable name (used to set metadata in the Zarr stores). See [Variable metadata](#variable-metadata) for what each key must contain — `units` has to parse under udunits2 and any `standard_name` has to exist in the CF table.
3. If the variable is a CDS/ERA5 accumulated or averaged product (GRIB files with a `time × step` structure), set `merge_time_step: true` in its config entry.
4. If each downloaded file covers a date range encoded in its filename as two groups (e.g. `2021-01-01-2021-01-31.nc`), set `filename_date_range: true` and make the second group optional so a one-day download still parses. Leave it unset for variables whose filenames encode a single date (e.g. AVISO FSLE).
5. To publish a rolling std or a kinetic energy computed from the downloaded fields, declare it under `derived_vars` rather than writing a processor.
6. If the variable is a trajectory dataset that requires spatial binning (observations indexed by `obs`, not a lat/lon/time grid), set `trajectory_format: true`.
7. If the source is new, implement a downloader class inheriting from `BaseDownloader` and register it in `h2mare/cli/main.py`.
