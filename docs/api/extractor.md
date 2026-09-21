# Extractor

`Extractor` extracts environmental values at point locations (CSV / `DataFrame`) or
spatial geometries (SHP / `GeoDataFrame`). By default it reads the per-variable Zarr stores
via `ZarrCatalog`, falling back to the compiled h2ds for variables stored hourly (see
[Cadence](#cadence)); it can also extract against an arbitrary in-memory `xarray` dataset
that is not yet in the store (see [`extract_from_dataset()`](#extract_from_dataset)).

```python
from h2mare.processing.extractor import Extractor

extractor = Extractor("data/points.csv", time_col="date", index_col="id_row")
df = extractor.run("sst")                       # returns a DataFrame
extractor.run("sst", output_path="out.csv")     # or writes a CSV and returns None
```

The input (points or geometries) is supplied to the **constructor**, not to `run()`.
There are no `start_date` / `end_date` arguments: the extraction window is the set of
dates present in the input, intersected with each variable's store coverage.

---

## Constructor

```python
Extractor(
    file_path,            # path to .csv/.shp, or an in-memory DataFrame/GeoDataFrame
    *,
    index_col,            # REQUIRED — name of a unique key column already in the data
    time_col=None,        # default "time"
    lon_col=None,         # default "lon"  (CSV only)
    lat_col=None,         # default "lat"  (CSV only)
    app_config=None,      # default: settings.app_config
    store_root=None,      # default: STORE_ROOT
    crs=4326,             # EPSG code for geometry extraction (SHP only)
    time_cadence="auto",  # "auto" | "daily" | "hourly"    — see Cadence
    read_from="auto",     # "auto" | "native" | "compiled" — see Cadence
    bathy_layer=None,     # default: extract_layer of the bathy config entry
    log_file=None,        # default: LOGS_DIR/extractor.log
)
```

| Parameter | Default | Description |
|---|---|---|
| `file_path` | — | A `.csv` / `.shp` path, or an in-memory `pd.DataFrame` (points) / `gpd.GeoDataFrame` (geometries). The suffix or object type selects point vs geometry mode. |
| `index_col` | **required** | Name of the unique key column used to merge results back onto your input. It must already exist in the data and be unique — the Extractor consumes the key, it never creates one. A missing or duplicated key raises `ValueError`. Use [`ensure_row_id`](#establishing-the-merge-key-ensure_row_id) to establish it. |
| `time_col` | `"time"` | Name of the time column in the input. |
| `lon_col` | `"lon"` | Longitude column name (CSV/point input only). |
| `lat_col` | `"lat"` | Latitude column name (CSV/point input only). |
| `app_config` | `settings.app_config` | Override the application configuration (variable registry, depth levels, etc.). |
| `store_root` | `STORE_ROOT` | Root directory of the Zarr stores. Each `var_key` is read from its own `store_root` where `config.yaml` declares one; see [Where a variable's store lives](../configuration.md#where-a-variables-store-lives). |
| `crs` | `4326` | EPSG code that geometries are reprojected to (SHP/geometry input only). |
| `time_cadence` | `"auto"` | How `time_col` is read: `"daily"` truncates to midnight, `"hourly"` keeps the precision, `"auto"` infers. See [Cadence](#cadence). |
| `read_from` | `"auto"` | Which store each `var_key` is read from: its own Zarr (`"native"`), the compiled h2ds (`"compiled"`), or per-`var_key` (`"auto"`). See [Cadence](#cadence). |
| `bathy_layer` | `extract_layer` in config | Which bathy layer (a key of its `layers`, e.g. `"15s"`, `"60s"`, `"0.25deg"`) `bathy` is read from, for points and geometries alike. An undeclared name raises at construction. See [Standard-deviation columns](#standard-deviation-columns). |
| `log_file` | `LOGS_DIR/extractor.log` | Extraction log file. The first `Extractor` in the process fixes this; later values are ignored. |

---

## Establishing the merge key: `ensure_row_id`

`index_col` is required and the `Extractor` never creates it — the key has to exist on the
frame *you* keep, since that is the other side of the eventual join. `ensure_row_id` puts
one there:

```python
from h2mare.processing.extractor import Extractor, ensure_row_id

pts = ensure_row_id(pts)                       # adds "row_id" if absent
results = Extractor(pts, index_col="row_id").run({"sst": None})
```

| Input | Result |
|---|---|
| Column present and unique | Returned unchanged |
| Column present with duplicates | `ValueError` — a duplicated key collapses rows on merge-back |
| Column absent | A positional `range(len(data))` key is added, on a copy |

Use the returned frame for both the extraction and the merge. The key is positional when
generated, which is why the checkpoint fingerprints the input as well: two different frames
of the same length would otherwise line up perfectly.

---

## Cadence

Some variables are stored at their source's native **hourly** cadence
(`time_step: hourly` in `config.yaml`). For those, the daily reduction and every
feature derived from it are produced at compile time and written only to h2ds.
The per-variable Zarr is therefore the raw *source*; **h2ds is the daily
*product***. In the shipped config the hourly variables are the four ERA5 ones —
`atm-instante`, `atm-accum-avg`, `radiation`, `waves` — but nothing is hardcoded:
the same variables converted with `time_step: daily` compute their reductions up
front and hold everything in their own store.

Two independent arguments control this. `time_cadence` decides **how your input
is read**; `read_from` decides **which store answers**.

### `time_cadence` — reading `time_col`

| value | behaviour |
|---|---|
| `"auto"` | infer: a time component that **varies** across rows means hours; a date-only input, or one stamped identically on every row, means days |
| `"daily"` | truncate to midnight regardless |
| `"hourly"` | keep whatever precision is there |

The uniform-stamp case is the one to watch. `14:00:00` on *every* row reads as an
export default rather than a deliberate hour, so `"auto"` truncates it. Pass
`"hourly"` when your rows really do mean 14:00.

```python
Extractor(pts, index_col="row_id", time_cadence="hourly")
```

### `read_from` — choosing the store

| value | behaviour |
|---|---|
| `"auto"` | per `var_key`: a daily store answers for itself; an hourly one answers a sub-daily request and otherwise defers to the compiled store |
| `"native"` | the per-variable Zarr |
| `"compiled"` | h2ds |

`"compiled"` works for **any** `var_key`, including daily ones — useful when you
want every column on the same grid and units as your Parquet-based analysis:

```python
Extractor(pts, index_col="row_id", read_from="compiled").run("sst")
```

### What `auto` resolves to

| store cadence | input cadence | reads |
|---|---|---|
| daily | any | its own Zarr |
| hourly | daily | h2ds |
| hourly | hourly | its own Zarr, plus h2ds for anything derived at compile time |

### Things worth knowing

- **The two stores are not interchangeable.** h2ds is on the 0.25° base grid and
  carries the pipeline's units; an hourly native store holds the raw source as
  published. ERA5 `msl` is **hPa** in h2ds and **Pa** natively; `tp` is a daily
  total versus a per-hour accumulation. Column names are the same either way, so
  a native hourly read logs a warning naming the stored units.
- **Reading a var_key from h2ds needs a current compile.** If h2ds is missing a
  column the `var_key` publishes, extraction raises and names
  `uv run h2mare compile` rather than returning a thinner frame.
- **Some features have no hourly value at all.** `ekman_anom` is a 7-day rolling
  mean against a day-of-year climatology; `wind_max` is a maximum over a day.
  When they are absent from the native store — which happens only when that
  `var_key` converts hourly — they are read from h2ds and each sample takes the
  value for the day it falls in. This holds even under `read_from="native"`,
  with a warning, because the alternative is returning nothing for a column that
  the same variable converted daily would have had natively.
- **A daily store missing what it publishes is an error, not a route.** The rule
  is uniform: absent where the design puts it elsewhere is routed; absent where
  the design says it should be present is a hole in the store, and says so.
- The store your input never reaches is never opened, and h2ds is opened **once**
  per `Extractor` however many `var_keys` a `run()` walks.

---

## `run()`

```python
extractor.run(
    var_dict=None,       # which var_keys / variables to extract; None = all
    output_path=None,    # None → return DataFrame; path → write CSV, return None
    n_workers=8,         # parallel workers for geometry (SHP) extraction only
)
```

| Parameter | Description |
|---|---|
| `var_dict` | Selects what to extract. A `str` (single `var_key`), a `list[str]` (several `var_keys`), or a `dict[var_key, vars]` to pick specific variables inside a `var_key` (e.g. `{"radiation": ["tisr", "ssrd"]}`). `vars` may also be a `{variable: depths}` dict, which picks the variables *and* the depth levels of 3-D ones for this run — see [Depth levels](#depth-levels). `None` extracts every `var_key` in config (excluding compiled `h2mare`-source outputs). |
| `output_path` | If `None`, `run()` returns the result `DataFrame`. If a path is given, the result is written to **CSV** and `run()` returns `None`. |
| `n_workers` | Number of `ThreadPoolExecutor` workers. **Only used for geometry (SHP) extraction**; point (CSV) extraction is vectorized and ignores it. |

```python
var_dict = {"seapodym": [], "radiation": ["tisr", "ssrd", "slhf"]}
extractor = Extractor("input.csv", time_col="date", index_col="id_row")
results = extractor.run(var_dict, output_path="out.csv", n_workers=12)
```

Extraction is checkpointed per `var_key` (`INTERIM_DIR/extraction_checkpoint.feather`), so
an interrupted run resumes where it stopped. The checkpoint is deleted once a run finishes
cleanly, so it only ever survives a failure.

It lives at one fixed path, which means the next run finds whatever the last one left there.
A fingerprint of the input — the key column's name and values, plus every prepared column —
is stored alongside it, and a checkpoint written for anything else is **discarded rather
than resumed**, with a warning. Without that, a different input of the same shape had the
previous run's rows replayed onto it silently: `ensure_row_id` keys positionally (`0..n-1`),
so any two frames of the same length line up perfectly.

A resume logs which `var_keys` it is replaying rather than re-extracting. That matters when
you have changed something *other* than the input — config, or the pipeline itself — since
a fingerprint match cannot tell "carry on where you stopped" from "re-run this with the
fix". Delete `INTERIM_DIR/extraction_checkpoint.*` when you want a genuinely clean run.

The checkpoint is two files — the feather, then a sidecar naming what it holds — so a kill
between them leaves a `var_key`'s columns stored but the key unrecorded. The resume detects
that, discards the stale columns and re-extracts, warning as it goes. The write order is
deliberate: reversed, the same interruption would mark the `var_key` done with its columns
missing and the resume would skip it, dropping the variable silently.

### Depth levels

A store with a `depth` axis is sliced into one column per level, `<variable>_<level>`
(`thetao_50`), before extraction; left unsliced, the axis would be averaged away. The
levels come from, in increasing priority:

1. `depth_levels` in the var_key's config — also what compile publishes;
2. `extract_depth_levels` — an extraction-only override, per variable;
3. the request itself.

```python
extractor.run({
    "dyn_rep": {
        "thetao": [0, 50, 100],  # these depths, for this run only
        "uo": None,              # the levels config gives uo
        "zos": None,             # a 2-D variable, as it is
    },
})
```

A variable the request lists replaces its configured levels; the others keep theirs, and
config is not changed. A plain list (`{"dyn_rep": ["thetao", "zos"]}`) uses config for
every variable. What may be named:

| Name | Meaning |
|---|---|
| a 3-D variable (`thetao`) | all of its levels |
| a level column (`thetao_50`) | that level only — it must be one of the levels in use |
| a 2-D variable (`zos`) | the variable as stored |
| the var_key itself | everything (the older single-variable stores, where `o2` names both) |

Only the variables requested are sliced, so a 3-D variable nobody asked for needs no levels.
Refused rather than guessed:

- a requested 3-D variable with no levels from any of the three sources;
- levels for a variable with no depth axis, or for a store without one;
- levels naming a variable the store does not hold;
- levels in a request answered from the compiled store (an hourly var_key with date-only
  input, or `read_from="compiled"`) — its columns are fixed at compile time, so name them
  (`thetao_100`) instead;
- a level list that is empty, negative, fractional or repeated.

Levels are matched to the store's axis by nearest depth and keep the requested name, so
`thetao_1000` off a store that ends at 902 m holds the 902 m values.

The checkpoint is keyed by `var_key`, not by the levels asked for: after a failed run, a
re-run with *different* levels replays the columns already extracted. Delete
`INTERIM_DIR/extraction_checkpoint.*` when you change them.

---

## `extract_from_dataset()`

Extract against an arbitrary in-memory `xarray` object instead of the store — useful for
new data that has not been ingested yet. It runs the same engine as `run()` but bypasses
`ZarrCatalog`, reusing the points/geometries already prepared in the constructor.

```python
extractor.extract_from_dataset(
    ds,                       # xr.Dataset | xr.DataArray with coords lon, lat, [time]
    *,
    vars=None,                # subset of data_vars (xr.Dataset only)
    n_workers=8,              # geometry (SHP) extraction only
    clip_to_coverage=False,   # drop input rows outside the ds extent → NaN
) -> pd.DataFrame
```

| Parameter | Description |
|---|---|
| `ds` | Gridded data with coords named `lon`, `lat`, and optionally `time`. For geometry input the dataset is assumed to be in `crs` — its CRS is overwritten (not reprojected) to match the geometries. |
| `vars` | Subset of variables to extract. Only valid when `ds` is an `xr.Dataset`; passing it with a `DataArray` raises `TypeError`. |
| `n_workers` | Parallel workers for geometry (SHP) extraction only. |
| `clip_to_coverage` | When `True`, input rows whose location (and time, if `ds` has a time coord) fall outside the `ds` extent are dropped and surface as `NaN` in the result. Default `False`, since nearest-neighbour (CSV) and clip-or-NaN (SHP) already handle out-of-extent inputs. |

Only config-free preparation is applied. Config-driven steps that the store path performs
— depth-level slicing, store selection (`read_from`), and store date/bbox coverage resolution — are
the **caller's** responsibility: prepare `ds` beforehand.

```python
import numpy as np, pandas as pd, xarray as xr

ds = xr.Dataset(
    {"sst": (("time", "lat", "lon"), np.arange(2 * 3 * 3.0).reshape(2, 3, 3))},
    coords={"time": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "lat": [40, 41, 42], "lon": [-10, -9, -8]},
)
pts = pd.DataFrame({"time": ["2021-01-01", "2021-01-02"], "lon": [-10, -8], "lat": [40, 42]})
pts = ensure_row_id(pts)
out = Extractor(pts, index_col="row_id").extract_from_dataset(ds)   # sst at nearest cells
```

---

## Output format

`run()` produces a single tabular result, indexed by `index_col`:

- **Returned as a `DataFrame`** when `output_path` is `None`.
- **Written as a CSV** when `output_path` is given (and `run()` returns `None`). The
  `index_col` key is written as a column so you can merge the result back. If the file
  already exists, overlapping columns are dropped from the existing file and the new
  columns are joined in.

Columns are the **input columns carried through**, plus one column per extracted variable:

| Input | Carried-through columns | Extracted columns |
|---|---|---|
| CSV / points | `time`, `lon`, `lat` | one column per variable (e.g. `sst`, `tisr`); depth-sliced variables expand to `var_<depth>` |
| SHP / geometries | `time`, `geometry` | one column per variable (polygon mean) |

### Standard-deviation columns

`_std` does **not** mean the same thing in every column of a geometry extraction.

For `sst`, `adt` and `sla` the geometry engine only ever computes a mean — `_extract_geometry`
reduces each clip with `.mean()` and nothing else. Their `_std` columns are therefore the
**polygon-mean of a std layer computed upstream**, not a spread measured across the polygon:

| Layer | Computed at | Window | Relative to the 0.25° cell |
|---|---|---|---|
| `sst_std` | convert time, 0.05° native | 3×3 ≈ 0.15° | sub-cell texture |
| `adt_std`, `sla_std` | convert time, 0.125° native | 3×3 ≈ 0.375° | **wider** than the cell |

Both are then placed on the base grid at compile time by the area-weighted mean every coarsened
variable gets, so the stored 0.25° value is the **mean of the native std layer over the cell**.
That is the mean of a local spread, not the spread across the cell: it ignores the variance
*between* windows, so it reads lower than a std computed over the whole 0.25° footprint would.
The definition is deliberate and fixed — it is the same quantity at every resolution, and it is
what the archive has always published. Because the windows differ in physical size, `sst_std` and
`adt_std` magnitudes are not comparable with each other.

`bathy_std` follows the same rule. Each native bathy layer stores a 3×3 rolling std built by
`scripts/bathymetry.py` (≈ 1.4 km at 15″, ≈ 5.5 km at 60″), so a point takes the nearest cell's
value and a geometry the polygon mean of that layer — the same estimator for CSV and SHP, chosen
by `bathy_layer` rather than by input type. The 0.25° layer (what h2ds holds) is different again:
the std of all 15″ cells inside each 0.25° cell. The 15″ and 60″ values are not on a common scale.

Averaging a stored std layer is the deliberate choice. A within-polygon std is
polygon-size dependent — a polygon touching one 0.25° cell yields `0` or `NaN`, a large one is
dominated by the regional gradient — so it is not comparable across rows of differing geometry
size, whereas the layer mean is defined even for a single-cell polygon and stays on a fixed
physical scale.

Two related traps. `analysis_error` (shipped with the SST product) is retrieval uncertainty, not
spatial variability, and substitutes for neither quantity. And if you ever do want variability at
both scales at once, the law of total variance gives it from layers already stored — no new
extraction path needed:

```
Var_total(polygon) ≈ mean_i(σ_i²) + var_i(μ_i)
```

with `σ_i` the stored `*_std` and `μ_i` the stored mean field over the clipped cells. Note that
averaging `σ` rather than `σ²` understates variability (Jensen), so pool the variances and take
the square root at the end.

The in-memory `run()` return keeps the SHP `geometry` column as a `GeoDataFrame` with live
shapely geometries (restored from the input on a checkpoint resume, since feather cannot
round-trip them). It is dropped only when writing to CSV, where shapely geometries would
serialize to WKT strings that cannot be read back as geometries.

---

## Parallelism

Geometry (SHP) extraction runs each geometry through a `ThreadPoolExecutor`, sized by the
`n_workers` argument (default `8`). Point (CSV) extraction is fully vectorized — it uses a
cached KDTree for the nearest grid cell and `searchsorted` for the nearest time — and does
not use `n_workers`.
