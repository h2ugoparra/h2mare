# Compiler

`Compiler` reads every per-variable Zarr store, puts each one on a common daily
grid, and writes the merged `h2ds` dataset, one file per year by default. The grid
is declared on the `h2ds` entry in `config.yaml` (0.25°, cell-centred, in the
shipped config). How a variable gets onto it — an area-weighted mean when its own
grid is finer, linear interpolation when it is the same or coarser — is decided per
variable from the two resolutions, so nothing about it has to be declared. See
[Regridding](#regridding).

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
    file_period=FilePeriod.YEAR,
    date_format="year",
)
```

| Parameter | Default | Description |
|---|---|---|
| `var_key` | `"h2ds"` | Output variable key. Its config entry defines the output grid (`bbox`, `cells_per_degree`, `values_at`) and store (`local_folder`); see [The compile grid](#the-compile-grid) |
| `app_config` | settings | Override the application configuration |
| `remote_store_root` | `STORE_ROOT` | Default root for source Zarr stores. A source variable declaring its own `store_root` in `config.yaml` is read from there instead; see [Where a variable's store lives](../configuration.md#where-a-variables-store-lives) |
| `local_store_root` | `ZARR_DIR` | Local copy destination for the compiled output |
| `file_period` | `YEAR` | Output file granularity: `YEAR` or `MONTH` |
| `date_format` | `"year"` | Output filename date format: `"year"`, `"yearmonth"`, or `"date"` |

---

## `run()`

```python
Compiler().run(
    start_date=None,
    end_date=None,
    var_keys=None,
    zarr_backup=False,
    zarr_backup_dir=None,
)
```

| Parameter | Description |
|---|---|
| `start_date` | Start of compilation period. Inferred from the stores if `None` |
| `end_date` | End of compilation period. Inferred from the stores if `None` |
| `var_keys` | List of variable keys to include. Defaults to all keys in `config.yaml` |
| `zarr_backup` | Copy compiled Zarr files to the local backup store. Defaults to `False` |
| `zarr_backup_dir` | Override backup destination. Defaults to `local_store_root` |

The method builds the output grid, checks it against the store already on disk,
splits the requested range into yearly (or monthly) chunks, runs each variable's
processor on each chunk, merges the results with `xr.merge`, and writes via
`write_append_zarr`. After all chunks are written, the compiled files are copied to
`local_store_root` (or `zarr_backup_dir`) only when `zarr_backup=True`.

Variables with no data for a given chunk are skipped with a warning rather than
raising an error. Compiling a subset (`var_keys=[...]`) writes only those variables'
columns; the rest of the store is preserved and catches up on the next full compile.

---

## The compile grid

The output grid is built from three keys of the `h2ds` entry, not passed to `run()`:

| Key | Shipped | Meaning |
|---|---|---|
| `bbox` | `[-80, 0, 10, 70]` | Extent, `[xmin, ymin, xmax, ymax]` in degrees |
| `cells_per_degree` | `4` | Step, as a whole number of cells per degree: 4 is 0.25° |
| `values_at` | `cell_center` | Where the values sit: cell centres (…, 0.125, 0.375, …) or grid-line crossings (…, 0.00, 0.25, …) |

The shipped grid is 360 × 280 cells, lon −79.875 … 9.875 and lat 0.125 … 69.875.
The step is a count rather than a decimal so it is exact (`0.083` is not 1/12°); see
[`cells_per_degree`](../configuration.md#variable-entries) and
[Where the values sit](../configuration.md#where-the-values-sit).

**A different grid needs a new store.** Before the first chunk is read, the grid is
compared with the one the existing `h2ds` files hold, and a different step or phase
is refused. Appending would otherwise merge the two lattices by outer join into one
axis holding both, each variable NaN at the other's cells. To compile at another
resolution, give it its own entry (`local_folder`, `dataset_id_rep`) and compile
into that. Widening `bbox` is fine: it adds cells to the same lattice.

---

## Regridding

Every variable is put on the compile grid by
[`regrid_to`](#calling-it-yourself) (`h2mare.utils.spatial`), whichever processor
opened it. The method follows from comparing the variable's native step with the
grid's, so moving to a finer or coarser `h2ds`, or adding a source at a new
resolution, needs no configuration.

### How the method is chosen

```
ratio = max(target step / native step)   over lat and lon
```

| Ratio | Direction | Method | What it does |
|---|---|---|---|
| > 1.01 | coarsening — the source is finer than the grid ("downscaling" the resolution) | `conservative` | Area-weighted mean of every source cell overlapping the target cell |
| ≤ 1.01 | same resolution, or refining — the source is coarser ("upscaling") | `linear` | Bilinear interpolation at the target cell centres |

The native step is measured from the store's own axis, end to end, so the ~1e-5°
rounding stored axes carry (`GRID_COORD_DECIMALS`) cannot make a 0.25° store read as
slightly coarser than a 0.25° grid. The 1% tolerance is for the same reason. An axis
that is not regularly spaced is refused: both methods assume a rectilinear grid.

What that means for the shipped variables at 0.25°:

| Variable | Native step | Ratio | Method |
|---|---|---|---|
| `sst` | 0.05° | 5 | conservative: 25 source cells per output cell |
| `chl` | 1/24° | 6 | conservative |
| `thetao`, `mld`, `seapodym` | 1/12° | 3 | conservative |
| `eddies` | 1/12° (own raster) | 3 | conservative, except the columns set to `nearest` (below) |
| `ssh` | 0.125° | 2 | conservative |
| ERA5 (×4), `o2` | 0.25° | 1 | linear, off by half a cell: see [Phase](#phase-at-the-same-resolution) |
| `bathy` (`0.25deg` layer) | 0.25° | 1 | linear, aligned: an exact copy |
| `waves` | 0.5° | 0.5 | linear |

The choice is logged per variable at DEBUG
(`[sst] regrid: 0.05° → 0.25° (ratio 5) via conservative`); set
`H2MARE_LOG_LEVEL=DEBUG` to see it in `pipeline.log`.

### Coarsening: the area-weighted mean

Each output cell is the mean of the source cells overlapping it, each weighted by
how much of the output cell it covers. Longitude weights are overlap lengths.
Latitude weights are overlap in `sin(lat)`, which is the true area of the band
between two parallels, so a cell's poleward rows do not count as much as its
equatorward ones.

**NaN cells are skipped, and the mean is normalised by the valid area.** A cell that
is part land reports the mean of its water, rather than being dragged towards zero
or dropped. Any valid area is enough to give a cell a value, so the product reaches
as far into the coast as its source does (74,993 `sst` sea cells on 2024-02-15,
against 71,743 when this was a point interpolation). `regrid_to` takes a
`min_coverage` fraction to require more; the compile leaves it at 0.

The weights are applied one axis at a time as sparse matrices, lazily and
block-parallel under dask. A source cell meets at most two target cells per axis,
so the matrices are well under 1% non-zero, and nothing is formed at full
resolution in memory.

### Same resolution and refining: linear

Bilinear interpolation at each target cell centre. Unlike the mean, it cannot skip
a NaN neighbour: a target cell with any of its four source neighbours on land comes
out NaN. That matters little when refining (`waves`) and is the reason coarsening
does not use it — sampling 4 of 25 source cells, it both ignored most of each
output cell and lost coastal cells.

### Phase at the same resolution

At ratio 1 the result depends on where the two grids put their values. Aligned
(the `bathy` layer), each target centre coincides with a source value and the
field is copied through exactly. Half a cell off, each target value is the average
of the four source cells around it, which costs roughly 3–11% of a field's own
spatial variability, concentrated where its gradients are. The shipped `h2ds` is
cell-centred and ERA5 and `o2` sit on grid lines, so those are averaged. The
compile logs a warning once per variable in that position. The trade-off and the
measured costs are in [Where the values sit](../configuration.md#where-the-values-sit).

### Overrides: `nearest` and per-column methods

`regrid` on a source variable's config entry sets the method per column:

```yaml
eddies:
  regrid:
    ac_track: nearest
    c_track: nearest
```

`nearest` is never chosen automatically. It takes the value of the nearest source
cell with no averaging, which is what a column a mean would destroy needs: an
identifier, a class, or a property of some *thing* rather than of the cell. The
eddy columns are the shipped case: `ac_track` averaged with its neighbour names an
eddy that does not exist, so the ten identity and property columns are `nearest`,
while `ac_dist_km` and `ac_normdist`, which are continuous, stay on `auto`. `linear`
and `conservative` can be named too, to pin the automatic choice.

`nearest` does not check how far the nearest source cell is, so the source store
must cover the whole compile `bbox` (the eddies store is written on it).

A variable with overrides is split by method, each group is regridded on its own,
and the groups are merged back with an exact join, so every part must land on
identical coordinates. Column names in `regrid` are checked against `compiled_vars`
at config load.

**Directions are not regridded as numbers.** `waves`' mean wave direction `mdts` is
turned into unit-vector components, the components are regridded, and the direction
is recombined. Averaging 350° with 10° directly would give 180°.

### Calling it yourself

```python
from h2mare.utils.spatial import GridBuilder, regrid_to
from h2mare.types import BBox

target = GridBuilder(BBox(-20, 30, 0, 45), 0.25, 0.25).generate_grid()
out = regrid_to(
    ds,                          # lat/lon both increasing, regularly spaced
    target,
    method="auto",               # or "linear", "conservative", "nearest"
    methods={"flag": "nearest"}, # per-variable overrides
    min_coverage=0.0,            # conservative only: minimum valid fraction per cell
    label="my_var",              # names the variable in the log lines
)
```

`out` carries the target's own coordinate arrays, so separately regridded datasets
merge rather than union. Variables without both `lat` and `lon` pass through
unchanged.

---

## Per-variable processing

What a var_key contributes is decided by `compiler_registry.COMPILE_PROCESSORS`.
Every processor ends in the regrid above.

| Var_key | Processor | Behaviour |
|---|---|---|
| `bathy` | `_compile_bathy` | Read from the `compile_layer` of its `layers` (the 0.25° mean/std netCDF), clipped to `bbox`. Static: no time axis |
| `moon` | `_compile_moon` | Lunar illumination computed with `ephem` for each day at the grid's centre, broadcast to all sea cells |
| `atm-accum-avg` | `_compile_atm_accum_avg` | A daily store already holds the Ekman features. An hourly store holds raw stress and precipitation, so the whole Ekman chain runs here, in memory-bounded slabs. `dayofyear`/`month`/`quantile` coordinates dropped before merge |
| `atm-instante` | `_compile_atm_instante` | An hourly store is reduced to its daily aggregates here (wind, cloud, pressure); a daily store is only regridded |
| `waves` | `_compile_waves` | Reduced to daily if the store is hourly; `mdts` regridded through its unit-vector components |
| `sst` | `_compile_sst` | `sst_fdist` clipped to ≥ 0, then regridded |
| any var_key with `depth_levels` | `_compile_depth_var` | Each listed 3-D variable sliced at its configured depths (`thetao_0`, `o2_100`, …) before regridding; hourly 3-D stores are refused |
| anything else | `compile_default` | Opened from its catalog, reduced to a daily **mean** if the store is hourly, regridded. Refuses a store that still has a depth axis. An accumulated hourly variable needs its own processor, since a mean of accumulations is plausible-looking and wrong |

h2ds is daily whatever cadence a source is kept at: an hourly store handed over
unreduced would union its 24 stamps a day into the output axis.
