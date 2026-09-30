# Resolution-aware regridding for the compile step

Status: proposal, nothing implemented.
Written 2026-09-18 against `dev` @ 7edf09e.

## 1. Problem

Every compile processor puts its variable on the base grid with the same call:

```python
ds.interp_like(compiler.base_grid, method="linear", assume_sorted=True)
```

`interp` samples the field *at the target cell centres*. That is the right tool when the
target is the same resolution or finer. It is the wrong tool when the target is coarser,
because it reads the 2×2 source cells around each centre and ignores every other source
cell in the target footprint.

The native grids (measured from the stores on 2026-09-17, not from config):

| Store | Native step | Ratio at 0.25° | Ratio at 1/12° | Registration |
|---|---|---|---|---|
| sst | 0.05 | **5** | 1.67 | cell-centred (0.025 + k·0.05) |
| chl | 1/24 | **6** | 2 | cell-centred |
| fsle | 0.04 | **6.25** | 2.08 | cell-centred |
| mld, thetao | 1/12 | **3** | 1 | node (k/12 from 0) |
| eddies | 0.1 | **2.5** | 0.83 | node |
| ssh | 0.125 | **2** | 0.67 | cell-centred |
| atm-instante, atm-accum-avg, radiation, o2 | 0.25 | 1 | 0.33 | node |
| waves | 0.5 | 0.5 | 0.17 | node |
| bathy (`data_file_hires`) | 1/240 | 60 | 20 | cell-centred |
| bathy (`data_file`) | 0.25 | 1 (identity) | n/a | cell-centred |

Ratio = target step ÷ native step. Above 1 the compile coarsens. **At today's 0.25° grid,
eight of eleven gridded sources are coarsened by a single-point sample.** This is not an
SST-only issue; it was found while looking at SST.

### 1.1 Measured consequences at 0.25°

SST (0.05° → 0.25°, 5×5 blocks, centres coincide so `interp` returns exactly the centre
cell — verified equal to 1e-16), compared against a cos(lat)-weighted block mean, for
2024-02-15 and 2024-08-15:

| Cells | Median abs diff | RMSE | p99 | Max |
|---|---|---|---|---|
| All ocean (~72k) | 0.012 °C | 0.05–0.06 | 0.22–0.27 | 2.2 |
| Coastal (partial-ocean cells, 1.7k) | 0.02–0.07 | 0.08–0.19 | 0.31–0.71 | 1.3 |
| Gulf Stream box (−75..−50, 35..45) | 0.03 | 0.11–0.14 | 0.43–0.54 | 2.2 |

Bias is zero; offshore the differences are below the product's own analysis error. Two
findings are more serious than the values:

1. **Lost coastal cells.** Weighted mean keeps 74,595 ocean cells, `interp` keeps 71,743
   (−3.8%); 847 of the lost cells are ≥50% ocean. Part of that is a plain bug: picking the
   centre cell alone would keep 72,474. Linear interp loses a further **731 cells** whose
   only non-finite neighbour carries weight exactly 0, because `0 * NaN` is NaN.
   `method="nearest"` returns identical values without losing them.
2. **Categorical fields are being averaged.** `ac_track` / `c_track` are eddy trajectory
   IDs. In h2ds 2024-03-01, **4,223 of 70,356** `ac_track` values are non-integer
   (`c_track`: 4,539; `ac_ndays`: 4,254). The native store has zero. Linear interpolation
   between two neighbouring cells' IDs produces an ID that identifies nothing.
3. **Node-registered sources are smoothed even at ratio 1.** ERA5 and o2 sit on whole
   degrees/quarters; h2ds cell centres sit half a cell away, so `interp` averages 2×2
   neighbours. Unavoidable given the target registration, but it should be recorded, not
   discovered.

Bathy is fine and needs no change: `data_file` is already a 60×60 block mean of the 15″
data (matches a plain block mean to 0.002 m; cos-weighting would move it ≤2 m) and is
already on the base grid, so `interp_like` is an identity there (verified).

### 1.2 Why this blocks a finer grid

At 1/12° the ratios change but the dispatch problem does not: sst/chl/fsle still coarsen
(1.67–2.08), thetao/mld land at ratio 1, and o2/ERA5/waves/ssh now *refine*. A hardcoded
`interp` is wrong for the first group; a hardcoded block mean would be wrong for the
third. The step has to be chosen per variable from the two grids.

## 2. Design

### 2.1 Measure the native step from the store

Not from config. Two reasons: config drifts from the data, and several stores round their
coordinate labels to 4 dp on write (`snap_grid_coords`, `GRID_COORD_DECIMALS = 4`), so
consecutive diffs are not the true step. Measured jitter: thetao/mld/chl diffs vary by
~1e-4; `np.median(np.diff(lat))` reads 0.083302 for a 1/12 grid and 0.0042 for a 1/240 one.

```python
def axis_step(values: np.ndarray) -> float:
    """Mean spacing of a regular axis, tolerant of 4 dp label rounding."""
    return (values[-1] - values[0]) / (len(values) - 1)
```

plus a regularity check (max deviation from `linspace` below, say, 2× the rounding
quantum) so an irregular axis raises instead of being silently treated as regular.

### 2.2 Dispatch on the ratio

Per axis, `r = target_step / native_step`, with a relative tolerance of ~1% so a
4 dp-rounded 0.25° store does not flip branch on `r = 1.0004`.

- `r <= 1 + tol` → **linear interpolation** (same resolution or refining). Keeps today's
  behaviour for those variables exactly.
- `r > 1 + tol` → **area-weighted mean** (conservative).

Both axes are evaluated; if they disagree (no current store does), aggregate when either
axis coarsens, since only the aggregating path is correct when any axis is downsampled.

### 2.3 Conservative weights, kept simple

Our grids are all regular rectilinear lat/lon, so the 2-D overlap weight factorises into
two 1-D matrices. For one axis, with source cell edges `se` and target cell edges `te`:

```
W[i, j] = max(0, min(se[i+1], te[j+1]) - max(se[i], te[j]))
```

i.e. the length of source cell *i* inside target cell *j*. Latitude weights are multiplied
by the true cell area factor `sin(lat_hi) - sin(lat_lo)` rather than `cos(lat_mid)` — same
cost, exact. Edges come from the *fitted* regular axis (`x0 + i·step`), not from the
rounded labels, so 4 dp jitter cannot leak into the weights.

**The matrices are small enough to keep dense.** Largest case is bathy at 1/12°:
16,800 × 840 float64 = 113 MB, and the routine one is sst→0.25°, 1400 × 280 = 3 MB. No
sparse dependency, no `xesmf` (needs ESMF, painful on Windows), no `xarray-regrid`.

NaN-aware application, separable, two `xr.dot` passes per axis:

```
m     = da.notnull()
numer = dot(dot(da.fillna(0) * m, W_lat), W_lon)
denom = dot(dot(m,               W_lat), W_lon)
out   = numer / denom.where(denom > 0)
cover = denom / dot(dot(ones,    W_lat), W_lon)   # valid-area fraction per target cell
```

`out` is then masked where `cover < min_coverage` (see open question 8.1). This works on
dask arrays; `xr.dot` over one dim at a time avoids materialising a 2-D weight tensor.

### 2.4 One helper, one call site shape

```python
def regrid_to(
    ds: xr.Dataset,
    target: xr.Dataset,
    *,
    method: Literal["auto", "linear", "nearest", "conservative"] = "auto",
    min_coverage: float = 0.0,
) -> xr.Dataset
```

Lives in `h2mare/utils/spatial.py` next to `GridBuilder`. It replaces the seven
`interp_like(compiler.base_grid, ...)` calls in `processing/compiler_registry.py`
(lines 97, 162, 413, 471, 511, 528, 595). It logs one line per variable naming the
measured native step, the ratio and the method chosen, the way `chunk_dataset` logs
its layout.

**It must assign the target's own coordinate objects to the result**, not recomputed
values. `Compiler.run` merges the per-variable results with `xr.merge(..., join="outer")`;
coordinates differing in the last float bit would union into a doubled axis instead of
aligning. `interp_like` gives this for free today, so the helper has to preserve it
deliberately.

### 2.5 Per-variable semantics the ratio cannot decide

The ratio picks *how to resample a continuous field*. It cannot know what the field means.

| Variable(s) | Correct handling | Why |
|---|---|---|
| `ac_track`, `c_track` | `nearest` (never averaged) | Trajectory IDs. Already corrupt in h2ds (§1.1). |
| `ac_ndays`, `c_ndays` | `nearest` | Age of the *nearest eddy*, an attribute of that eddy, not a field. |
| `ac_amp`, `ac_speed`, `ac_speedrad_km` (+ `c_*`) | `nearest` (proposed) | Same: attributes of the nearest eddy, piecewise-constant over its Voronoi cell. Averaging mixes two eddies. |
| `ac_dist_km`, `ac_normdist`, `sst_fdist`, `chl_fdist` | mean | Genuinely continuous distance fields. |
| `mdts` | mean of u/v, then recombine | Already handled in `_compile_waves`; stays correct under a weighted mean (it is linear in the components). |
| `sst_std`, `adt_std`, `sla_std`, `wind_std` | mean (documented) | A mean of stds, not a std of the cell. Defensible, but it must be written down. See open question 8.2. |
| `tp`, radiation, `gke`, `ekman_*` | mean | Intensive per unit area (mm, W/m², J/kg), so the area-weighted mean is the conserving answer. |
| `bathy`, `bathy_std` | unchanged for now | `data_file` is already the aggregate; see §3 phase C. |

Mechanism: an optional `regrid` key on the config entry, `{var_name: method}`, read by the
helper. Defaulting to `auto` keeps every unlisted variable on the ratio rule. Putting it in
config rather than in the registry keeps it next to `derived_vars` and `depth_levels`,
which are the same kind of per-variable declaration.

### 2.6 Target grid definition

Two changes to how the grid itself is built.

**a. `GridBuilder.generate_grid` must not use `np.arange` with a fractional step.**
Today (`utils/spatial.py:83`):

```python
lat = np.arange(self.ymin + (self.dy / 2), self.ymax + (self.dy / 2), self.dy)
```

With `dy = 0.25` this is exact (0.25 is a binary fraction). With `dy = 1/12` the length is
`ceil((ymax - ymin)/dy)` computed in floating point and can come out one element long, and
the accumulated labels drift. Replace with an explicit count:

```python
n = int(round((self.ymax - self.ymin) / self.dy))
lat = self.ymin + (np.arange(n) + 0.5) * self.dy
```

Deterministic across runs, which matters because each yearly h2ds file is written
separately and `ZarrReader` compares axes exactly.

**b. The resolution must be declared, not hardcoded.** Today it is `DX = DY = 0.25` at
`processing/compiler.py:40`, used only as the default of `Compiler.run(dx, dy)`, which no
caller overrides (`cli/compile.py:106`, `pipeline_manager.py:171`). Move it onto the
`h2ds` config entry — `docs/configuration.md:146` already claims that entry "defines the
output grid", which is currently only true of `bbox`.

Declare it as an integer count per degree so it is exact and self-documenting:

```yaml
  h2ds:
    local_folder: h2ds
    dataset_id_rep: compiled-data-0.25deg-P1D
    bbox: [-80, 0, 10, 70]
    cells_per_degree: 4        # 0.25°; 12 → 1/12°
```

`dx = 1 / cells_per_degree`. Writing `0.083` instead would put 1,084 cells across the 90°
span and end ~0.3° off the 1/12 lattice. Keep `Compiler.run(dx, dy)` as an override for
experiments, defaulting to the config value rather than to a module constant.

**c. Registration stays cell-centred.** No single choice matches every source (sst, chl,
fsle are cell-centred; the CMEMS models, ERA5 and eddies are node-registered), and h2ds,
the Parquet layer and the Extractor all assume the current convention. Accept the half-cell
offset for thetao/mld at 1/12° and record it (§2.7).

### 2.7 Provenance

At 1/12°, ERA5, o2, waves and ssh *look* 1/12° and are not. Per variable, on write:

- `native_resolution_deg` and `regrid_method` (`linear` / `nearest` / `conservative`).
- CF `cell_methods: "area: mean"` where the weighted mean was used.

These are computed from what actually happened, so they belong with `provenance.py`'s
`extent_root_attrs` (which already derives `time_coverage_resolution` from the axis), not
in the static `variable_attrs` table. Note `apply_cf_attrs` is shared with the convert
path, so a per-variable `cell_methods` added at compile must not leak into native stores.

## 3. Phases

**Phase A — the helper, still at 0.25°.**
`regrid_to` + step measurement + dispatch, wired into the seven call sites. No config
changes. This is where the h2ds values change.
Verification: unit tests per §5, then a diff of one recompiled year against the current
store, expecting the §1.1 magnitudes and no change for the ratio ≤ 1 variables.

**Phase B — per-variable overrides.**
`regrid` config key; `nearest` for the eddy attribute columns. Fixes the fractional
track IDs.

**Phase C — bathy from the 15″ file (optional).**
With the general weights, `_compile_bathy` can read `data_file_hires` and produce
`bathy` *and* `bathy_std` (`std = sqrt(E[x²] − E[x]²)` using the same weights) at any
resolution, removing the resolution-specific `data_file`. Keep `data_file` regardless:
`Extractor._extract_bathy` reads it directly for the csv/point path
(`extractor.py:1793-1798`), independent of the compile grid.

**Phase D — configurable resolution.**
`cells_per_degree`, `GridBuilder` fix, the store guard (§4.1), provenance attrs. After
this, 1/12° is a config change plus a full recompile.

Phases A/B are worth doing on their own even if the finer grid never happens.

## 4. Compatibility review

Everything below was checked against the code; verdicts are what the design must respect.

### 4.1 Changing resolution requires a new store, and a guard

- `write_append_zarr` → `_append_data` merges incoming data with what is on disk
  (`storage.py:363`, `:440`). Compiling 1/12° into the existing 0.25° h2ds would
  **outer-join two grids into one doubled axis**, silently. Nothing currently prevents it:
  `Compiler.run(dx=...)` is a free parameter.
  → **Required:** on open, compare the target grid with the store's existing axes and
  refuse on mismatch, naming both. Cheap (one coordinate read) and it is the single most
  destructive failure available here.
- A different resolution means a different `local_folder` *and* `dataset_id_rep` (the
  latter feeds file names and, via `zarr2parquet`, the Parquet folder name —
  `zarr2parquet.py:158`). `compiled-data-0.25deg-P1D` would otherwise lie.
- **Only one compiled store can exist at a time.** `var_routing.compiled_var_key` raises if
  two config entries declare `source: h2mare` (`var_routing.py:62`). A second entry would
  also be picked up as a *source* variable by `Compiler.run` (it iterates all config keys
  and skips only `self.var_key`) and by the download loops. So switching resolution is an
  edit to the single `h2ds` entry, not a second entry beside it. Supporting both at once is
  a larger change (generalising `SYSTEM_VAR_KEYS`, `models.py:416`) and is out of scope.

### 4.2 Coordinate labels and axis alignment

- `snap_grid_coords` rounds lon/lat to 4 dp on every write. At 1/12° the centres
  (0.0417, 0.125, 0.2083, …) stay distinct, so the guard inside it does not trip, but the
  stored labels are *not* exactly `x0 + i·dx`. Consequences: (a) the step estimator must be
  the endpoint-based one in §2.1; (b) weights must be built from fitted edges, not labels.
  Both are already in the design. At 0.25° nothing changes (0.125 is exact at 4 dp).
- `ZarrReader.AXIS_SNAP_TOL` is 1e-9 for lat/lon — a drift tolerance, not a resolution
  assumption. Still ~7 orders below a 1/12° step. The comment at `zarr_reader.py:36`
  ("coarsest grid step in the pipeline is 0.25") needs rewording only.
- `xr.merge(join="outer")` in `Compiler.run:278` — covered by §2.4 (assign target coords).

### 4.3 Downstream consumers

| Consumer | Verdict |
|---|---|
| `zarr2parquet` | Works; folder name follows `dataset_id_rep`, so a renamed store writes a new folder. At 1/12° row count is **9×** — the store and every `ParquetIndexer` scan grow accordingly. |
| `parquet2zarr` | Config-free; rebuilds the grid from the rows themselves (`_long_df_to_grid`). No assumption to break. |
| `export_map_zarr` / `h2ds_map` | Pure projection, no grid assumption. `chunk_dataset(layout="map")` pins time to `map_time_chunk=14` and fills the space axes to the 32 MB budget, so a finer grid yields more, not larger, chunks. Fine. |
| `Extractor` (points) | KDTree cache key is `(shape, first, last)` per axis (`extractor.py:1587`), so a different grid simply keys a different tree. Only the comment at `extractor.py:64` is stale. |
| `Extractor` (geometry) | Uses `rio.clip` on h2ds; finer cells mean more cells per polygon, which is an improvement. Coordinate attrs still required (already handled by `apply_cf_attrs`). |
| `Extractor` (bathy csv path) | Reads `data_file` directly — unaffected by phases A/B/D; see phase C note. |
| `cli/catalog` bbox check | `_BBOX_TOL_DEG = 0.5` (`cli/catalog.py:34`) is wider than any half-cell offset at either resolution. Fine; comment stale. |
| `audit` | No resolution assumption found (gap detection is temporal). |
| `sel_padded_bbox` | Pads the read by one source cell each side, which is exactly what conservative weights need at the bbox edge. Keep. |
| `postprocess_sst_fdist` | Clips `sst_fdist` ≥ 0 because `interp` undershoots. A weighted mean of non-negative values cannot go negative, so the clip becomes a no-op. Harmless; keep. |
| `compile_default` depth guard | Unchanged — still refuses a store with a depth axis. |
| Hourly path (`_reduce_hourly_in_slabs`) | Reduction happens at native resolution before the regrid; slab sizing (`_slab_days`) reads the *source* dataset, so it is unaffected. At 1/12° the daily result is larger but it is materialised after reduction, and the source budget is what bounds memory. |
| `clip_land_data` | Only used by `_compile_moon`; follows whatever grid it is given. |
| `global_land_mask` | Its own resolution is independent of ours; it is queried per target cell. |

### 4.4 Cost

- Phase A/B on the current grid: full recompile of h2ds (all years, all variables) plus a
  Parquet rebuild, because every affected column changes value.
- 1/12° later: ~9× cells. h2ds and the Parquet store scale with it; compile time is
  dominated by the source read, which does not change, plus the dot products, which are
  cheap relative to it.

## 5. Tests

Per CLAUDE.md, each regression test must fail on unfixed code (`git stash push` the source
file, run, `git stash pop`).

1. `axis_step` recovers 1/12 and 1/240 from 4 dp-rounded labels; raises on an irregular axis.
2. Dispatch: ratio 5 → conservative; ratio 1.0004 (rounding) → linear; ratio 0.33 → linear.
3. Conservative correctness: on an aligned whole-number ratio the result equals
   `coarsen(...).mean()` (the check already run in scratch); weights per target cell sum
   to 1; a constant field regrids to the same constant (including across a partial edge cell).
4. NaN policy: a target cell with one land source cell returns the mean of the valid ones,
   not NaN (the 731-cell bug); `min_coverage` masks as configured.
5. Coordinate identity: the result's lat/lon are bit-identical to `base_grid`'s, so two
   processors' outputs merge without unioning.
6. Per-variable override: `ac_track` regrids to values present in the source (integers
   preserved), while `ac_dist_km` on the same store is averaged.
7. Guard: compiling a grid that differs from the existing store raises, naming both.
8. `GridBuilder` at `dy = 1/12` over 0–70 returns exactly 840 cells with the first centre
   at 1/24 (fails today).
9. Refinement direction unchanged: a ratio < 1 variable produces exactly what
   `interp_like` produces today (pins that phase A does not touch ERA5/o2/waves).

## 6. Docs to update

`docs/architecture.md:59,65`, `docs/api/compiler.md:3,44-56,70-74`,
`docs/configuration.md:146-151` (+ the new keys), `docs/variables.md:31,37,42`
("resampled to 0.25°" → say how), `docs/api/extractor.md:323-338` (the `_std` section
states the layers are point-sampled at compile — that stops being true), `docs/index.md:15`,
and the CLAUDE.md gotchas.

## 7. What this does not change

- The convert step. `derived_vars` (`sst_std`, `gke`, …) are still computed on the native
  grid, with windows defined in native cells, so their spatial scale still does not follow
  the target grid. Unchanged by this work and still worth documenting.
- Time. Cadence handling, slabbing and the hourly→daily reductions are untouched.
- The Parquet write path and its overlap semantics.

## 8. Open questions

1. **`min_coverage` default.** What fraction of a target cell must be valid (ocean) for it
   to get a value? `0.0` (any ocean counts) recovers all 2,852 cells at 0.25°; `0.5`
   recovers 847 and keeps the coastline closer to today's. This is a data-product decision:
   it sets how far h2ds extends into the coast, and it interacts with extraction for
   near-shore samples.
2. **`_std` estimator.** Keep the mean of stds (simplest, current meaning preserved) or
   switch to RMS? Changing it changes the columns' meaning as well as their values.
3. **Eddy attribute columns.** Is `nearest` right for `ac_amp` / `ac_speed` /
   `ac_speedrad_km`, or is a mean acceptable there? `*_track` and `*_ndays` are not in
   question.
4. **Recompile history, or only go forward?** A partial recompile leaves one store with two
   regridding regimes across a date boundary, which is worse than either. Recommendation:
   recompile everything, in one pass, when phase A lands.
