# Distance metric defect: `*_dist_km`, `*_normdist`, `sst_fdist`, `chl_fdist`

Status: code fixed on `fix/eddy-distance-metric`; **stored data not yet repaired**.
Written 2026-09-18. Found while checking whether the 0.1° eddies store needed
regenerating for the regrid work (`plans/regrid.md`) — it does not, but it needs
regenerating for this.

## 1. The defect

`haversine_min_distance_kdtree` (`h2mare/utils/spatial.py`) was not haversine. It
built a KD-tree on `(lat, lon)` **in radians** and took the Euclidean distance:

```python
tree = KDTree(np.radians(coords2))
distances, _ = tree.query(np.radians(coords1), k=1)
return distances * _EARTH_RADIUS_KM
```

That is a plane, not a sphere. It omits the `cos(lat)` factor on longitude, so a
degree of longitude counts as 111 km at every latitude:

| Latitude | True E–W distance, 1° lon | What it returned | Overstated by |
| --- | --- | --- | --- |
| 0° | 111.2 km | 111.2 km | 1.00× |
| 40° | 85.2 km | 111.2 km | 1.31× |
| 60° | 55.6 km | 111.2 km | 2.00× |
| 70° | 38.0 km | 111.2 km | 2.92× |

The docstring asserted the approximation was "close enough for nearest-neighbour
ranking over typical oceanographic spatial scales", which is what kept anyone
from checking. It is exact only on the equator and for purely meridional offsets.

**Second consequence — two searches that disagree.** `aviso._process_daily_static`
measures the distance with this function but reads the eddy's attributes with
`find_nearest_vectorized`, which projects onto the unit sphere and is correct. On
the old metric the two picked a *different eddy* for **13.6%** of cells over the
h2mare bbox, rising to 34.6% at 60–70°N. So a cell's `ac_dist_km` described one
eddy while its `ac_track`, `ac_amp`, `ac_speed`, `ac_ndays` and `ac_speedrad_km`
described another.

## 2. What it affects

| Column | How | Where it is stored |
| --- | --- | --- |
| `ac_dist_km`, `c_dist_km` | distance is the value | eddies native store + h2ds |
| `ac_normdist`, `c_normdist` | `dist / effective_radius` | eddies native store + h2ds |
| `ac_track`, `ac_ndays`, `ac_amp`, `ac_speed`, `ac_speedrad_km` (+ `c_*`) | attributed to the wrong eddy where the two searches disagreed | eddies native store + h2ds |
| `sst_fdist`, `chl_fdist` | distance to the nearest front (`processing/core/fronts.py:274`) | sst / chl native stores + h2ds |

Measured, recomputing 1998-01-01 over the whole bbox with both metrics:

| Latitude band | `dist_km` overstated by | Nearest eddy changes for |
| --- | --- | --- |
| 0–20°N | +1.1% | 1.9% of cells |
| 20–40°N | +7.8% | 7.7% of cells |
| 40–60°N | +29.2% | 20.6% of cells |
| 60–70°N | +73.1% | 34.6% of cells |
| whole bbox | +21.3% (max 1292 km) | 13.6% of cells |

Front distances are shorter, so their absolute error is smaller, but the relative
error follows the same law (synthetic fronts at 0.05°, ~2% pixel density):

| Region | old mean | new mean | overstated by |
| --- | --- | --- | --- |
| tropics (10°N) | 18.1 km | 17.9 km | +0.9% |
| Azores (38°N) | 17.3 km | 15.2 km | +14.0% |
| N Atlantic (55°N) | 21.8 km | 16.3 km | +36.7% |

## 3. The fix (this branch)

- `to_unit_sphere(lats, lons)` is now the single projection in
  `utils/spatial.py`. Chord distance on the unit sphere rises with the angle
  between two points, so the nearest neighbour by chord *is* the nearest by
  great-circle.
- `haversine_min_distance_kdtree` indexes on it and converts the chord back to
  an arc, `d = 2R·asin(c/2)`. Both the neighbour chosen and the distance
  reported are now exact.
- `aviso.find_nearest_vectorized` calls the same `to_unit_sphere` instead of its
  own private copy, so the two searches cannot drift apart again.

`fronts.py` needs no change — it calls the fixed function.

Tests (`tests/test_spatial.py`, all five fail on the old code):
a degree of longitude at 0/40/60/70°N; agreement with a reference haversine over
200 random pairs; and the two searches picking the same target where the old
metric picked different ones.

## 4. A second, separate problem: the store predates the current code

While verifying, the 1998 eddies store turned out not to be reproducible from
today's code and the archived `META3.2_..._Anticyclonic_long` file:

- All 423 stored track IDs for 1998-01-01 are present in that exact day's
  observations, so the source file and date are right.
- **Every one of the 448,313 `ac_dist_km` values is a whole number of km.** The
  array is plain float32 with no scale factor, and no code path here rounds.
- Recomputing that day with the current code gives 182 km RMSE against the
  stored values and only 84% track agreement — more than the metric alone
  explains.

So the store was written by an older version of this code, and what else that
version did differently is unknown. This is not caused by the regrid work and
not by the metric alone. It is a second reason to regenerate rather than trust
the existing arrays.

## 5. Repair plan

Ordered, because each step invalidates the next one's input.

1. **Merge the metric fix.** Code only; nothing on disk changes.
2. **Regenerate the eddies store** from the archived raw trajectories
   (1993→present). No download needed: `rep/` and `nrt/` are on disk. Note the
   `archive_raw: true` hazard — `convert --in-dir` pointed at the store tree
   makes `download_root == store_root`; see `project_raw_file_cleanup_hazard`
   and the 2026-07-31 incident in `project_aviso_provenance_and_grid`.
   Verify afterwards by recomputing one day and expecting an exact match, which
   is the check that fails today.
3. **Recompute `sst_fdist` and `chl_fdist`.** These live in the sst and chl
   native stores, so this is a reconversion of those layers across the whole
   archive — the expensive step. Worth confirming first whether the fdist layers
   can be rewritten in place rather than reconverting sst/chl wholesale.
4. **Recompile h2ds**, which the regrid work needs anyway
   (`plans/regrid.md` phase A). Doing 2–3 first means one recompile covers both.
5. **Rebuild Parquet** from the recompiled h2ds.

## 6. Relationship to the regrid work

Independent. The regrid plan changes how a native store is put on the base grid;
this changes what the native store contains. They meet only at step 4, where one
recompile picks up both. The eddy plots that prompted this (Azores, 33–43°N) show
`*_dist_km` biased ~+26% in both the old and new compile, because the bias is in
the native store either way.

## 7. Open

- **What produced the integer-valued distances?** Unresolved (§4). Regenerating
  makes it moot for the data, but if an older code path is still reachable it
  should be found.
- **Is `*_normdist` right otherwise?** It divides by `effective_radius * 0.001`,
  i.e. metres→km; worth checking the raw units while the file is open.
- **Should the fronts distance be capped?** `BOA_application` returns front
  pixels for one day; where a day has few, the nearest front can be thousands of
  km away and the column is then meaningless rather than merely imprecise. Out
  of scope here, noted while reading the code.
