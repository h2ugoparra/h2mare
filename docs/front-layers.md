# Front Layers

Layers that describe ocean fronts for species distribution models (SDMs),
detected daily from `sst` and `chl` when they are converted. They replace the
Belkin–O'Reilly front distances (`sst_fdist`, `chl_fdist`), which were retired.

This page is for using the layers. The design record behind them, with every
measurement, the alternatives considered and the decisions revised along the
way, is
[`plans/front-layers.md`](https://github.com/h2ugoparra/h2mare/blob/main/plans/front-layers.md).

---

## The layers

For each field (`sst`, `chl`):

| Variable | Meaning | Units | Native store | h2ds / Parquet |
|---|---|---|---|---|
| `{v}_front` | Daily front mask: 1 front, 0 no front, NaN not assessed | 1 | yes | no |
| `{v}_grad` | Gradient magnitude after smoothing: how sharp the field changes, per km | sst K km⁻¹; chl km⁻¹ (of log10 mg m⁻³) | yes | yes |
| `{v}_ffreq30` | Share of the last 30 assessed days with a front within 12.5 km | 0–1 | yes | yes |
| `{v}_pdist30` | Distance to the nearest persistent front: a pixel with `ffreq30` ≥ 0.5. 0 inside such a zone | km | yes | yes |

The sst gradient is in K km⁻¹ because CF tools read a bare °C as an absolute
temperature. A temperature difference is the same in both: 0.03 K km⁻¹ is
0.03 °C km⁻¹.

### Which to use

- **`grad`** is local front strength. It is continuous, needs no threshold, and
  is the least collinear of the candidates measured (variance inflation 2.4–2.9).
- **`ffreq30`** and **`pdist30`** measure the same thing, regional and persistent
  front activity (Spearman −0.84 to −0.92). Use one of them in a model, not
  both.
  - `ffreq30` is bounded and robust, but it is 0 anywhere no front came within
    12.5 km: about 41–50% of native pixels, mostly open ocean.
  - `pdist30` separates those front-free areas, telling a pixel 50 km from a
    frontal zone from one 1,000 km away. Its values depend on the 0.5 cut (see
    [Thresholds](#thresholds)).
- **sst and chl fronts are complementary** (gradient ρ 0.39, frequency ρ 0.49),
  so a model can use both fields' layers.
- **The mask** is kept in the native store because the frequency is computed
  from it. It is not compiled.

---

## How fronts are detected

The detector is the Canny (1986) edge detector, adapted to gridded L4 fields.
Per day:

1. **Transform:** chl → log10, because its gradients otherwise scale with
   concentration and vanish in oligotrophic water. sst is used as is.
2. **Gap fill:** missing cells (land, ice) are filled from the nearest valid
   cell, so a coastline is not read as a front. They are masked again at the end.
3. **Smooth:** a Gaussian of width σ in km on both axes. The east–west width is
   set per row, because a longitude cell narrows with latitude.
4. **Gradient:** per km, with the cos(latitude) metric, so its magnitude means
   the same at every latitude, orientation and grid resolution.
5. **Thin:** keep only the ridge of each gradient band (non-maximum
   suppression).
6. **Hysteresis:** a ridge becomes a front if it reaches the high threshold
   somewhere, and is traced along while it stays above the low one. A weak ridge
   that never reaches the high threshold is dropped, which removes speckle.
7. **Confidence (sst):** pixels whose `analysis_error` exceeds 1.52 K are *not
   assessed*.

The frequency and distance are then computed from the daily masks.

---

## Missing values

**Not assessed is not "no front".** A pixel is not assessed on a day when it has
no data (land, ice) or, for sst, when the analysis error is above the cut. Such
a day is left out of the frequency's denominator rather than counted as 0.

| Variable | NaN where |
|---|---|
| `{v}_front` | the pixel was not assessed that day |
| `{v}_grad` | the pixel has no data that day. The sst confidence cut does not apply: it decides where fronts can be called, not whether the gradient exists |
| `{v}_ffreq30` | fewer than half the window's 30 days were assessed |
| `{v}_pdist30` | `ffreq30` is NaN; or, on a day when no pixel reaches the cut, everywhere |

The record starts in January 1998, so the frequency (and the distance) is NaN
for the first 14 days: it needs 15 assessed days.

---

## Thresholds

Every value is physical and fixed for every year, so layers compare across
years. None is recomputed from the data when it runs.

| Threshold | sst | chl | Why |
|---|---|---|---|
| Smoothing σ | 5 km | 7 km | Set by what the product resolves (effective resolution ~50–80 km sst, ~60–110 km chl). sst has no small-scale noise, so ~1 cell is enough to steady the gradient direction. chl has a noise floor below ~30 km, and 7 km removes it while keeping half the power at 60 km |
| High threshold | 0.0299 °C km⁻¹ | 0.0070 km⁻¹ | The 90th percentile of the smoothed gradient over the domain, mean of 2004, 2014 and 2024 (stable within ±3%). A front is where gradients are among the strongest 10% |
| Low threshold | 0.0155 °C km⁻¹ | 0.0035 km⁻¹ | The 75th percentile; 2:1 with the high, as usual for Canny |
| Confidence cut | `analysis_error` > 1.52 K | — | The 2024 99th percentile. A lower cut (0.84 K) masked much of the Gulf Stream, because the error grows with variability; 1.52 K masks mostly ice margins and high-latitude cloud. chl has no uncertainty field |
| Frequency window | 30 days | 30 days | A monthly scale; a 7-day frequency correlates 0.82–0.83 with it |
| Frequency tolerance | 12.5 km | 12.5 km | Front positions are uncertain to about a quarter of the ~50 km effective resolution, so an exact pixel hit is mostly noise |
| Persistent front | `ffreq30` ≥ 0.5 | `ffreq30` ≥ 0.5 | A front nearby more often than not. Above chance recurrence at the domain's own front rate, and high enough that the distance does not turn into a mask |

The thresholds are percentiles of *this* domain's gradients (80°W–10°E, 0–70°N).
They are not literature values: those were set on L2/L3 imagery, where the same
front is sharper than in a smoothed L4 analysis. A different domain or product
needs them recalibrated.

How each value was measured, its sensitivity and what would justify changing
it: the plan's threshold register (§9.2).

---

## In h2ds

Compile puts `{v}_grad`, `{v}_ffreq30` and `{v}_pdist30` on the 0.25° grid by
area-weighted mean, over 5×5 native cells for sst and 6×6 for chl (see
[Regridding](api/compiler.md#regridding)). The mask is dropped. What the
compiled values mean:

- **`grad`:** the mean front strength in the cell. This is not the gradient of
  the 0.25° field, which averaging flattens.
- **`ffreq30`:** the mean of the cell's pixel frequencies. Its values are lower
  than "the share of days with a front anywhere in the cell", but it ranks
  cells the same way (ρ 0.94–0.98 already at native resolution).
- **`pdist30`:** the mean distance over the cell's pixels.

---

## Configuration and operations

Declared per var_key under `front_layers` (full reference:
[Configuration](configuration.md#variable-entries)):

```yaml
sst:
  compiled_vars: [sst, analysis_error, sst_std, sst_grad, sst_ffreq30, sst_pdist30]
  front_layers:
    sst:
      source: sst
      sigma_km: 5.0
      low_per_km: 0.0155
      high_per_km: 0.0299
      confidence: {var: analysis_error, max: 1.52}
      frequency_days: [30]
      frequency_radius_km: 12.5
      persistent_distance: {window: 30, min_frequency: 0.5}
```

- **Convert** detects the layers for the days it writes. The first 29 days'
  frequencies are seeded from the masks already in the store. Rewriting past
  days (REP replacing NRT) recomputes the frequency and distance of the stored
  days that follow, so the store matches a continuous run.
- **Backfill** adds the layers to a store converted before they were declared,
  from the store's own field, year by year:

    ```bash
    uv run python scripts/backfill_front_layers.py sst chl            # report only
    uv run python scripts/backfill_front_layers.py sst chl --apply
    ```

    Then compile and Parquet over the same period.
- **Changing a threshold** means recomputing the layers over the whole record:
  run the backfill again, then compile and Parquet.

---

## Describing the method

For a paper using the layers:

> Fronts were detected daily with an implementation of the Canny (1986) edge
> detector adapted to gridded L4 fields: Gaussian smoothing with σ = 5 km (SST)
> or 7 km (log10 chl), gradients computed per km with a cos(latitude) metric,
> non-maximum suppression, and hysteresis thresholding with fixed thresholds
> (0.0155 / 0.0299 °C km⁻¹ for SST; 0.0035 / 0.0070 km⁻¹ for log10 chl) set at
> the 75th and 90th percentiles of the smoothed gradient in the study domain.
> SST pixels with an analysis error above 1.52 K were treated as unobserved.

Add the derived layers you use: the 30-day frequency counts a front within
12.5 km, and persistent fronts are pixels with a front nearby on at least half
the assessed days.

Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions
on Pattern Analysis and Machine Intelligence*, 8(6), 679–698.
