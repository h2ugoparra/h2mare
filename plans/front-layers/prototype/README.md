# Front-layer prototype: the scripts behind `plans/front-layers.md`

Research code, not part of the package: not linted, not tested, not imported by
h2mare. Kept so every number in the plan can be reproduced. All scripts are
read-only against the stores and need `H2MARE_ROOT` pointing at the deployed
project (the one whose `config.yaml` names the live sst/chl stores):

```bash
H2MARE_ROOT=/path/to/h2mare-run uv run python plans/front-layers/prototype/<script>.py [sst|chl]
```

Run on 2026-09-29/30 against `dev` (h2mare 0.8.1 + #238–#253), sst store
`CMEMS_SST_010_011` (OSTIA L4, 0.05°) and chl store `CMEMS_BGC_009_104` (multi-sensor
gap-free L4, 1/24°), bbox 80°W–10°E, 0–70°N.

| Script | Produces | Plan section |
|---|---|---|
| `01_boa_threshold_sweep.py` | Front coverage and distances vs BOA threshold (4 days, 2024) | §1.3. **Ran before #239**, so its BOA still filled gaps with 0: coverage includes the coastline artefact that PR removed |
| `02_gradient_distribution.py` | Gradient percentiles in physical units (sst, chl, log10 chl) | §1.3, §4.2 |
| `03_effective_resolution.py` | Meridional/zonal spectra, half-power wavelengths, `spectra.npy` | §2 |
| `04_effective_resolution_years.py` | The same for 2004 / 2014 / 2024, plus the 12–25 km noise ratio | §2 |
| `detect.py` | The detector (current = **v2**: σ in km on both axes; `read_values` retry) | §3 |
| `05_calibrate_v1.py` | v1 thresholds: percentiles over 24 days of 2024 | §4.2 |
| `06_run_year.py` | Full-year 2024 layers + daily fragmentation/stability stats | §5 |
| `07_analyze.py` | Summaries, Spearman correlations, VIF, maps | §5 |
| `08_calibrate_multi_year.py` | v2 thresholds: percentiles per year × σ; `calibration_multi.json` | §4.2, §4.3 |
| `09_sensitivity.py` | One-at-a-time sensitivity; `sensitivity_{sst,chl}.csv` | §4.5 |
| `10_persistent_distance.py` | Persistence cut for the distance to persistent fronts; `persistent_distance_{sst,chl}.json` | §4.4 |

`report.json` is the output of `07_analyze.py`.

## v1 vs v2

`06_run_year.py` and `07_analyze.py` (the full-year run, §5) were run with **v1**
settings: the Gaussian σ was converted to cells using the *latitude* spacing on
both axes (so narrower than intended east–west at high latitude), and thresholds
came from 2024 alone (sst 0.0157 / 0.0312 °C/km, chl 0.0037 / 0.0076 per km).
`detect.py` was then corrected to **v2** (σ in km on both axes), and the
multi-year calibration and the sensitivity analysis (§4) use v2. Re-running
`06`/`07` with the current `detect.py` gives v2 layers; the sensitivity results
say the difference is small for sst and moderate for chl.

The spectra figure was plotted inline from `spectra.npy` (log–log spectra and the
ratio to the 150–600 km power-law fit per box); the maps are from `07_analyze.py`.
