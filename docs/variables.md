# Variable Catalog

All variables configured in `config.yaml`. Each key maps to one data stream with its own downloader, Zarr store, and output variables.

---

## Overview

| Key | Source | Provider | NRT | Output variables |
|---|---|---|---|---|
| `sst` | CMEMS | Satellite L4 | yes | `analysed_sst`, `analysis_error`, `sst_fdist` |
| `ssh` | CMEMS | Satellite L4 | yes | `adt`, `sla`, `ugos`, `vgos`, `gke` |
| `mld` | CMEMS | Model (NEMO) | no | `mld` |
| `chl` | CMEMS | Satellite L4 | no | `chl`, `chl_fdist` |
| `seapodym` | CMEMS | Model (SEAPODYM) | no | `mnkc_epi`, `mnkc_umeso`, `mnkc_mumeso`, `npp`, `zooc`, `zeu` |
| `o2` | CMEMS | Model (PISCES) | no | `o2_0`, `o2_100`, `o2_500`, `o2_1000` |
| `fsle` | AVISO | FTP | yes | `fsle_max` |
| `eddies` | AVISO | FTP | yes | eddy geometry and track fields |
| `atm-instante` | CDS / ERA5 | Reanalysis | no | `msl`, `u10`, `v10`, `tcc`, `wind_mean`, `wind_max`, `wind_std` |
| `atm-accum-avg` | CDS / ERA5 | Reanalysis | no | `tp`, `avg_iews`, `avg_inss`, `ekman_pumping`, `ekman_anom`, … |
| `radiation` | CDS / ERA5 | Reanalysis | no | `slhf`, `ssrd`, `tisr` |
| `waves` | CDS / ERA5 | Reanalysis | no | `swh`, `mdts` |

---

## Detailed descriptions

### `sst` — Sea Surface Temperature
- **Dataset (rep):** `METOFFICE-GLO-SST-L4-REP-OBS-SST`
- **Dataset (nrt):** `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2`
- **Resolution:** 0.05°, area-averaged onto the 0.25° grid
- **Variables:** analysed SST (°C), analysis error (K), distance to nearest SST front (km)

### `ssh` — Sea Surface Height
- **Dataset (rep):** `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`
- **Dataset (nrt):** `cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D`
- **Resolution:** 0.125°, area-averaged onto the 0.25° grid
- **Variables:** ADT (m), SLA (m), geostrophic velocities u/v (m s⁻¹), geostrophic kinetic energy (m² s⁻²)

### `mld` — Mixed Layer Depth
- **Dataset (rep):** `cmems_mod_glo_phy_my_0.083deg_P1D-m`
- **Resolution:** 0.083°, area-averaged onto the 0.25° grid
- **Variables:** mixed layer depth (m) — depth where density increase corresponds to a 0.2 °C temperature decrease relative to 10 m

### `chl` — Chlorophyll-a
- **Dataset (rep):** `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D`
- **Resolution:** 4 km, area-averaged onto the 0.25° grid
- **Variables:** CHL concentration (mg m⁻³), distance to nearest CHL front (km)

### `seapodym` — Micronekton (SEAPODYM)
- **Dataset (rep):** `cmems_mod_glo_bgc_my_0.083deg-lmtl_P1D-i`
- **Variables:** epipelagic, upper/migrant mesopelagic micronekton (g m⁻²); net primary productivity (mg m⁻² day⁻¹); zooplankton carbon (g m⁻²); euphotic depth (m)

### `o2` — Dissolved Oxygen
- **Dataset (rep):** `cmems_mod_glo_bgc_my_0.25deg_P1D-m`
- **Resolution:** 0.25° (native)
- **Variables:** O₂ concentration (mmol m⁻³) at 0, 100, 500, and 1000 m depth

### `fsle` — Finite-Size Lyapunov Exponents
- **Source:** AVISO FTP (delayed-time and NRT)
- **Variables:** backward FSLE maximum eigenvalue (days⁻¹)

### `eddies` — Mesoscale Eddies
- **Source:** AVISO META3.2 eddy trajectory atlas
- **Variables:** eddy center coordinates, track ID, effective and speed radii, amplitude, mean speed; distance and normalised distance to nearest cyclonic/anticyclonic eddy center

### `atm-instante` — Instantaneous Atmospheric Variables
- **Source:** ERA5 hourly single-level reanalysis (CDS)
- **Variables:** mean sea level pressure (hPa), 10 m wind components u/v (m s⁻¹), total cloud cover, wind speed statistics (mean, max, std)

### `atm-accum-avg` — Accumulated / Averaged Atmospheric Variables
- **Source:** ERA5 hourly single-level reanalysis (CDS)
- **Variables:** total precipitation (mm), surface wind stress components (N m⁻²), daily Ekman pumping velocity and anomaly (m s⁻¹), lagged Ekman anomalies (3, 7, 14 days), upwelling event counts (3, 7, 14-day windows)

### `radiation` — Surface Radiation
- **Source:** ERA5 hourly single-level reanalysis (CDS)
- **Variables:** surface latent heat flux (W m⁻²), surface solar radiation downwards (W m⁻²), TOA incident solar radiation (W m⁻²)

### `waves` — Ocean Waves
- **Source:** ERA5 hourly single-level reanalysis (CDS)
- **Variables:** significant height of combined wind waves and swell (m), mean direction of total swell (degrees)

---

## Special variables (compile-only)

These keys are not downloaded — they are generated during the compile step.

| Key | Description |
|---|---|
| `bathy` | Ocean bathymetry from ETOPO 2022 v1 at 0.25°. Mean depth (m) and std per grid cell. Files are named per layer under `layers` in `config.yaml` and built by `scripts/bathymetry.py`: `15s` and `60s` (native grid, `bathy_std` a 3×3 rolling std) and `0.25deg` (std of the 15s cells in each cell). Compile reads `compile_layer`; extraction reads `extract_layer` or `Extractor(bathy_layer=...)`, for CSV and SHP alike. |
| `moon` | Lunar illumination (%) computed from the `ephem` library. Same value broadcast across all grid cells per day. |
