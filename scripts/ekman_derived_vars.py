"""
Apply Ekman climatology to derive variables and merge into yearly Zarr files.

Requires climatology files produced by ekman_climatology.py:
    - cds_ekman-doy-mean_80W-10E-0-70N_1998-2017.nc
    - cds_ekman-montly-90thquantile_80W-10E-0-70N_1998-2017.nc

Derived variables written to each yearly Zarr file:
    - ekman_7d, ekman_anom, ekman_anom_lag{3,7,14}, n_upwell_events_{3,7,14}d
"""

import warnings
from pathlib import Path

import xarray as xr
from loguru import logger

from h2mare.config import get_settings
from h2mare.utils.spatial import clip_land_data

warnings.simplefilter("ignore", UserWarning)


def get_path_for_year(files: list[Path], year: int) -> Path | None:
    for f in files:
        if f.stem.endswith(str(year)):
            return f
    return None


var_key = "atm-instante"
var_cfg = get_settings().app_config.variables[var_key]

assert get_settings().STORE_ROOT is not None, (
    "STORE_ROOT must be set in the environment"
)
var_dir = get_settings().STORE_ROOT / var_cfg.local_folder
clim_dir = get_settings().STORE_ROOT / "Climatology"

files = sorted(var_dir.glob("*.zarr"))
ekman = xr.open_mfdataset(files, engine="zarr")["ekman_pumping"]

p90 = xr.open_dataset(
    clim_dir / "cds_ekman-montly-90thquantile_80W-10E-0-70N_1998-2017.nc"
)["ekman_pumping_anom"]
clim = xr.open_dataset(clim_dir / "cds_ekman-doy-mean_80W-10E-0-70N_1998-2017.nc")

logger.info("Computing Ekman derived variables")

ekman_7d = ekman.rolling(time=7, min_periods=1).mean()
anom = ekman_7d.groupby("time.dayofyear") - clim

ds_ekman = xr.Dataset(
    {
        "ekman_7d": ekman_7d,
        "ekman_anom": anom["ekman_pumping"],
    }
)

ds_ekman["ekman_7d"].attrs.update(
    {
        "long_name": "Ekman 7day-mean vertical velocity",
        "units": "m/s",
        "description": "Ekman pumping mean velocity within a rolling 7-day window.",
    }
)
ds_ekman["ekman_anom"].attrs.update(
    {
        "long_name": "Ekman anomaly",
        "units": "m/s",
        "description": (
            "Ekman anomaly calculated as the difference between 7day mean Ekman pumping "
            "and 1998-2017 climatology, per day-of-year (DOY) and grid cell."
        ),
    }
)

# Lag anomalies
for lag in [3, 7, 14]:
    ds_ekman[f"ekman_anom_lag{lag}"] = ds_ekman["ekman_anom"].shift(time=lag)
    ds_ekman[f"ekman_anom_lag{lag}"].attrs.update(
        {
            "long_name": f"Ekman anomaly with a {lag} day lag",
            "units": "m/s",
            "description": (
                f"{lag}-days lagged Ekman anomaly calculated as the difference between "
                f"7day mean Ekman pumping and 1998-2017 climatology, per day-of-year (DOY) and grid cell."
            ),
        }
    )

# Event exceedance detection
p90_broadcast = xr.apply_ufunc(
    lambda m: p90.sel(month=m),
    ds_ekman["time"].dt.month,
    vectorize=True,
    dask="parallelized",
    input_core_dims=[[]],
    output_core_dims=[["lat", "lon"]],
    output_dtypes=[ds_ekman["ekman_anom"].dtype],
)

exceed = ds_ekman["ekman_anom"] > p90_broadcast

for w in [3, 7, 14]:
    ds_ekman[f"n_upwell_events_{w}d"] = clip_land_data(
        exceed.rolling(time=w, min_periods=1).sum()
    )
    ds_ekman[f"n_upwell_events_{w}d"].attrs.update(
        {
            "long_name": f"Number of Ekman pumping upwelling events within {w}-days",
            "units": "count",
            "description": (
                f"Daily count of events where Ekman pumping anomaly exceeded the 90th percentile "
                f"threshold from the 1998 to 2017 monthly climatology computed for each grid cell "
                f"and accumulated within a rolling {w}-day window. "
                f"Values range from 0 (no events) to {w} (all days in the window exceed threshold). "
                f"Note: values represent event frequency, not number of days."
            ),
        }
    )

# Merge into yearly Zarr files
for year in range(1998, 2018):
    logger.info(f"Processing year: {year}")

    year_path = get_path_for_year(files, year)
    if year_path is None:
        logger.warning(f"No Zarr file found for year {year}, skipping")
        continue

    ds2 = ds_ekman.where(ds_ekman.time.dt.year == year, drop=True).drop_vars(
        "dayofyear"
    )
    ds2 = ds2.chunk({"time": 30, "lat": len(ds2.lat), "lon": len(ds2.lon)})

    ds1 = xr.open_zarr(year_path)
    xr.testing.assert_equal(ds1.time, ds2.time)

    ds_merged = xr.merge([ds1, ds2])
    ds_merged.to_zarr(year_path, mode="w")
    logger.success(f"Merged data saved at {year_path}")
