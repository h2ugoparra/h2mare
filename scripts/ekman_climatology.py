"""
Compute and save Ekman day-of-year (DOY) mean and monthly 90th percentile
climatology from the 1998-2017 baseline period.

Outputs (saved to $STORE_ROOT/Climatology/):
    - cds_ekman-doy-mean_80W-10E-0-70N_1998-2017.nc
    - cds_ekman-montly-90thquantile_80W-10E-0-70N_1998-2017.nc

Run this script once before ekman_derived_vars.py.
"""

import warnings

import xarray as xr
from loguru import logger

from h2mare.config import settings

warnings.simplefilter("ignore", UserWarning)

var_key = "atm-instante"
var_cfg = settings.app_config.variables[var_key]

assert settings.STORE_ROOT is not None, "STORE_ROOT must be set in the environment"
var_dir = settings.STORE_ROOT / var_cfg.local_folder

files = sorted(var_dir.glob("*.zarr"))
ekman = xr.open_mfdataset(files, engine="zarr")["ekman_pumping"]

# Baseline period
baseline = ekman.sel(time=slice("1998-01-01", "2017-12-31"))

# 7-day rolling mean
ekman_7d = baseline.rolling(time=7, min_periods=1).mean()

# Remove leap days (February 29)
ek_noleap = ekman_7d.where(
    ~((ekman_7d.time.dt.month == 2) & (ekman_7d.time.dt.day == 29)), drop=True
)

# DOY mean climatology
clim = ek_noleap.groupby("time.dayofyear").mean("time")

clim_dir = settings.STORE_ROOT / "Climatology"
clim_dir.mkdir(parents=True, exist_ok=True)

file_path = clim_dir / "cds_ekman-doy-mean_80W-10E-0N-70N_1998-2017.nc"
logger.info("Saving DOY mean climatology")
clim.to_netcdf(file_path)
logger.success(f"Saved: {file_path}")

# Monthly 90th percentile of anomalies
anom = ek_noleap.groupby("time.dayofyear") - clim
ds = xr.Dataset({"ekman_pumping_anom": anom.reindex_like(ekman)})

logger.info("Computing monthly 90th percentile")
p90_monthly = (
    ds["ekman_pumping_anom"].groupby("time.month").quantile(0.90, dim="time").compute()
)

file_path = clim_dir / "cds_ekman-monthly-90thquantile_80W-10E-0N-70N_1998-2017.nc"
logger.info("Saving monthly 90th percentile")
p90_monthly.to_netcdf(file_path)
logger.success(f"Saved: {file_path}")
