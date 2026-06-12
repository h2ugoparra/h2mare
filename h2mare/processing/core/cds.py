"""
Process downloaded CDS-ERA5 hourly grib data to daily means.

"""

# Warnings raised:
# RunTimeWarning: data has 0 and nan, a warning is emitted by NumPy (via np.divide) while Dask is evaluating a task
# UserWarning: Zarr possible incmpatilibility outside Python ecosystem
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare import get_settings
from h2mare.models import KeyVarConfigEntry
from h2mare.storage.xarray_helpers import rename_dims, unified_time_chunk
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.utils.spatial import clip_land_data

warnings.filterwarnings("ignore")

_EKMAN_P90_FILE = "cds_ekman-monthly-90thquantile_80W-10E-0N-70N_1998-2017.nc"
_EKMAN_DOY_FILE = "cds_ekman-doy-mean_80W-10E-0N-70N_1998-2017.nc"


# ----------------------------
#   ---- Helpers ----
# ----------------------------
def _get_ds_for_month(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove datetimes before and after the target month.
    This was implemented for radiation and Atm-Accum-avg because first and last day of the month has lower values compared to adjacent days.

    Args:
        ds (xr.Dataset): _description_

    Returns:
        tuple: first and last date for sel
    """
    arr = np.asarray(ds.time.values, dtype="datetime64[ns]")
    months = arr.astype("datetime64[M]")
    unique_months, counts = np.unique(months, return_counts=True)
    true_month = unique_months[np.argmax(counts)]
    mask = months == true_month
    dt_ini, dt_fin = arr[mask].min(), arr[mask].max()
    return ds.sel(time=slice(dt_ini, dt_fin))


def merge_time_step(
    ds: xr.Dataset, time_dim: str = "time", step_dim: str = "step"
) -> xr.Dataset:
    """
    Create a single datetime coordinate from base time and step dimensions.
    Implemented for accumulated/avg variables (atm-accum-avg and radiation).
    """
    time = ds[time_dim]
    step = ds[step_dim]

    # Compute full datetime values directly (time + step)
    valid_time = xr.DataArray(
        (time.values[:, None] + step.values[None, :]).ravel(),  # flatten to 1D
        dims=("valid_time",),
        name="valid_time",
    )

    # Reindex dataset to use valid_time as the only dimension
    ds = (
        ds.stack(point=("time", "step"))  # intermediate
        .assign_coords(valid_time=("point", valid_time.data))  # attach datetime64[ns]
        .swap_dims({"point": "valid_time"})  # promote as dimension
        .drop_vars(["time", "step", "number", "surface", "point"])
        .sortby("valid_time")
        .rename({"valid_time": "time"})
    )
    return ds


def drop_dims(
    ds: xr.Dataset,
    dims_to_drop: list[str] = ["step", "number", "surface", "meanSea"],
) -> xr.Dataset:
    """
    Drop coordinates/dimensions from dataset"

    Args:
        ds (xr.Dataset): dataset to drop variables
        dims_to_drop (list[str]): List of vars to drop. Default to ['step', 'number', 'surface', 'number', 'meanSea']

    Returns:
        xr.Dataset: ds without dims_to_drop
    """
    return ds.drop_vars(dims_to_drop, errors="ignore")


def resample_daily_mean(ds, time_dim="time"):
    return ds.resample({time_dim: "1D"}).mean()


# ------------------------------------------------------
# ---- ATM-INSTANTE : pressure-wind-clouds features ----
# -------------------------------------------------------
def daily_wind(
    ds: xr.Dataset, u: str = "u10", v: str = "v10", time_dim: str = "time"
) -> xr.Dataset:
    """
    Compute daily wind features from hourly u (eastward) and v (northward) wind components.

    Args:
        ds (xr.Dataset): Input datatset containing u and v wind components.
        u (str, optional): Name of the eastward wind component variable. Defaults to "u10".
        v (str, optional): Name of the northward wind component variable.Defaults to "v10".
        time_dim (str, optional): Name of the time dimension. Defaults to "time".

    Returns:
        xr.Dataset: Dataset containing daily mean, std, max and mean wind speed.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Dataset does not have dimension '{time_dim}'")

    u10 = ds[u]
    v10 = ds[v]
    ws = (u10**2 + v10**2) ** 0.5

    r = ws.resample({time_dim: "1D"})
    out = xr.Dataset(
        {
            "wind_mean": r.mean(),
            "wind_std": r.std(),
            "wind_max": r.max(),
            u: u10.resample({time_dim: "1D"}).mean(),
            v: v10.resample({time_dim: "1D"}).mean(),
        }
    )
    # attrs
    for k, ln in {
        "wind_mean": "Daily mean 10m wind speed",
        "wind_std": "Daily std 10m wind speed",
        "wind_max": "Daily max 10m wind speed",
        u: "Daily mean eastward 10m wind",
        v: "Daily mean northward 10m wind",
    }.items():
        out[k].attrs.update(
            {
                "short_name": k,
                "long_name": ln,
                "units": "m/s",
            }
        )

    # out = float64_to_float32(out)
    return drop_dims(out)


def daily_cloud_cover(
    ds: xr.Dataset, var_name: str = "tcc", time_dim: str = "time"
) -> xr.Dataset:
    """Compute daily cloud cover from hourly cloud cover data

    Args:
        ds (xr.Dataset): Input datatset containing total cover cloud data.
        var_name (str, optional): Total cloud cover variable name. Defaults to "tcc".
        time_dim (str, optional): Name of the time dimension. Defaults to "time".

    Returns:
        xr.Dataset: Dataset containing daily mean total cloud cover.
    """
    da = ds[var_name]
    out = xr.Dataset({var_name: da.resample({time_dim: "1D"}).mean()})
    out[var_name].attrs.update(
        {
            "GRIB_name": "Daily mean total cloud cover",
            "long_name": "Daily mean total cloud cover",
        }
    )
    # out = float64_to_float32(out)
    return drop_dims(out)


def daily_sea_level_pressure(
    ds: xr.Dataset,
    var_name: str = "msl",
) -> xr.Dataset:
    """Compute daily mean sea level pressure from hourly cloud cover data

    Args:
        ds (xr.Dataset): Input datatset containing mean sea level pressure data.
        var_name (str, optional): Mean sea level pressure variable name. Defaults to "msl".
        time_dim (str, optional): Name of the time dimension. Defaults to "time".

    Returns:
        xr.Dataset: Dataset containing daily mean sea level pressure.
    """
    da = ds[var_name] * 0.01  # Pa to hPA
    out = xr.Dataset({var_name: da})
    out = resample_daily_mean(out)

    out[var_name].attrs.update(
        {"long_name": "Daily mean sea level pressure", "units": "hPa"}
    )
    return drop_dims(out)


# -----------------------------
# ---- Radiation features ----
# ----------------------------
def hourly_radiation(
    da: xr.DataArray,
    time_dim: str = "time",
    units_out: str = "W/m²",
    clip_small_negatives: bool = True,
) -> xr.DataArray:
    """Convert accumulated quantity (J/m2) to per-second rate over each interval.

    Args:
        da (xr.DataArray): data array with accumulated radiation data.
        time_dim (str, optional): time dimension name. Defaults to "time".
        units_out (str, optional): Output units. Defaults to "W m^-2".
        clip_small_negatives (bool, optional): Clip extremely negative values. Defaults to True.

    Returns:
        xr.DataArray: Hourly mean rates of the accumulated data.
    """
    dt = da[time_dim].diff(time_dim) / np.timedelta64(1, "s")
    dacc = da.diff(time_dim)
    rate = dacc / dt

    if clip_small_negatives:
        rate = rate.where(rate >= -1e-6, 0.0)

    rate.attrs.update(da.attrs)
    rate.attrs.update(
        {
            "units": units_out,
            "GRIB_units": units_out,
            "long_name": f"Hourly mean rate from accumulated {da.name or ''}".strip(),
        }
    )
    rate.name = da.name
    # dims are swapped compared to the rest
    rate = rate.transpose("time", "lat", "lon")
    return rate


def daily_radiation(da: xr.DataArray, time_dim: str = "time") -> xr.Dataset:
    """
    Convert hourly to daily averages of radiation flux data.

    Args:
        da (xr.DataArray): Hourly radiation data array.

    Returns:
        xr.DataArray: Daily mean array.
    """
    da = hourly_radiation(da, time_dim=time_dim).astype("float32")
    out = xr.Dataset({da.name: da})
    return resample_daily_mean(out)


# ----------------------------------------------
# ---- atm-accum-avg (rain and wind stress) ----
# ----------------------------------------------
def daily_total_rain(
    ds: xr.Dataset, var_name: str = "tp", time_dim: str = "time"
) -> xr.Dataset:
    """
    Compute daily total precipitation from hourly data

    Args:
        ds (xr.Dataset): Input datatset containing total precipitation data.
        var_name (str, optional): Total precipitation variable name. Defaults to "tp".
        time_dim (str, optional): Name of the time dimension. Defaults to "time".

    Returns:
        xr.Dataset: Dataset containing daily total precipitation.
    """
    da = ds[var_name] * 1000  # Convert m to mm
    out = xr.Dataset({var_name: da.resample({time_dim: "1D"}).sum()})
    out[var_name].attrs.update(
        {
            "long_name": "Daily total precipitation",
            "units": "mm",
        }
    )
    out = out.transpose("time", "lat", "lon")
    return out


def compute_curl_and_ekman(
    ds: xr.Dataset,
    tx_name: str = "avg_iews",
    ty_name: str = "avg_inss",
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Compute Ekman vertical velocities from wind stress components.
    Assumes stresses are in N m^-2 and lat/lon in degrees.

    Args:
        ds: (xarray.Dataset) with time, lat, lon dims and wind stress components
        tx_name, ty_name (str): variable names of eastward (tx_name) and northward (ty_name) components

    Returns:
        ds with added variables: curl_tau (N m^-3), ekman_pumping (m s^-1) and ekman_pumping_7d (i.e. 7 days trailing mean)
    """
    # constants
    R = 6_371_000.0  # earth radius (m)
    Omega = 7.2921159e-5  # s^-1
    rho_w = 1025.0  # seawater density kg m^-3

    # Clip land cells
    ds = clip_land_data(ds)

    tx, ty = ds[tx_name], ds[ty_name]

    # convert lat/lon to radians arrays
    lat_rad = np.deg2rad(ds["lat"])
    lon_rad = np.deg2rad(ds["lon"])

    # compute mean grid spacing in radians (assumes regular grid)
    dlon = float(np.diff(lon_rad).mean())
    dlat = float(np.diff(lat_rad).mean())

    # approximate partial derivatives with xarray's differentiate or manual diff
    # Here we do a centered finite difference using roll to keep coordinates aligned.
    # dτy/dx (note: derivative w.r.t lon; need to divide by dx which depends on lat)
    dty_dlon = (ty.roll(lon=-1) - ty.roll(lon=1)) / (2.0 * dlon)  # per rad
    dtx_dlat = (tx.roll(lat=-1) - tx.roll(lat=1)) / (2.0 * dlat)  # per rad

    # convert per rad -> per meter: divide by R*cos(lat)
    dty_dx = dty_dlon / (R * np.cos(lat_rad))
    # dτx/dy (derivative w.r.t lat)
    dtx_dy = dtx_dlat / R

    # curl = dτy/dx - dτx/dy  (units N m^-3)
    curl_tau = dty_dx - dtx_dy
    # ds = ds.copy()
    # ds['curl_tau'] = curl_tau

    # Coriolis f (same shape as lat)
    f = 2 * Omega * np.sin(lat_rad)
    # align f with dataset dims -> make it 2D lat x lon if needed
    f_da = xr.DataArray(f, coords={"lat": ds["lat"]}, dims=["lat"])
    f_grid = f_da.broadcast_like(ds[tx_name])

    ## Ekman pumping: w_E = curl / (rho_w * f)
    # mask near equator to avoid blow-ups
    equator_mask = np.abs(f_grid["lat"]) < 2.0
    ekman = xr.where(~equator_mask, curl_tau / (rho_w * f_grid), np.nan)

    out = xr.Dataset({tx_name: tx, ty_name: ty, "ekman_pumping": ekman})

    out = resample_daily_mean(out)

    out["ekman_pumping"].attrs.update(
        {
            "long_name": "Daily Ekman vertical velocity",
            "units": "m/s",
            "description": "Daily Ekman pumping mean velocity derived from hourly data "
            "of surface wind stress curl/turbulence components. Pumping is defined as vertical velocity at the base of the Ekman layer. "
            "positive = upwelling (suction), negative = downwelling (pumping).",
        }
    )

    out = out.transpose("time", "lat", "lon")
    return out


def get_previous_dates_da(da: xr.DataArray, var_key: str):
    """Get previous 15-days for ekman lag calculation"""
    da_dt_ini = da.time.values[0]

    repo = ZarrCatalog(var_key)

    date_prev = pd.to_datetime(da_dt_ini) - pd.Timedelta(days=15)
    ds_prev = repo.open_dataset(
        start_date=date_prev, end_date=da_dt_ini - pd.Timedelta(days=1)
    )
    if ds_prev is not None:
        if isinstance(ds_prev, xr.Dataset):
            ds_prev = drop_dims(
                ds_prev, dims_to_drop=["quantile", "month", "dayofyear"]
            )
            da_prev = ds_prev["ekman_pumping"]
        else:
            da_prev = ds_prev
        return xr.concat([da, da_prev], dim="time").sortby("time")
    else:
        logger.warning("No previous data available. Returning input data array.")
        return da


def add_engineered_ekman(da: xr.DataArray, var_key: str):
    """Compute Ekman pumping related variables.
    Args:
        da: (xarray.DataArray) DataArray with variable 'ekman_pumping'
    """
    clim_dir = get_settings().CLIMATOLOGY_DIR
    if clim_dir is None:
        raise FileNotFoundError(
            "Directory for Ekman pumping Climatological data not found"
        )

    p90 = xr.open_dataset(clim_dir / _EKMAN_P90_FILE)["ekman_pumping_anom"]
    p90 = p90.chunk({"month": -1, "lat": 200, "lon": 200})

    clim_doy = xr.open_dataset(clim_dir / _EKMAN_DOY_FILE)
    clim_doy = clim_doy.chunk({"dayofyear": -1, "lat": 200, "lon": 200})

    # Get previous days for rowling mean
    da_dt_ini = da.time.values[0]
    da_dt_fin = da.time.values[-1]

    da = get_previous_dates_da(da, var_key)

    # Try to make it more efficient
    da = da.chunk({"time": 30, "lat": 200, "lon": 200})

    # Get 7-day rolling mean of Ekman pumping (Since files are yearly, the first 6days of the year are not complete)
    ekman_7d = da.rolling(time=7, min_periods=1).mean()
    clim_align = clim_doy.sel(dayofyear=ekman_7d["time"].dt.dayofyear)
    anom = ekman_7d - clim_align

    ds_ekman = xr.Dataset({"ekman_7d": ekman_7d, "ekman_anom": anom["ekman_pumping"]})

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
            "description": "Ekman anomaly calculated by the difference between 7day rolling mean Ekman pumping and 1998-2017 climatology, per day-of-year (DOY) and grid cell.",
        }
    )

    # Create lag anomalies
    for lag in [3, 7, 14]:
        ds_ekman[f"ekman_anom_lag{lag}"] = ds_ekman["ekman_anom"].shift(time=lag)
        ds_ekman[f"ekman_anom_lag{lag}"].attrs.update(
            {
                "long_name": f"Ekman anomaly with a {lag} day lag",
                "units": "m/s",
                "description": f"{lag}-days lagged Ekman anomaly calculated as the difference between 7day mean Ekman pumping and 1998-2017 climatology, per day-of-year (DOY) and grid cell.",
            }
        )

    # Align 90th percentile climatology with the time axis
    p90_aligned = p90.sel(month=ds_ekman["time"].dt.month)

    # Exceedances: anomaly > local monthly p90
    exceed = ds_ekman["ekman_anom"] > p90_aligned

    # Rolling counts for 3, 7, 14 days
    for w in [3, 7, 14]:
        ds_ekman[f"n_upwell_events_{w}d"] = exceed.rolling(time=w, min_periods=1).sum()
        ds_ekman[f"n_upwell_events_{w}d"].attrs.update(
            {
                "long_name": f"Number of Ekman pumping upwelling events within {w}-days",
                "units": "count",
                "description": f"Daily count of events where Ekman pumping anomaly exceeded the 90th percentile "
                f"threshold from the 1998 to 2017 monthly climatology computed for each grid cell and accumulated within a rolling {w}-day window. "
                f"Values range from 0 (no events) to {w} (all days in the window exceed threshold). Note: values dont represent days but frequency of events.",
            }
        )

    # Remove previous days added before
    ds_ekman = ds_ekman.sel(time=slice(da_dt_ini, da_dt_fin))
    return clip_land_data(ds_ekman)


# ----------------
# ---- Waves ----
# ----------------
def direction_to_uv(da: xr.DataArray) -> xr.Dataset:
    """
    Convert directional variable (degrees) into vector components.

    Parameters
    ----------
    mdts : xr.DataArray
        Mean direction in degrees (0–360).

    Returns
    -------
    xr.Dataset
        Dataset with u, v components (unit vectors).
    """
    radians = np.deg2rad(da)
    u = np.cos(radians)
    v = np.sin(radians)
    ds_daily = xr.Dataset({"u_ts": u, "v_ts": v})
    ds_daily["u_ts"].attrs.update(
        {
            "GRIB_name": "Mean northward wave direction",
            "long_name": "Mean northward wave direction",
            "GRIB_units": "",
            "units": "",
        }
    )
    ds_daily["v_ts"].attrs.update(
        {
            "GRIB_name": "Mean eastward wave direction",
            "long_name": "Mean eastward wave direction",
            "GRIB_units": "",
            "units": "",
        }
    )
    return ds_daily


def daily_waves(
    ds: xr.Dataset,
    swell_height_name: str = "swh",
    swell_direction_name: str = "mdts",
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Compute daily mean significant wave height from hourly data

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with hourly data (must have a time dimension).
    swell_height_name, swell_direction_name: str, optional
        Name of the variables in ds for swell significant height and direction. (default: 'swh' and 'mdts')
    time_dim : str, optional
        Name of the time dimension (default: 'time').

    Returns
    -------
    xr.Dataset
        Dataset with daily mean values for each variable.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Dataset does not have dimension '{time_dim}'")

    da_h = ds[swell_height_name]
    da_d = ds[swell_direction_name]
    out = xr.Dataset(
        {
            swell_height_name: da_h,  # .resample({time_dim: "1D"}).mean(),
            swell_direction_name: da_d,  # .resample({time_dim: "1D"}).mean()
        }
    )
    out = resample_daily_mean(out)
    out.attrs.update(ds.attrs)
    return drop_dims(out)


# -----------------------------------------
# ---- Processing key variables groups ----
# -----------------------------------------


def process_atm_accum_avg(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """A first preprocessing is done in processor.py because data overlap at adjacent days in monthly grib files"""
    ds_ekman = compute_curl_and_ekman(ds)
    datasets = [
        ds_ekman,
        add_engineered_ekman(ds_ekman["ekman_pumping"], var_key=var_key),
        daily_total_rain(ds),
    ]
    # isel to reverse lat values order
    merged = xr.merge(datasets, compat="override", join="outer")
    assert isinstance(merged, xr.Dataset)
    return merged.isel(lat=slice(None, None, -1))


def process_atm_instante(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    datasets = []
    ds = rename_dims(ds)
    ds = ds.chunk(
        {"time": unified_time_chunk(ds), "lat": len(ds.lat), "lon": len(ds.lon)}
    )
    datasets.append(daily_wind(ds))
    datasets.append(daily_cloud_cover(ds))
    datasets.append(daily_sea_level_pressure(ds))
    merged = xr.merge(datasets, compat="override", join="outer")
    assert isinstance(merged, xr.Dataset)
    return merged.isel(lat=slice(None, None, -1))


def process_radiation(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """A first preprocessing is done in processor.py because data overlap at adjacent days in monthly grib files"""
    datasets = []
    for var in ds.data_vars:
        datasets.append(daily_radiation(ds[var]).sortby("time"))
    merged = xr.merge(datasets, compat="override", join="outer")
    assert isinstance(merged, xr.Dataset)
    return merged.isel(lat=slice(None, None, -1))


def process_waves(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    ds = rename_dims(ds)
    ds = ds.chunk(
        {"time": unified_time_chunk(ds), "lat": len(ds.lat), "lon": len(ds.lon)}
    )
    return daily_waves(ds).isel(lat=slice(None, None, -1))
