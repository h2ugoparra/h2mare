"""
Process downloaded CDS-ERA5 hourly grib data to daily means.

"""

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare import get_settings
from h2mare.models import KeyVarConfigEntry, step_freq
from h2mare.storage.xarray_helpers import rename_dims, unified_time_chunk
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.utils.spatial import clip_land_data

_EKMAN_P90_FILE = "cds_ekman-monthly-90thquantile_80W-10E-0N-70N_1998-2017.nc"
_EKMAN_DOY_FILE = "cds_ekman-doy-mean_80W-10E-0N-70N_1998-2017.nc"

#: Buckets in the day-of-year climatology: one per calendar day of a common
#: year. A file with 366 was built on raw ``dayofyear`` and is misaligned (see
#: :func:`calendar_doy`), so the count doubles as a format check.
_EKMAN_DOY_BUCKETS = 365

#: Time chunk used while the curl stencil runs. Matches the GRIB read the
#: convert path has always used (`chunks={"time": 168, ...}`), which keeps each
#: chunk ~68 MB with lat/lon whole and lets time chunks stream independently.
_CURL_TIME_CHUNK = 168

# Rolling window behind ekman_7d, and the feature depths layered on top of it.
_EKMAN_ROLL_DAYS = 7
_EKMAN_LAGS = (3, 7, 14)
_EKMAN_EVENT_WINDOWS = (3, 7, 14)

# Days of real history that must sit in front of the first output day.
#
# ekman_anom is a 7-day rolling mean minus climatology, so it is only right once
# 7 days are behind it. A lag-N anomaly reads that value at t-N and therefore
# needs N + 7 - 1 days; an N-day event count sums exceedances back to t-(N-1),
# needing (N-1) + 7 - 1. Derived from the tuples above rather than written as a
# literal so adding a deeper lag cannot silently outrun the seed.
#
# Seeding less than this does not fail — rolling(min_periods=1) fills the
# shortfall with a partial window — so the deficit shows up only as wrong values
# in the first weeks of a range.
_EKMAN_WARMUP_DAYS = max(
    max(_EKMAN_LAGS) + _EKMAN_ROLL_DAYS - 1,
    max(_EKMAN_EVENT_WINDOWS) - 1 + _EKMAN_ROLL_DAYS - 1,
)


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


#: Scalar coordinates cfgrib attaches to every ERA5 field, carrying no
#: information once the variable is known.
_GRIB_SCALAR_COORDS = ["step", "number", "surface", "meanSea"]


def drop_scalar_dims(ds: xr.Dataset) -> xr.Dataset:
    """
    Drop the scalar coordinates cfgrib attaches to every ERA5 field.

    Safe to call anywhere, and the name is what carries that guarantee: every
    entry is a scalar, so dropping one can never remove an axis. Coordinates
    that vary along a dimension do not belong here — see :func:`drop_valid_time`
    for what one of those costs.

    Args:
        ds (xr.Dataset): dataset to drop coordinates from

    Returns:
        xr.Dataset: ds without :data:`_GRIB_SCALAR_COORDS`
    """
    return ds.drop_vars(_GRIB_SCALAR_COORDS, errors="ignore")


def drop_valid_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Drop ``valid_time`` once ``time`` carries the axis, leaving it duplication.

    ERA5 arrives in two layouts: ``valid_time`` *as* the time dimension, which
    :func:`rename_dims` turns into ``time``, and ``valid_time`` as a coordinate
    riding along an existing ``time``. Only the second is safe to drop, and the
    difference does not announce itself — dropping a dimension coordinate leaves
    the dimension in place with no labels at all, so every timestamp disappears
    without raising. Hence a guard here rather than a place in
    :data:`_GRIB_SCALAR_COORDS`, whose entries need no precondition.

    Args:
        ds (xr.Dataset): dataset to drop ``valid_time`` from

    Returns:
        xr.Dataset: ds without ``valid_time``, or unchanged if it carries the axis
    """
    if "time" in ds.dims and "valid_time" in ds.coords:
        return ds.drop_vars("valid_time")
    return ds


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
    return drop_scalar_dims(out)


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
    return drop_scalar_dims(out)


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
    return drop_scalar_dims(out)


# -----------------------------
# ---- Radiation features ----
# ----------------------------
#: Rates in [-_RATE_NOISE, 0) are rounding noise in ERA5's accumulation and are
#: flattened to zero. Anything more negative is signal — ``slhf`` is negative
#: over essentially the whole ocean — and is left exactly as it is.
_RATE_NOISE = 1e-6

#: Watts per square metre, in udunits/CF form rather than with a superscript.
#: This string is written into the store and travels on to Parquet and CSV,
#: where a non-ASCII "²" is mangled by any reader that does not assume UTF-8 —
#: Windows' default codepage among them. Every other unit the pipeline writes
#: ("m/s", "hPa", "mm") is already plain ASCII; this was the one exception.
_WATTS_PER_M2 = "W m-2"


def accumulation_period_seconds(da: xr.DataArray, time_dim: str = "time") -> float:
    """
    Seconds each accumulation covers, read off the time axis.

    Taken from the axis rather than hardcoded so a 3-hourly product converts
    correctly too, and as a median so one ragged step cannot skew it.
    """
    times = np.asarray(da[time_dim].values)
    if times.size < 2:
        raise ValueError(
            f"Cannot infer the accumulation period from {times.size} timestep(s) — "
            f"at least two are needed to measure the spacing."
        )
    return float(np.median(np.diff(times) / np.timedelta64(1, "s")))


def hourly_radiation(
    da: xr.DataArray,
    time_dim: str = "time",
    units_out: str = _WATTS_PER_M2,
    clip_small_negatives: bool = True,
) -> xr.DataArray:
    """Convert ERA5's per-interval accumulation (J/m2) to a mean rate (W/m2).

    Each value already covers only the interval ending at its own timestamp, so
    the conversion is a division by that interval and nothing else. It used to
    difference consecutive values first, which silently treated the field as a
    running total. ERA5's own numbers say otherwise: a single forecast block of
    ``tisr`` rises and falls with the sun ::

        0, 0, 0, 149248, 905088, ..., 2423552, 2218112, 1781760, 1144320

    and a running total cannot decrease. Dividing straight through reproduces
    the astronomical top-of-atmosphere insolation to within 1.2% (174.3 vs
    172.3 W/m² at 40°N on 1998-01-15); differencing first gave ~28.

    Args:
        da (xr.DataArray): data array with accumulated radiation data.
        time_dim (str, optional): time dimension name. Defaults to "time".
        units_out (str, optional): Output units. Defaults to
            :data:`_WATTS_PER_M2` ("W m-2").
        clip_small_negatives (bool, optional): Flatten rounding-noise negatives
            to zero. Genuinely negative fluxes are kept. Defaults to True.

    Returns:
        xr.DataArray: Mean rate over each accumulation interval.
    """
    rate = da / accumulation_period_seconds(da, time_dim)

    if clip_small_negatives:
        # Bounded on both sides. The old form was `rate >= -1e-6, 0.0`, which
        # zeroed everything *below* the threshold rather than the sliver just
        # under zero — taking the whole of slhf with it.
        rate = xr.where((rate < 0) & (rate >= -_RATE_NOISE), 0.0, rate)

    rate.attrs.update(da.attrs)
    rate.attrs.update(
        {
            "units": units_out,
            "GRIB_units": units_out,
            # The source says 'accum'; these values are a rate. Left unchanged
            # it invites a reader to difference an already-differenced field.
            "GRIB_stepType": "avg",
            "long_name": f"Mean rate from accumulated {da.name or ''}".strip(),
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
    out = store_hourly_radiation(da, time_dim=time_dim)
    return resample_daily_mean(out)


def store_hourly_radiation(da: xr.DataArray, time_dim: str = "time") -> xr.Dataset:
    """
    Radiation at ERA5's own cadence — :func:`daily_radiation` without the mean.

    Rates rather than the raw J/m² accumulations, unlike the other hourly
    stores. Those keep their source untouched because the source is a physical
    field; here it is a delivery artifact — a per-interval energy total whose
    meaning depends on knowing the interval. W/m² is the quantity, and
    converting once at write time is what lets the compile be a plain daily
    mean and the extractor return something interpretable.
    """
    da = hourly_radiation(da, time_dim=time_dim).astype("float32")
    return xr.Dataset({da.name: da})


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

    # The curl below is a spatial stencil (roll along lat and lon). Tiled
    # spatial chunks make it shuffle across every tile boundary, and because
    # roll is circular the last tile wraps onto the first — coupling opposite
    # ends of each axis into one graph that neither streams nor fits in memory.
    #
    # Reading from GRIB always handed this lat/lon-contiguous, so it never came
    # up. A store written with the "timeseries" layout is tiled by design, so
    # since the ekman chain moved to compile time the layout has to be restored
    # here rather than assumed of the caller.
    if ds.chunks:
        ds = ds.chunk({"time": _CURL_TIME_CHUNK, "lat": -1, "lon": -1})

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
    # Mask near the equator to avoid blow-ups. The mask goes on the *denominator*
    # rather than the result: dividing first and discarding after still evaluates
    # curl/0 at lat=0, and dask raises that as a RuntimeWarning at compute time —
    # far from this line, where a blanket warnings filter would swallow it.
    # Dividing by NaN yields NaN silently, so the values are identical.
    equator_mask = np.abs(f_grid["lat"]) < 2.0
    ekman = curl_tau / (rho_w * f_grid.where(~equator_mask))

    out = xr.Dataset({tx_name: tx, ty_name: ty, "ekman_pumping": ekman})

    out = resample_daily_mean(out)
    out = out.transpose("time", "lat", "lon")
    return out


def get_previous_dates_da(da: xr.DataArray, var_key: str):
    """
    Prepend the stored days preceding ``da`` so the rolling features start warm.

    Fetches :data:`_EKMAN_WARMUP_DAYS` days, which is what the deepest feature
    (``ekman_anom_lag14``) needs. A shorter seed still produces output, because
    ``rolling(min_periods=1)`` accepts a partial window — the values are simply
    wrong for the first weeks, which is invisible downstream.
    """
    da_dt_ini = da.time.values[0]

    repo = ZarrCatalog(var_key)

    date_prev = pd.to_datetime(da_dt_ini) - pd.Timedelta(days=_EKMAN_WARMUP_DAYS)
    ds_prev = repo.open_dataset(
        start_date=date_prev, end_date=da_dt_ini - pd.Timedelta(days=1)
    )
    if ds_prev is not None:
        if isinstance(ds_prev, xr.Dataset):
            # Not GRIB scalars but climatology leftovers, from .sel on the
            # dayofyear/month means — dropped here rather than through
            # drop_scalar_dims, whose list is exactly cfgrib's own.
            ds_prev = ds_prev.drop_vars(
                ["quantile", "month", "dayofyear"], errors="ignore"
            )
            da_prev = ds_prev["ekman_pumping"]
        else:
            da_prev = ds_prev
        # Explicit join: xarray's concat default changes from "outer" to "exact".
        return xr.concat([da, da_prev], dim="time", join="outer").sortby("time")
    else:
        logger.warning("No previous data available. Returning input data array.")
        return da


def calendar_doy(time: xr.DataArray) -> xr.DataArray:
    """
    Day index on a fixed 365-day calendar, so one index is one calendar date.

    ``time.dt.dayofyear`` is not that. It slips by one after February in a leap
    year: 1 March is 60 in three years out of four and 61 in the fourth. A
    climatology grouped on it therefore averages two different calendar dates
    into every post-February bucket, and an anomaly taken against it subtracts
    the wrong day for ten months of every leap year — a quarter of the archive,
    silently, since nothing about the result looks wrong.

    29 February gets 28 February's index rather than one of its own: a twenty
    year baseline holds five of them against twenty samples for every other day,
    and a bucket estimated from a quarter of the data does not belong beside the
    rest. Both the climatology and the anomaly that reads it go through here, so
    the two cannot drift apart.
    """
    month, day = time.dt.month, time.dt.day
    adjusted = time.dt.dayofyear - xr.where(time.dt.is_leap_year & (month > 2), 1, 0)
    return xr.where((month == 2) & (day == 29), 59, adjusted)


def add_engineered_ekman(
    da: xr.DataArray, var_key: str, *, seed_from_store: bool = True
):
    """Compute Ekman pumping related variables.

    Args:
        da: (xarray.DataArray) DataArray with variable 'ekman_pumping'
        var_key: variable key, used to reach the store for warm-up history
        seed_from_store: fetch :data:`_EKMAN_WARMUP_DAYS` of history from the
            stored daily Zarr. The hourly path passes ``False``: it computes
            ``ekman_pumping`` on the fly from an already-widened window, so
            there is no stored daily ``ekman_pumping`` to read and none needed.
    """
    clim_dir = get_settings().CLIMATOLOGY_DIR
    if clim_dir is None:
        raise FileNotFoundError(
            "Directory for Ekman pumping Climatological data not found"
        )

    p90 = xr.open_dataset(clim_dir / _EKMAN_P90_FILE)["ekman_pumping_anom"]
    p90 = p90.chunk({"month": -1, "lat": 200, "lon": 200})

    clim_doy = xr.open_dataset(clim_dir / _EKMAN_DOY_FILE)
    # A 366-bucket file was grouped on raw dayofyear, which pairs each March
    # onward day with the wrong calendar date in leap years. Selecting from it
    # with calendar_doy would still succeed — every index it asks for exists —
    # and quietly return a climatology one day out, so refuse it here instead.
    n_doy = clim_doy.sizes.get("dayofyear")
    if n_doy != _EKMAN_DOY_BUCKETS:
        raise ValueError(
            f"{_EKMAN_DOY_FILE} has {n_doy} dayofyear buckets, expected "
            f"{_EKMAN_DOY_BUCKETS}. A 366-bucket file predates the leap-year "
            f"alignment fix and is offset by a day from March onward in leap "
            f"years. Rebuild it with scripts/ekman_climatology.py."
        )
    clim_doy = clim_doy.chunk({"dayofyear": -1, "lat": 200, "lon": 200})

    # Get previous days for rowling mean
    da_dt_ini = da.time.values[0]
    da_dt_fin = da.time.values[-1]

    if seed_from_store:
        da = get_previous_dates_da(da, var_key)

    # Try to make it more efficient
    da = da.chunk({"time": 30, "lat": 200, "lon": 200})

    # Get 7-day rolling mean of Ekman pumping (Since files are yearly, the first 6days of the year are not complete)
    ekman_7d = da.rolling(time=_EKMAN_ROLL_DAYS, min_periods=1).mean()
    clim_align = clim_doy.sel(dayofyear=calendar_doy(ekman_7d["time"]))
    anom = ekman_7d - clim_align

    ds_ekman = xr.Dataset({"ekman_7d": ekman_7d, "ekman_anom": anom["ekman_pumping"]})

    # Create lag anomalies
    for lag in _EKMAN_LAGS:
        ds_ekman[f"ekman_anom_lag{lag}"] = ds_ekman["ekman_anom"].shift(time=lag)

    # Align 90th percentile climatology with the time axis
    p90_aligned = p90.sel(month=ds_ekman["time"].dt.month)

    # Exceedances: anomaly > local monthly p90
    exceed = ds_ekman["ekman_anom"] > p90_aligned

    # Rolling counts for 3, 7, 14 days.
    #
    # min_periods=w, not 1: a partial window returns a count over however many
    # days exist, which is indistinguishable downstream from a genuine low
    # count. The lagged anomalies above already report missing history as NaN
    # (shift fills the leading positions), so this keeps the two families
    # honest in the same way. Everywhere the range is warmed the NaN edge falls
    # inside the warm-up and is trimmed off; it survives only where there is
    # genuinely no prior data, i.e. the first days of the archive.
    for w in _EKMAN_EVENT_WINDOWS:
        ds_ekman[f"n_upwell_events_{w}d"] = exceed.rolling(time=w, min_periods=w).sum()

    # Remove previous days added before
    ds_ekman = ds_ekman.sel(time=slice(da_dt_ini, da_dt_fin))
    return clip_land_data(ds_ekman)


# ----------------
# ---- Waves ----
# ----------------
def uv_to_direction(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """
    Recombine unit-vector components into a direction in degrees (0–360).

    Inverse of :func:`direction_to_uv`. Any averaging or interpolation of a
    direction has to go through the components: degrees wrap, so the arithmetic
    mean of 350° and 10° is 180° — the opposite heading — and a linear
    interpolation between them sweeps the long way round.

    Known and accepted: when the averaged directions nearly cancel, the
    resultant approaches zero and this returns 0° (north) rather than "no
    coherent direction". Rare for real swell, and the alternatives — nulling
    below a resultant-length threshold, or carrying the length as a coherence
    column — both change what downstream sees, so the convention stands.
    """
    return np.rad2deg(np.arctan2(v, u)) % 360


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

    # Height is a magnitude and averages directly. Direction does not: it is
    # degrees on a circle, so the daily mean is taken over unit vectors and
    # recombined. A plain mean of 350° and 10° gives 180°, the opposite heading.
    out = resample_daily_mean(xr.Dataset({swell_height_name: ds[swell_height_name]}))
    components = (
        direction_to_uv(ds[swell_direction_name]).resample({time_dim: "1D"}).mean()
    )
    out[swell_direction_name] = uv_to_direction(components["u_ts"], components["v_ts"])
    out[swell_direction_name].attrs.update(ds[swell_direction_name].attrs)
    out.attrs.update(ds.attrs)
    return drop_scalar_dims(out)


# -----------------------------------------
# ---- Processing key variables groups ----
# -----------------------------------------


#: Raw ERA5 fields kept as-is by the hourly store. Everything else this variable
#: publishes (ekman_*, n_upwell_events_*, daily tp) is derived at compile time.
_ATM_ACCUM_HOURLY_VARS = ("avg_iews", "avg_inss", "tp")


def store_hourly_atm_accum(ds: xr.Dataset) -> xr.Dataset:
    """
    Keep ERA5's native hourly stress and precipitation, deriving nothing.

    The ekman chain cannot live here: it is daily by construction (a 7-day
    rolling mean, day-lagged anomalies, per-day event counts), and computing it
    at convert time is what forced the store to be daily. With ``time_step:
    hourly`` those features move to ``compiler_registry._compile_atm_accum_avg``.

    Left unclipped and in native units — ``compute_curl_and_ekman`` clips land
    itself, and ``daily_total_rain`` does the m→mm conversion, so both still see
    exactly the input they saw when they ran at convert time.
    """
    ds = rename_dims(ds)
    ds = ds.chunk(
        {"time": unified_time_chunk(ds), "lat": len(ds.lat), "lon": len(ds.lon)}
    )
    keep = [v for v in _ATM_ACCUM_HOURLY_VARS if v in ds.data_vars]
    return ds[keep].isel(lat=slice(None, None, -1))


def process_atm_accum_avg(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """A first preprocessing is done in processor.py because data overlap at adjacent days in monthly grib files"""
    if step_freq(var_config) == "h":
        return store_hourly_atm_accum(ds)

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


#: Raw ERA5 fields kept as-is by the hourly store. The daily reductions this
#: variable publishes (wind_mean/std/max and the daily means) are derived at
#: compile time instead — see ``compiler_registry._compile_atm_instante``.
_ATM_INSTANTE_HOURLY_VARS = ("msl", "u10", "v10", "tcc")


def hourly_atm_instante(ds: xr.Dataset) -> xr.Dataset:
    """
    Keep ERA5's native hourly wind, cloud and pressure, deriving nothing.

    Left in native units — ``daily_sea_level_pressure`` does the Pa→hPa
    conversion, so it still sees exactly the input it saw at convert time. The
    store therefore holds ``msl`` in Pa while h2ds holds it in hPa.

    GRIB's own coordinates have to be dropped explicitly here. The daily path
    sheds them twice over — every ``daily_*`` builder ends in
    ``drop_scalar_dims``, and the resample discards ``valid_time`` on the way
    through — while nothing at all sheds them on a cadence that does neither.
    ``atm-accum-avg`` gets away without this only because ``merge_time_step``
    already dropped them upstream, and atm-instante has no such step.
    """
    keep = [v for v in _ATM_INSTANTE_HOURLY_VARS if v in ds.data_vars]
    return drop_valid_time(drop_scalar_dims(ds[keep]))


def process_atm_instante(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """
    Prepare the instantaneous atmospheric fields, aggregating to daily only for
    a daily store.

    With ``time_step: hourly`` the daily reductions are skipped here and happen
    at compile instead (see ``compiler_registry._compile_atm_instante``), so the
    store keeps ERA5's native hourly axis while h2ds stays daily.
    """
    ds = rename_dims(ds)
    ds = ds.chunk(
        {"time": unified_time_chunk(ds), "lat": len(ds.lat), "lon": len(ds.lon)}
    )
    if step_freq(var_config) == "h":
        merged: xr.Dataset = hourly_atm_instante(ds)
    else:
        datasets = [daily_wind(ds), daily_cloud_cover(ds), daily_sea_level_pressure(ds)]
        merged = xr.merge(datasets, compat="override", join="outer")
    assert isinstance(merged, xr.Dataset)
    return merged.isel(lat=slice(None, None, -1))


def process_radiation(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """
    Prepare the radiation fields, averaging to daily only for a daily store.

    A first preprocessing is done in processor.py because data overlap at
    adjacent days in monthly grib files.

    With ``time_step: hourly`` the daily mean is skipped here and happens at
    compile instead (see ``compiler_registry.compile_default``, which reduces
    any hourly store it is handed), so the store keeps ERA5's native hourly axis
    while h2ds stays daily. Both cadences share the J/m²→W/m² conversion, so
    they publish the same units.
    """
    build = store_hourly_radiation if step_freq(var_config) == "h" else daily_radiation
    datasets = [build(ds[var]).sortby("time") for var in ds.data_vars]
    merged = xr.merge(datasets, compat="override", join="outer")
    assert isinstance(merged, xr.Dataset)
    # No-ops today, since merge_time_step already sheds these upstream. Stated
    # anyway so the store does not depend on that having run — which is the gap
    # that let GRIB's coords reach the hourly atm-instante store.
    merged = drop_valid_time(drop_scalar_dims(merged))
    return merged.isel(lat=slice(None, None, -1))


def hourly_waves(
    ds: xr.Dataset,
    swell_height_name: str = "swh",
    swell_direction_name: str = "mdts",
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Wave fields at the source's own cadence — :func:`daily_waves` without the
    resample.

    Deliberately a sibling rather than a flag on ``daily_waves``: the daily path
    feeds every existing store and h2ds column, so it is left byte-identical.

    ``valid_time`` needs dropping here for the same reason it does on the hourly
    atm-instante path: with no resample to shed it, it reaches the store. The
    existing waves store carries it and is not worth reconverting over 70 kB a
    year, so this only takes effect the next time waves is converted.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Dataset does not have dimension '{time_dim}'")

    out = xr.Dataset(
        {
            swell_height_name: ds[swell_height_name],
            swell_direction_name: ds[swell_direction_name],
        }
    )
    out.attrs.update(ds.attrs)
    return drop_valid_time(drop_scalar_dims(out))


def process_waves(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """
    Prepare the wave fields, aggregating to daily only for a daily store.

    With ``time_step: hourly`` the resample is skipped here and happens at
    compile instead (see ``compiler_registry._compile_waves``), so the store
    keeps ERA5's native hourly axis while h2ds stays daily.
    """
    ds = rename_dims(ds)
    ds = ds.chunk(
        {"time": unified_time_chunk(ds), "lat": len(ds.lat), "lon": len(ds.lon)}
    )
    build = hourly_waves if step_freq(var_config) == "h" else daily_waves
    return build(ds).isel(lat=slice(None, None, -1))
