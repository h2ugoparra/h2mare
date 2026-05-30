"""Function to process downloaded datasets from CMEMS"""

from __future__ import annotations

from typing import Optional

import xarray as xr

from h2mare.models import KeyVarConfigEntry
from h2mare.processing.core.fronts import FrontProcessor
from h2mare.storage.xarray_helpers import ds_float64_to_float32


def process_ssh(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """Process sea surface height vars: adt and sla _std and gke (geostrophic kinentic energy variables"""

    # Convert o float32 and 1 time step to avoid memory issues
    ds = ds_float64_to_float32(ds).chunk({"time": 1, "lat": -1, "lon": -1})

    adt_da = (
        ds["adt"]
        .rolling(lon=3, lat=3, center=True, min_periods=1)
        .std(skipna=True)
        .compute()
    )
    adt_da.name = "adt_std"

    sla_da = (
        ds["sla"]
        .rolling(lon=3, lat=3, center=True, min_periods=1)
        .std(skipna=True)
        .compute()
    )
    sla_da.name = "sla_std"

    ds_var = xr.merge([ds, adt_da, sla_da], join="outer")
    ds_var["gke"] = (ds_var["ugos"] ** 2 + ds_var["vgos"] ** 2) * 1 / 2
    return ds_var


def process_chl(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """Process chlorophyll dataset"""
    _var = "chl"
    ds = (
        ds.rename_vars({"CHL": _var})
        .astype("float32")
        .chunk({"time": 1, "lat": 500, "lon": 500})
    )
    ds_fdist = FrontProcessor(_var).from_dataset(ds)
    return xr.merge([ds, ds_fdist], join="outer")


def process_sst(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """Process sea surface temperature downloaded dataset"""
    _var = "sst"
    ds = ds.rename_vars({"analysed_sst": _var})
    ds[_var] = (
        (ds[_var] - 273.15).astype("float32").chunk({"time": 1, "lat": 500, "lon": 500})
    )

    # Run front detection process (lazy)
    ds_fdist = FrontProcessor(_var).from_dataset(ds)

    # Calculate sea surface temperature standard deviation
    da_std = (
        ds[_var]
        .rolling(lon=3, lat=3, center=True, min_periods=1)
        .construct(lon="lon_win", lat="lat_win")
        .std(dim=["lon_win", "lat_win"], skipna=True)
        .astype("float32")
    )
    ds[f"{_var}_std"] = da_std
    return xr.merge([ds, ds_fdist], join="outer")


def process_mld(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """Process mixed layer depth dataset"""
    return ds.rename_vars({"mlotst": "mld"})
