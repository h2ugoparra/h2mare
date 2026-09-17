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
    """
    Prepare the sea surface height dataset for its derived variables.

    The std layers and gke are declared in config (``derived_vars``); this
    only casts to float32 and chunks one time step per spatial slab, which
    the rolling windows read across.
    """
    return ds_float64_to_float32(ds).chunk({"time": 1, "lat": -1, "lon": -1})


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

    # Run front detection process (lazy). sst_std is declared in config
    # (derived_vars) and computed after this returns.
    ds_fdist = FrontProcessor(_var).from_dataset(ds)
    return xr.merge([ds, ds_fdist], join="outer")


def process_mld(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """Process mixed layer depth dataset"""
    return ds.rename_vars({"mlotst": "mld"})
