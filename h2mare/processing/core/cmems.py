"""Function to process downloaded datasets from CMEMS"""

from __future__ import annotations

from typing import Optional

import xarray as xr

from h2mare.models import KeyVarConfigEntry
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
    """
    Prepare the chlorophyll dataset for its front-distance layer.

    ``CHL`` → ``chl`` is config's (``source_renames``), applied before this
    runs. ``chl_fdist`` is declared in config too (``boa_fronts``) and detected
    after this returns; the chunking here is what the detection stages from.
    """
    return ds.astype("float32").chunk({"time": 1, "lat": 500, "lon": 500})


def process_sst(
    ds: xr.Dataset,
    var_config: Optional[KeyVarConfigEntry] = None,
    var_key: str | None = None,
) -> xr.Dataset:
    """
    Convert sea surface temperature from kelvin to degrees Celsius.

    ``analysed_sst`` → ``sst`` is config's (``source_renames``), applied before
    this runs. ``sst_std`` and ``sst_fdist`` are declared in config too —
    ``derived_vars`` and ``boa_fronts`` — and computed after this returns.
    """
    _var = "sst"
    ds[_var] = (
        (ds[_var] - 273.15).astype("float32").chunk({"time": 1, "lat": 500, "lon": 500})
    )
    return ds
