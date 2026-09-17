"""
Convert-time derived variables, declared per var_key in ``derived_vars``.

Kept out of the per-var_key processors so that a computation is named by its
inputs in config rather than by variable names hardcoded here — the same
kinetic energy serves ``ugos``/``vgos`` in one store and ``uo``/``vo`` in
another.
"""

from __future__ import annotations

from typing import Callable

import xarray as xr

from h2mare.models import DerivedOp, DerivedVarSpec


def rolling_std(da: xr.DataArray, window: int) -> xr.DataArray:
    """
    Standard deviation over a ``window``×``window`` lon/lat box centred on each cell.

    Edge and coastal cells use whatever part of the box holds data
    (``min_periods=1``), so a cell with a single valid neighbour gets 0 and one
    with none stays NaN. Any depth or time axis is left alone.
    """
    missing = {"lon", "lat"} - set(map(str, da.dims))
    if missing:
        raise ValueError(f"rolling_std needs lon/lat dims; '{da.name}' has {da.dims}")
    return da.rolling(lon=window, lat=window, center=True, min_periods=1).std(
        skipna=True
    )


def kinetic_energy(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Kinetic energy per unit mass, ``0.5 * (u**2 + v**2)``."""
    return 0.5 * (u**2 + v**2)


_OPS: dict[DerivedOp, Callable[..., xr.DataArray]] = {
    DerivedOp.ROLLING_STD: lambda srcs, spec: rolling_std(srcs[0], spec.window),
    DerivedOp.KINETIC_ENERGY: lambda srcs, _: kinetic_energy(*srcs),
}


def apply_derived_vars(
    ds: xr.Dataset, derived: dict[str, DerivedVarSpec] | None, owner: str
) -> xr.Dataset:
    """
    Add each declared derived variable to ``ds``, in declaration order.

    Lazy: nothing is computed here. ``owner`` is the var_key named in errors.
    """
    for name, spec in (derived or {}).items():
        missing = [s for s in spec.sources if s not in ds.data_vars]
        if missing:
            raise ValueError(
                f"[{owner}] derived_vars.{name} reads {missing}, which the "
                f"dataset does not hold. Variables: {sorted(map(str, ds.data_vars))}."
            )
        srcs = [ds[s] for s in spec.sources]
        ds[name] = _OPS[spec.op](srcs, spec).rename(name)
    return ds
